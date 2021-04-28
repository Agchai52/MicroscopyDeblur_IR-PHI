from __future__ import print_function  # help to use print() in python 2.x
import os
from math import log10
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import *
from network import *
from Dataset import DeblurDataset


def train(args):
    print('===> Loading datasets')
    f_train = open("./dataset/train_instance_names.txt", "r")
    f_test = open("./dataset/test_instance_names.txt", "r")

    train_data = f_train.readlines()
    test_data = f_test.readlines()

    f_train.close()
    f_test.close()

    train_data = [line.rstrip() for line in train_data]
    test_data = [line.rstrip() for line in test_data]

    train_data_loader = DataLoader(DeblurDataset(train_data, args), batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(DeblurDataset(test_data, args, False), batch_size=1, shuffle=False)

    if torch.cuda.device_count() >= 1:
        device = torch.device("cuda")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cpu")

    model_G = Generator(args, device)
    model_G = nn.DataParallel(model_G)

    print('===> Building models')
    netG = model_G.to(device)
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    net_g_path = "checkpoint/netG"

    if not find_latest_model(net_g_path):
        print(" [!] Load failed...")
        netG.apply(weights_init)
        pre_epoch = 0
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)

        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        optimizer_G.load_state_dict(checkpointG['optimizer_state_dict'])

        pre_epoch = checkpointG['epoch']

    netG.train()
    print(netG)

    # Gemometric Blur Model as Second Generator
    netG_S2B = BlurModel(args, device)

    print('===> Setting up loss functions')
    criterion_L2 = nn.MSELoss().to(device)
    criterion_grad = GradientLoss(device=device).to(device)

    counter = 0
    PSNR_average = []
    SSIM_average = []

    loss_record = "loss_record.txt"
    psnr_record = "psnr_record.txt"
    ssim_record = "ssim_record.txt"

    print('===> Training')
    print('Start from epoch: ', pre_epoch)
    for epoch in range(pre_epoch, args.epoch):
        for iteration, batch in enumerate(train_data_loader, 1):
            real_B, real_S, img_name = batch[0], batch[1], batch[2]
            real_B, real_S = real_B.to(device), real_S.to(device)

            threshold = -0.3
            max_v = 1.0 * torch.ones_like(real_B)
            min_v = -1.0 * torch.ones_like(real_B)
            roi_B = torch.where(real_B <= threshold, min_v, max_v)
            real_B = roi_B

            fake_S = netG(real_B)
            recov_B = netG_S2B(fake_S)

            ############################
            # (1) Update G network:
            ###########################
            optimizer_G.zero_grad()

            loss_l2 = criterion_L2(fake_S, real_S) * args.L2_lambda
            loss_grad = criterion_grad(fake_S, real_S) * args.L2_lambda
            loss_recover = criterion_L2(recov_B, real_B) * args.LR_lambda

            loss_g = loss_l2 + loss_grad + loss_recover

            loss_g.backward()
            optimizer_G.step()

            counter += 1

            print(
                "===> Epoch[{}]({}/{}): Loss_G: {:.4f} Loss_L2: {:.4f} Loss_Recover: {:.4f}".format(
                    epoch, iteration, len(train_data_loader),
                    loss_grad.item(), loss_l2.item(), loss_recover.item()))

            # To record losses in a .txt file
            losses_dg = [loss_grad.item(), loss_l2.item(), loss_recover.item()]
            losses_dg_str = " ".join(str(v) for v in losses_dg)

            with open(loss_record, 'a+') as file:
                file.writelines(losses_dg_str + "\n")

            if (counter % 500 == 1) or ((epoch == args.epoch - 1) and (iteration == len(train_data_loader) - 1)):
                net_g_save_path = net_g_path + "/G_model_epoch_{}.pth".format(epoch+1)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict()
                }, net_g_save_path)

                print("Checkpoint saved to {}".format("checkpoint/"))

        # Update Learning rate
        #lr_scheduler_G.step()

        if args.save_intermediate:
            all_psnr = []
            all_ssim = []
            with torch.no_grad():
                for batch in test_data_loader:
                    real_B, real_S, img_name = batch[0], batch[1], batch[2]
                    real_B, real_S = real_B.to(device), real_S.to(device)  # B = (B, 1, 64, 64), S = (B, 1, 256, 256)
                    pred_S = netG(real_B)
                    cur_psnr, cur_ssim = compute_metrics(real_S, pred_S)
                    all_psnr.append(cur_psnr)
                    all_ssim.append(cur_ssim)
                    if img_name[0][-2:] == '01':
                        img_S = pred_S.detach().squeeze(0).cpu()
                        save_img(img_S, '{}/test_'.format(args.valid_dir) + img_name[0])
                        print('test_{}: PSNR = {} dB, SSIM = {}'.format(img_name[0], cur_psnr, cur_ssim))

                PSNR_average.append(sum(all_psnr) / len(test_data_loader))
                SSIM_average.append(sum(all_ssim) / len(test_data_loader))
                with open(psnr_record, 'a+') as file:
                    file.writelines(str(sum(all_psnr) / len(test_data_loader)) + "\n")
                with open(ssim_record, 'a+') as file:
                    file.writelines(str(sum(all_ssim) / len(test_data_loader)) + "\n")
                print("===> Avg. PSNR: {:.4f} dB".format(sum(all_psnr) / len(test_data_loader)))

    if args.save_intermediate:
        print("===> Average Validation PSNR for each epoch")
        print(PSNR_average)

    print("===> Saving Losses")
    plot_losses()
    print("===> Training finished")


