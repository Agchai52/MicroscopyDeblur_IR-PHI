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

    model_D = Classifier(args, device)
    model_D = nn.DataParallel(model_D)

    print('===> Building models')
    netG = model_G.to(device)
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    net_g_path = "checkpoint/netG"

    netD = model_D.to(device)
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    net_d_path = "checkpoint/netD"

    if not find_latest_model(net_g_path) or not find_latest_model(net_d_path):
        print(" [!] Load failed...")
        netG.apply(weights_init)
        netD.apply(weights_init)
        pre_epoch = 0
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        optimizer_G.load_state_dict(checkpointG['optimizer_state_dict'])

        model_path_D = find_latest_model(net_d_path)
        checkpointD = torch.load(model_path_D)
        netD.load_state_dict(checkpointD['model_state_dict'])
        optimizer_D.load_state_dict(checkpointD['optimizer_state_dict'])

        pre_epoch = checkpointG['epoch']

    netG.train()
    netD.train()
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
            real_B, real_S, label, img_name = batch[0], batch[1], batch[2], batch[3]
            real_B, real_S, label = real_B.to(device), real_S.to(device), label.to(device)  # (b, 1, 64, 64)  # (b, 1, 64, 64)

            fake_S = netG(real_B)  # (64, 64) -> [0](64, 64) [1](128, 128) [2](256, 256)

            recov_B = netG_S2B(fake_S[-1])

            real_S0 = F.interpolate(real_S, (args.fine_size * 1, args.fine_size * 1), mode="bilinear")
            real_S1 = F.interpolate(real_S, (args.fine_size * 2, args.fine_size * 2), mode="bilinear")
            real_S2 = real_S  # (256, 256)
            ############################
            # (1) Update G network:
            ###########################
            optimizer_G.zero_grad()

            loss_l2 = (criterion_L2(fake_S[0], real_S0) +
                       criterion_L2(fake_S[1], real_S1) +
                       criterion_L2(fake_S[2], real_S2)) * args.L2_lambda / 3
            loss_grad = (criterion_grad(fake_S[0], real_S0) +
                         criterion_grad(fake_S[1], real_S1) +
                         criterion_grad(fake_S[2], real_S2)) * args.L2_lambda / 3

            loss_recover = criterion_L2(recov_B[0], real_B) * args.L2_lambda * 2

            loss_g = loss_l2 + loss_grad

            loss_g.backward()
            optimizer_G.step()

            counter += 1

            print(
                "===> Epoch[{}]({}/{}): Loss_Grad: {:.4f} Loss_L2: {:.4f} Loss_Recover: {:.4f} ".format(
                    epoch, iteration, len(train_data_loader),
                    loss_grad.item(), loss_l2.item(), loss_recover.item()))

            # To record losses in a .txt file
            losses_dg = [loss_grad.item(), loss_l2.item(), loss_recover.item()]
            losses_dg_str = " ".join(str(v) for v in losses_dg)

            with open(loss_record, 'a+') as file:
                file.writelines(losses_dg_str + "\n")

            if (counter % 500 == 1) or ((epoch == args.epoch - 1) and (iteration == len(train_data_loader) - 1)):
                net_g_save_path = net_g_path + "/G_model_epoch_{}.pth".format(epoch + 1)
                net_d_save_path = net_d_path + "/D_model_epoch_{}.pth".format(epoch + 1)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict()
                }, net_g_save_path)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': netD.state_dict(),
                    'optimizer_state_dict': optimizer_D.state_dict()
                }, net_d_save_path)

                print("Checkpoint saved to {}".format("checkpoint/"))

        # Update Learning rate
        #lr_scheduler_G.step()

        if args.save_intermediate:
            all_psnr = []
            all_ssim = []
            with torch.no_grad():
                for batch in test_data_loader:
                    real_B, real_S, label, img_name = batch[0], batch[1], batch[2], batch[3]
                    real_B, real_S, label = real_B.to(device), real_S.to(device), label.to(device)
                    # B = (B, 1, 64, 64), S = (B, 1, 256, 256)

                    pred_S = netG(real_B)
                    pred_S = pred_S[-1]

                    pred_label = netD(pred_S)

                    cur_psnr, cur_ssim = compute_metrics(real_S, pred_S)
                    all_psnr.append(cur_psnr)
                    all_ssim.append(cur_ssim)
                    if img_name[0][-2:] == '01':
                        img_S = pred_S.detach().squeeze(0).cpu()
                        img_roi = pred_label.detach().squeeze(0).cpu()
                        img_roi = (img_roi*2-1.)
                        save_img(img_roi, '{}/roi_'.format(args.valid_dir) + img_name[0])

                        save_img(img_S, '{}/test_'.format(args.valid_dir) + img_name[0])
                        print('test_{}: PSNR = {} dB, SSIM = {}'
                              .format(img_name[0], cur_psnr, cur_ssim))

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

