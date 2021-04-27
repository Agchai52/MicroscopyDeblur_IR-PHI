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
    else:
        device = torch.device("cpu")
    model_G = Generator(args, device)
    model_R = ROINet(args, device)

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_G = nn.DataParallel(model_G)
        model_R = nn.DataParallel(model_R)

    print('===> Building models')
    net_g_path = "checkpoint/netG"
    net_r_path = "checkpoint/netR"

    netG = model_G.to(device)
    netR = model_R.to(device)

    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    optimizer_R = optim.Adam(netR.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)

    if not find_latest_model(net_g_path) or not find_latest_model(net_r_path):
        print(" [!] Load failed...")
        netG.apply(weights_init)
        netR.apply(weights_init)
        pre_epoch = 0
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        model_path_R = find_latest_model(net_r_path)

        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        optimizer_G.load_state_dict(checkpointG['optimizer_state_dict'])

        checkpointR = torch.load(model_path_R)
        netR.load_state_dict(checkpointR['model_state_dict'])
        optimizer_R.load_state_dict(checkpointR['optimizer_state_dict'])

        pre_epoch = checkpointG['epoch']

    netG.train()
    netR.train()

    print(netG)
    print(netR)

    # Gemometric Blur Model as Second Generator
    netG_S2B = BlurModel(args, device)

    print('===> Setting up loss functions')
    criterion_L2 = nn.MSELoss().to(device)

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

            roi_B = netR(real_B)
            fake_S = netG(real_B, roi_B.detach())

            recov_B = netG_S2B(fake_S)
            real_B_ = netG_S2B(real_S)
            ############################
            # (1) Update ROI network:
            ###########################
            optimizer_R.zero_grad()

            roi_B_interp = F.interpolate(roi_B, (args.load_size, args.load_size), mode="bilinear")

            roi_B_real = torch.where(rea_B_ > -0.4, real_B_, -1.0)
            roi_B_real = torch.where(roi_B_real <= -0.4, roi_B_real, 1.0)
            loss_roi = criterion_L2(roi_B_interp, roi_B_real)

            loss_roi.backward()
            optimizer_R.step()
            ############################
            # (2) Update G network:
            ###########################
            optimizer_G.zero_grad()

            loss_l2 = criterion_L2(fake_S, real_S) * args.L2_lambda
            loss_recover = criterion_L2(recov_B, real_B_) * args.LR_lambda

            loss_g = loss_l2 + loss_recover

            loss_g.backward()
            optimizer_G.step()

            counter += 1

            print(
                "===> Epoch[{}]({}/{}): Loss_ROI: {:.4f} Loss_L2: {:.4f} Loss_Recover: {:.4f}".format(
                    epoch, iteration, len(train_data_loader),
                    loss_roi.item(), loss_l2.item(), loss_recover.item()))

            # To record losses in a .txt file
            losses_dg = [loss_roi.item(), loss_l2.item(), loss_recover.item()]
            losses_dg_str = " ".join(str(v) for v in losses_dg)

            with open(loss_record, 'a+') as file:
                file.writelines(losses_dg_str + "\n")

            if (counter % 500 == 1) or ((epoch == args.epoch - 1) and (iteration == len(train_data_loader) - 1)):
                net_g_save_path = net_g_path + "/G_model_epoch_{}.pth".format(epoch+1)
                net_r_save_path = net_r_path + "/R_model_epoch_{}.pth".format(epoch+1)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict()
                }, net_g_save_path)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': netR.state_dict(),
                    'optimizer_state_dict': optimizer_R.state_dict()
                }, net_r_save_path)

                print("Checkpoint saved to {}".format("checkpoint/"))

        # Update Learning rate
        #lr_scheduler_G.step()
        #lr_scheduler_D.step()

        if args.save_intermediate:
            all_psnr = []
            all_ssim = []
            with torch.no_grad():
                for batch in test_data_loader:
                    real_B, real_S, img_name = batch[0], batch[1], batch[2]
                    real_B, real_S = real_B.to(device), real_S.to(device)  # B = (B, 1, 64, 64), S = (B, 1, 256, 256)
                    roi_B = netR(real_B)
                    pred_S = netG(real_B, roi_B)
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


