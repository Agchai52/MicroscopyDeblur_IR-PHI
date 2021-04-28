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

    model_DS = Discriminator(args, device)
    model_DS = nn.DataParallel(model_DS)

    print('===> Building models')
    netG = model_G.to(device)
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    net_g_path = "checkpoint/netG"

    netD_S = model_DS.to(device)
    optimizer_D_S = optim.Adam(netD_S.parameters(), lr=args.lr, betas=(args.beta1, 0.999), amsgrad=True)
    net_d_s_path = "checkpoint/netD_S"

    if not find_latest_model(net_g_path):
        print(" [!] Load failed...")
        netG.apply(weights_init)
        netD_S.apply(weights_init)
        pre_epoch = 0
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        optimizer_G.load_state_dict(checkpointG['optimizer_state_dict'])

        model_path_D_S = find_latest_model(net_d_s_path)
        checkpointDS = torch.load(model_path_D_S)
        netD_S.load_state_dict(checkpointDS['model_state_dict'])
        optimizer_D_S.load_state_dict(checkpointDS['optimizer_state_dict'])

        pre_epoch = checkpointG['epoch']

    netG.train()
    netD_S.train()
    print(netG)

    # Gemometric Blur Model as Second Generator
    netG_S2B = BlurModel(args, device)

    print('===> Setting up loss functions')
    criterion_L2 = nn.MSELoss().to(device)
    criterion_grad = GradientLoss(device=device).to(device)
    criterion_GAN = GANLoss().to(device)

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

            fake_S = netG(real_B)
            fake_B = netG_S2B(real_S)

            fake_B = F.interpolate(fake_B, (args.fine_size, args.fine_size), mode="bilinear")

            recov_B = netG_S2B(fake_S)
            recov_S = netG(fake_B)
            ############################
            # (1) Update D_S network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizer_D_S.zero_grad()

            # train with fake
            pred_fake_S = netD_S(fake_S.detach())
            loss_d_s_fake = criterion_GAN(pred_fake_S, False)

            # train with real
            pred_real_S = netD_S(real_S)
            loss_d_s_real = criterion_GAN(pred_real_S, True)

            # combine d loss
            loss_d_s = (loss_d_s_fake + loss_d_s_real)

            loss_d_s.backward()
            optimizer_D_S.step()

            ############################
            # (1) Update G network:
            ###########################
            optimizer_G.zero_grad()

            real_B = F.interpolate(real_B, (args.load_size, args.load_size), mode="bilinear")


            # S = G(B) should fake the discriminator S
            pred_fake_S = netD_S(fake_S)
            loss_g_gan_bs = criterion_GAN(pred_fake_S, True)

            loss_l2 = criterion_L2(fake_S, real_S) * args.L2_lambda
            loss_grad = criterion_grad(fake_S, real_S) * args.L2_lambda
            loss_recover = (criterion_L2(recov_B, real_B) + criterion_L2(recov_S, real_S)) * args.L2_lambda
            # loss_recover = criterion_L2(recov_B, real_B) * args.LR_lambda

            loss_g = loss_l2 + loss_grad + loss_recover + loss_g_gan_bs

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
                    # pred_S = F.interpolate(pred_S, (args.load_size, args.load_size), mode='bilinear')
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


