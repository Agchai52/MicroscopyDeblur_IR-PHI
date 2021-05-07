from __future__ import print_function
import os
import time
import torch
import torchvision.transforms as transforms
from Dataset import DeblurDataset
from torch.utils.data import DataLoader

from utils import *
from network import *
from Dataset import DeblurDataset, RealImage


def test(args):
    if torch.cuda.device_count() >= 1:
        device = torch.device("cuda")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cpu")

    model_G = Generator(args, device)
    model_G = nn.DataParallel(model_G)

    model_D = Detector(args, device)
    model_D = nn.DataParallel(model_D)

    print('===> Loading models')
    netG = model_G.to(device)
    net_g_path = "checkpoint/netG"

    netD = model_D.to(device)
    net_d_path = "checkpoint/netD"

    if not find_latest_model(net_g_path) or not find_latest_model(net_d_path):
        print(" [!] Load failed...")
        raise Exception('No model to load for testing!')
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        netG.eval()

        model_path_D = find_latest_model(net_d_path)
        checkpointD = torch.load(model_path_D)
        netD.load_state_dict(checkpointD['model_state_dict'])
        netD.eval()

    print("====> Loading data")
    ############################
    # For DeblurMicroscope dataset
    ###########################
    f_test = open("./dataset/test_instance_names.txt", "r")
    test_data = f_test.readlines()
    test_data = [line.rstrip() for line in test_data]
    f_test.close()
    test_data_loader = DataLoader(DeblurDataset(test_data, args, False), batch_size=1, shuffle=False)

    ############################
    # For Other datasets
    ###########################
    # image_dir = "dataset/{}/test/a/".format(args.dataset_name)
    # image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
    # transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # transform = transforms.Compose(transform_list)
    # for image_name in image_filenames:

    all_psnr = []
    all_ssim = []
    start_time = time.time()

    # Gemometric Blur Model as Second Generator
    netG_S2B = BlurModel(args, device)
    with torch.no_grad():
        for batch in test_data_loader:
            real_B, real_S, label, img_name = batch[0], batch[1], batch[2], batch[3]
            real_B, real_S, label = real_B.to(device), real_S.to(device), label.to(device)
            # B = (B, 1, 64, 64), S = (B, 1, 256, 256)

            pred_S = netG(real_B)
            recov_B = netG_S2B(pred_S[-1])

            pred_S = pred_S[-1]

            pred_label = netD(pred_S)
            real_label = netD(real_S)

            cur_psnr, cur_ssim = compute_metrics(real_S, pred_S)
            all_psnr.append(cur_psnr)
            all_ssim.append(cur_ssim)

            if img_name[0][-2:] == '01':

                img_roi = pred_label.detach().squeeze(0).cpu()
                img_roi = (img_roi * 2 - 1.)
                save_img(img_roi, '{}/roi_'.format(args.test_dir) + img_name[0])

                img_roi = real_label.detach().squeeze(0).cpu()
                img_roi = (img_roi * 2 - 1.)
                save_img(img_roi, '{}/roi0_'.format(args.test_dir) + img_name[0])


                img_S = pred_S.detach().squeeze(0).cpu()
                save_img(img_S, '{}/test_'.format(args.test_dir) + img_name[0])

                img_S = recov_B.squeeze(0).cpu()
                save_img(img_S, '{}/recover_'.format(args.test_dir) + img_name[0])

                img_S = real_B.squeeze(0).cpu()
                save_img(img_S, '{}/blur_'.format(args.test_dir) + img_name[0])

                img_S = real_S.squeeze(0).cpu()
                save_img(img_S, '{}/sharp_'.format(args.test_dir) + img_name[0])
                exit()
                print('test_{}: PSNR = {} dB, SSIM = {}'
                      .format(img_name[0], cur_psnr, cur_ssim))

    total_time = time.time() - start_time
    ave_psnr = sum(all_psnr) / len(test_data_loader)
    ave_ssim = sum(all_ssim) / len(test_data_loader)
    ave_time = total_time / len(test_data_loader)
    print("Average PSNR = {}, SSIM = {}, Processing time = {}".format(ave_psnr, ave_ssim, ave_time))


def test_real(args):
    if torch.cuda.device_count() >= 1:
        device = torch.device("cuda")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cpu")

    model_G = Generator(args, device)
    model_G = nn.DataParallel(model_G)

    model_D = Detector(args, device)
    model_D = nn.DataParallel(model_D)

    print('===> Loading models')
    netG = model_G.to(device)
    net_g_path = "checkpoint/netG"

    netD = model_D.to(device)
    net_d_path = "checkpoint/netD"

    if not find_latest_model(net_g_path) or not find_latest_model(net_d_path):
        print(" [!] Load failed...")
        raise Exception('No model to load for testing!')
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        netG.eval()

        model_path_D = find_latest_model(net_d_path)
        checkpointD = torch.load(model_path_D)
        netD.load_state_dict(checkpointD['model_state_dict'])
        netD.eval()

    print("====> Loading data")
    ############################
    # For Real Images
    ###########################
    image_dir = "dataset/{}/".format("real_images")
    image_filenames = [image_dir + x[0:-4] for x in os.listdir(image_dir) if x[-4:] in set([".png", ".jpg"])]
    test_data_loader = DataLoader(RealImage(image_filenames, args, False), batch_size=1, shuffle=False)

    start_time = time.time()
    with torch.no_grad():
        for batch in test_data_loader:
            real_B, img_name = batch[0], batch[1]
            real_B = real_B.to(device)

            pred_S = netG(real_B)

            pred_label = netD(pred_S[-1])

            img_roi = pred_label.detach().squeeze(0).cpu()
            img_roi = (img_roi * 2 - 1.)
            save_img(img_roi, '{}/roi_'.format(args.valid_dir) + img_name[0])

            img_S = pred_S[2].detach().squeeze(0).cpu()
            save_img(img_S, '{}/real_'.format(args.test_dir) + img_name[0])

            img_S = pred_S[1].detach().squeeze(0).cpu()
            save_img(img_S, '{}/real1_'.format(args.test_dir) + img_name[0])

            img_S1 = F.interpolate(pred_S[1], (args.fine_size * 4, args.fine_size * 4), mode="bilinear")
            img_S1 = img_S1.detach().squeeze(0).cpu()
            save_img(img_S1, '{}/interp1_'.format(args.test_dir) + img_name[0])

            img_S = pred_S[0].detach().squeeze(0).cpu()
            save_img(img_S, '{}/real0_'.format(args.test_dir) + img_name[0])

            img_S1 = F.interpolate(pred_S[0], (args.fine_size * 2, args.fine_size * 2), mode="bilinear")
            img_S1 = img_S1.detach().squeeze(0).cpu()
            save_img(img_S1, '{}/interp0_'.format(args.test_dir) + img_name[0])
            # print("Image Name: {}, predict number = {}, score = {}".format(img_name[0], pre_num + 1, score))

    total_time = time.time() - start_time
    ave_time = total_time / len(test_data_loader)
    print("Processing time = {}".format(ave_time))