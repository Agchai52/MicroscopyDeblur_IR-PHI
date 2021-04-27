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
    else:
        device = torch.device("cpu")

    model_G = Generator(args, device)
    model_R = ROINet(args, device)

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_G = nn.DataParallel(model_G)
        model_R = nn.DataParallel(model_R)

    print("====> Loading model")
    net_g_path = "checkpoint/netG"
    net_r_path = "checkpoint/netR"

    netG = model_G.to(device)
    netR = model_R.to(device)

    if not find_latest_model(net_g_path) or not find_latest_model(net_r_path):
        print(" [!] Load failed...")
        raise Exception('No model to load for testing!')
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        netG.eval()

        model_path_R = find_latest_model(net_r_path)
        checkpointR = torch.load(model_path_R)
        netR.load_state_dict(checkpointR['model_state_dict'])
        netR.eval()

    netG_S2B = BlurModel(args, device)
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
    with torch.no_grad():
        for batch in test_data_loader:
            real_B, real_S, img_name = batch[0], batch[1], batch[2]
            real_B, real_S = real_B.to(device), real_S.to(device)  # B = (B, 1, 64, 64), S = (B, 1, 256, 256)
            roi_B = netR(real_B)
            pred_S = netG(real_B, roi_B)

            real_B_ = netG_S2B(real_S)
            threshold = -0.3
            max_v = 1.0 * torch.ones_like(real_B_)
            min_v = -1.0 * torch.ones_like(real_B_)
            roi_B_real = torch.where(real_B_ <= threshold, min_v, max_v)

            cur_psnr, cur_ssim = compute_metrics(real_S, pred_S)
            all_psnr.append(cur_psnr)
            all_ssim.append(cur_ssim)
            if img_name[0][-2:] == '01':
                img_S = pred_S.detach().squeeze(0).cpu()
                img_R = roi_B.detach().squeeze(0).cpu()
                save_img(img_S, '{}/test_'.format(args.test_dir) + img_name[0])
                save_img(img_R, '{}/roi_'.format(args.test_dir) + img_name[0])
                print('test_{}: PSNR = {} dB, SSIM = {}'.format(img_name[0], cur_psnr, cur_ssim))

    total_time = time.time() - start_time
    ave_psnr = sum(all_psnr) / len(test_data_loader)
    ave_ssim = sum(all_ssim) / len(test_data_loader)
    ave_time = total_time / len(test_data_loader)
    print("Average PSNR = {}, SSIM = {}, Processing time = {}".format(ave_psnr, ave_ssim, ave_time))


def test_real(args):
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

    print("====> Loading model")
    net_g_path = "checkpoint/netG"
    net_r_path = "checkpoint/netR"

    netG = model_G.to(device)
    netR = model_R.to(device)

    if not find_latest_model(net_g_path) or not find_latest_model(net_r_path):
        print(" [!] Load failed...")
        raise Exception('No model to load for testing!')
    else:
        print(" [*] Load SUCCESS")
        model_path_G = find_latest_model(net_g_path)
        checkpointG = torch.load(model_path_G)
        netG.load_state_dict(checkpointG['model_state_dict'])
        netG.eval()

        model_path_R = find_latest_model(net_r_path)
        checkpointR = torch.load(model_path_R)
        netR.load_state_dict(checkpointR['model_state_dict'])
        netR.eval()

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
            roi_B = netR(real_B)
            pred_S = netG(real_B, roi_B)
            img_S = pred_S.detach().squeeze(0).cpu()
            save_img(img_S, '{}/real_'.format(args.test_dir) + img_name[0])

    total_time = time.time() - start_time
    ave_time = total_time / len(test_data_loader)
    print("Processing time = {}".format(ave_time))
