from __future__ import division
import os
import numpy as np
import math
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def save_img(image_tensor, filename):
    image_numpy = image_tensor.squeeze(0).float().numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename+'.png')


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_metrics(real, pred):
    real = real.squeeze(0).squeeze(0).cpu()
    pred = pred.detach().squeeze(0).squeeze(0).cpu()
    real = real.float().numpy()
    pred = pred.float().numpy()
    real = (real + 1.0) / 2.0
    pred = (pred + 1.0) / 2.0
    cur_psnr = psnr(real, pred)
    cur_ssim = ssim(real, pred, gaussian_weights=True, multichannel=False, use_sample_covariance=False)
    return cur_psnr, cur_ssim


def find_latest_model(net_path):
    file_list = os.listdir(net_path)
    model_names = [int(f[14:-4]) for f in file_list if ".pth" in f]
    if len(model_names) == 0:
        return False
    else:
        iter_num = max(model_names)
        if net_path[-1] == 'G':
            return os.path.join(net_path, "G_model_epoch_{}.pth".format(iter_num))
        elif net_path[-1] == 'D':
            return os.path.join(net_path, "D_model_epoch_{}.pth".format(iter_num))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def plot_losses():
    loss_record = "loss_record.txt"
    psnr_record = "psnr_record.txt"
    ssim_record = "ssim_record.txt"

    losses_dg = np.loadtxt(loss_record)
    psnr_ave = np.loadtxt(psnr_record)
    ssim_ave = np.loadtxt(ssim_record)

    plt.figure()
    plt.plot(losses_dg[0:-1:100, 0], 'b-', label='grad_loss')
    plt.plot(losses_dg[0:-1:100, 1], 'r--', label='l2_loss')
    plt.plot(losses_dg[0:-1:100, 2], 'g-', label='recover_loss')
    plt.plot(losses_dg[0:-1:100, 3], 'y-', label='d_real_loss')
    plt.plot(losses_dg[0:-1:100, 4], 'k-', label='d_fake_loss')
    plt.xlabel("iteration*100")
    plt.ylabel("Error")
    plt.legend()
    # plt.xlim(xmin=-5, xmax=480)
    plt.ylim(ymin=0, ymax=2)
    plt.title("L2_g_Recover Loss")
    plt.savefig("plot_3_losses.jpg")
    # plt.show()

    plt.figure()
    plt.plot(psnr_ave, 'r-')
    plt.xlabel("epochs")
    plt.ylabel("Average PSNR")
    # plt.xlim(xmin=-5, xmax=300)  # xmax=300
    # plt.ylim(ymin=0, ymax=30.)  # ymax=60
    plt.title("Validation PSNR")
    plt.savefig("plot_psnr_loss.jpg")

    plt.figure()
    plt.plot(ssim_ave, 'r-')
    plt.xlabel("epochs")
    plt.ylabel("Average SSIM")
    # plt.xlim(xmin=-5, xmax=300)  # xmax=300
    # plt.ylim(ymin=0, ymax=30.)  # ymax=60
    plt.title("Validation SSIM")
    plt.savefig("plot_ssim_loss.jpg")

    # plt.figure()
    # plt.plot(ddg_ave[:, 0], 'b-', label='d_fake')
    # plt.plot(ddg_ave[:, 1], 'r-', label='d_real')
    # plt.plot(ddg_ave[:, 2], 'g-', label='gan')
    # plt.xlabel("epochs")
    # plt.ylabel("Average loss")
    # plt.legend()
    # # plt.xlim(xmin=-5, xmax=300)  # xmax=300
    # plt.ylim(ymin=0, ymax=2.)  # ymax=60
    # plt.title("D1_D2_G PSNR")
    # plt.savefig("plot_ddg_loss.jpg")
# plot_losses()



