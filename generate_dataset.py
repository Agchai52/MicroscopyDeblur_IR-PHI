import numpy as np
import cv2
import os
import argparse
from skimage.exposure import rescale_intensity
from scipy.stats import multivariate_normal
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.signal import find_peaks, find_peaks, peak_widths


parser = argparse.ArgumentParser('Geneate dataset BlurMicroscopy')
parser.add_argument('--output_fold', dest='output_fold', help='output directory', type=str, default='./dataset/')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=3000)
parser.add_argument('--image_size', dest='image_size', help='image_size = H = W', type=int, default=256)
parser.add_argument('--std_r', dest='std_r', help='std of additive Gaussion white noise', type=float, default=5.)
parser.add_argument('--phase', dest='phase', help='test or train', type=str, default='train')
parser.add_argument('--is_label', dest='is_label', help='True or False', type=bool, default=True)
args = parser.parse_args()


def Gaussian_2D(m=0, sigma=1.):
    """
    :param M: sample num
    :param m: mean
    :param sigma: std
    :return: Gaussian distribution
    """
    mean = np.zeros(2) + m
    cov = np.eye(2) * sigma ** 2
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    return Gaussian


def generate_bean(bean_size, is_plot=False):
    """
    :param bean_size: bean_size
    :param sigma: std
    :param M: sample num
    :param is_plot: bool plot images
    :return: image of a bean
    """
    intensity = np.random.uniform(low=0.4, high=1.0)
    sigma = bean_size / 2.355 / 2
    Gaussian = Gaussian_2D(m=0, sigma=sigma)
    M = int(sigma * 6)
    if M % 2 == 1:
        M += 1
    X, Y = np.meshgrid(np.linspace(-M//2, M//2, M), np.linspace(-M//2, M//2, M))
    d = np.dstack([X, Y])
    Z = np.zeros((M, M))
    for i in range(len(d)):
        for j in range(len(d[0])):
            x, y = d[i][j]
            if x ** 2 + y ** 2 <= (M//2) ** 2 * 2:
                Z[i][j] = Gaussian.pdf((x, y))

    Z = Z.reshape(M, M)
    max_Z = np.max(Z)
    img_Z = np.uint8(np.asarray(Z)/max_Z*255)

    bean = img_Z * intensity

    if is_plot:
        # cv2.imwrite("bean_size10.png", bean)
        # cv2.imwrite("bean_size50.png", img_Z)

        plt.figure(0)
        plt.imshow(img_Z, cmap='gray', vmin=0, vmax=255)
        plt.figure(1)
        plt.imshow(bean, cmap='gray', vmin=0, vmax=255)

        plt.show()
    return bean


# def generate_elipse(loc, size):
#     """
#     :param loc: (x, y)
#     :return: image of an elipse
#     """
#     x_loc, y_loc = loc
#     width = size * np.random.uniform(low=0.3, high=1.5)
#     height = size * np.random.uniform(low=0.5, high=1.5)
#     angle = np.random.uniform(low=0, high=360)
#     intensity = np.random.uniform(low=0.5, high=1.0)
#
#     elip = patches.Ellipse((x_loc, y_log), width, height, angle=angle) * intensity
#     return elip


def plot_a_bean(background, bean_loc, bean_size, image_size=256):
    bean_loc_x, bean_loc_y = bean_loc
    bean = generate_bean(bean_size=bean_size)
    bean_wid = bean.shape[0]
    if 0 <= bean_loc_y - bean_wid // 2:
        left = bean_loc_y - bean_wid // 2
    else:
        left = 0
        bean = bean[:, (bean_wid // 2 - bean_loc_y):]

    if bean_loc_y + bean_wid // 2 < image_size:
        right = bean_loc_y + bean_wid // 2
    else:
        right = image_size
        bean = bean[:, 0:(image_size + bean_wid // 2 - bean_loc_y)]

    if 0 <= bean_loc_x - bean_wid // 2:
        up = bean_loc_x - bean_wid // 2
    else:
        up = 0
        bean = bean[(bean_wid // 2 - bean_loc_x):, :]

    if bean_loc_x + bean_wid // 2 < image_size:
        down = bean_loc_x + bean_wid // 2
    else:
        down = image_size
        bean = bean[:(image_size + bean_wid // 2 - bean_loc_x), :]

    background[up:down, left:right] += bean

    return background


def generate_sharp_img(image_size=256, bean_size=10, bean_min=3, bean_max=10):
    """
    Generate a sharp image with beans
    :param image_size: image_H = image_W = image_size
    :param bean_size: diameter
    :param bean_min: min num of beans
    :param bean_max: man num of beans
    :param is_plot: bool plot images
    :return: a sharp image with beans, number of beans
    """

    bean_size0 = bean_size
    bean_num = np.random.randint(low=bean_min, high=bean_max)
    # IR-PHI: low=bean_min=3, high=bean_max=10; Fluoresce: low=3, high=8
    bean_id = 0
    background = np.zeros((image_size, image_size))
    for i in range(bean_num):
        # Sample loc for the first bean
        if np.random.random() < 0.8:
            bean_size = np.int(np.ceil(bean_size0 * np.random.uniform(low=0.2, high=0.5)))  # IR-PHI: low=0.5, high=2.5; Fluoresce1: low=0.3, high=1; Fluoresce2: low=0.3, high=1
        else:
            bean_size = np.int(np.ceil(bean_size0 * np.random.uniform(low=0.5, high=0.7)))  # IR-PHI: low=2.5, high=8; ; Fluoresce2: low=1, high=2; Fluoresce2: low=0.3, high=1

        bean_loc = list(np.random.randint(low=0, high=image_size - bean_size // 2, size=(2, )))
        background = plot_a_bean(background, bean_loc, bean_size, image_size)
        bean_id += 1

        # Sample loc for the second bean
        if np.random.random() < 0.5 and 3 <= bean_size < 30:
            new_loc = [0, 0]
            dist1 = list(np.random.randint(low=bean_size // 3, high=bean_size + 1, size=(2, )))
            new_loc[0] = bean_loc[0] + dist1[0] if np.random.random() < 0.5 else bean_loc[0] - dist1[0]
            new_loc[1] = bean_loc[1] + dist1[1] if np.random.random() < 0.5 else bean_loc[1] - dist1[1]

            if (bean_size // 2) < new_loc[0] < (image_size - bean_size // 2) \
                    and (bean_size // 2) < new_loc[1] < (image_size - bean_size // 2):
                background = plot_a_bean(background, new_loc, bean_size, image_size)
                bean_id += 1

                # Sample loc for the third bean
                if np.random.random() < 0.5:
                    new_loc = [0, 0]
                    dist1 = list(np.random.randint(low=bean_size, high=bean_size * 1.5, size=(2,)))
                    new_loc[0] = bean_loc[0] + dist1[0] if np.random.random() < 0.5 else bean_loc[0] - dist1[0]
                    new_loc[1] = bean_loc[1] + dist1[1] if np.random.random() < 0.5 else bean_loc[1] - dist1[1]

                    if (bean_size // 2) < new_loc[0] < (image_size - bean_size // 2) \
                            and (bean_size // 2) < new_loc[1] < (image_size - bean_size // 2):
                        background = plot_a_bean(background, new_loc, bean_size, image_size)
                        bean_id += 1

    return background, bean_id


def convolve(image, kernel):
    """
    2D convolution
    :param image:
    :param kernel:
    :return: convolved image
    """
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
        # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output*255).astype("uint8")
    return output


def kernel_fit(loc):
    """
    Estimated psf of laser
    :param loc: (x, y)
    :return: z
    """
    x, y = loc
    scale = 25
    sigma = 3.6433  # IR-PHI: 160.5586; Fluoresce0: 2.2282; Fluoresce1: 3.6433
    a = 1.8155  # IR-PHI: 65.51; Fluoresce0: 1174.6063; Fluoresce1: 1.8155
    x, y = scale * x, scale * y
    z = np.sqrt(np.log(2)/np.pi) * a / sigma * np.exp(-np.log(2) * (x * x + y * y) / (sigma * sigma))
    return z


def kernel_fit_fluor(loc):
    """
    Estimated psf of laser
    :param loc: (x, y)
    :return: z
    """
    x, y = loc
    scale = 25
    sigma_x = 181.63153641101883  # Fluoresce2: 181.63153641101883 (vertical)
    sigma_y = 221.2152478747844  # Fluoresce2: 221.2152478747844 (horizontal)
    a = 1.0345  # Fluoresce2: 1.0345
    x, y = scale * x, scale * y
    # z = np.sqrt(np.log(2)/np.pi) * a / sigma * np.exp(-np.log(2) * (x * x / sigma_x ** 2 + y * y / sigma_y ** 2))
    z = a * np.exp(-np.log(2) * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2))
    return z


def get_kernel(is_plot=False):
    """
    Compute cropped blur kernel
    :param is_plot: bool
    :return: blur kernel
    """
    M = 61
    X, Y = np.meshgrid(np.linspace(-30, 31, M), np.linspace(-30, 31, M))
    d = np.dstack([X, Y])
    Z = np.zeros((M, M))
    for i in range(len(d)):
        for j in range(len(d[0])):
            x, y = d[i][j]
            Z[i][j] = kernel_fit_fluor((x, y))  # IR-PHI: kernel_fit((x, y))

    Z = Z.reshape(M, M)
    img_Z = np.asarray(Z)
    crop_size = 15
    crop_Z = img_Z[crop_size:M-crop_size, crop_size:M-crop_size]
    kernel = crop_Z / np.float(np.sum(crop_Z))
    if is_plot:
        print(crop_Z.shape)
        print(crop_Z)
        # psf = cv2.imread("psf.png", 0)
        # plt.figure()
        # plt.imshow(psf, cmap='gray', vmin=0, vmax=255)
        plt.figure()
        plt.imshow(img_Z, cmap='gray', vmin=0, vmax=255)
        plt.figure()
        plt.imshow(crop_Z, cmap='gray', vmin=0, vmax=255)
        plt.show()
        exit()
    return kernel


def generate_dataset(name_folder, num_imgs, image_size=256, std_r=5, bean_size=10, is_label=False, is_plot=False):
    name_path_file = name_folder + "_instance_names.txt"
    f_original = open(name_path_file, "w+")

    kernel = get_kernel()

    for i in range(num_imgs):
        name_prefix = '%04d' % (i+1)
        name_blur = name_folder + '/' + name_prefix + "_blur.png"
        name_sharp = name_folder + '/' + name_prefix + "_sharp.png"

        sharp, label = generate_sharp_img(image_size=image_size, bean_size=bean_size)
        blurry = cv2.filter2D(sharp, -1, kernel)  # convolve(sharp, kernel)
        noise_img = np.random.normal(loc=0, scale=std_r, size=blurry.shape)
        blurry_noisy = blurry + noise_img

        cv2.imwrite(name_sharp, sharp)
        cv2.imwrite(name_blur, blurry_noisy)
        print(i+1)

        if is_label:
            f_original.write(name_folder + '/' + name_prefix + "\t" + str(label) + "\r\n")

        if is_plot:
            # cv2.imwrite("sharp_sample.png", sharp)
            # cv2.imwrite("blurry_sample.png", blurry)
            # cv2.imwrite("blurry_noisy_sample.png", blurry_noisy)

            # plt.figure()
            # plt.imshow(sharp, cmap='gray', vmin=0, vmax=255)
            # plt.figure()
            # plt.imshow(blurry, cmap='gray', vmin=0, vmax=255)
            # plt.figure()
            # plt.imshow(blurry_noisy, cmap='gray', vmin=0, vmax=255)
            plt.figure()
            roi_x = np.max(sharp, axis=0, keepdims=False)
            peaks, _ = find_peaks(roi_x)
            results_half = peak_widths(roi_x, peaks, rel_height=0.5)

            plt.plot(roi_x)
            if len(results_half[0]) > 0:
                fwhm1 = results_half[0][0]
                fwhm1 = float("{:.2f}".format(fwhm1))
                print("FWHM = ", fwhm1)
                plt.hlines(*results_half[1:], color="C2")

            fig = plt.figure()
            plt.imshow(sharp, cmap=plt.get_cmap("jet"))
            # ax = fig.gca()
            # fwhm = 50
            # c1 = plt.Circle((128, 128), fwhm / 2, color='red', linewidth=1, fill=False)
            # ax.add_patch(c1)
            # c2 = plt.Circle((128+50, 128), fwhm / 2, color='green', linewidth=1, fill=False)
            # ax.add_patch(c2)
            plt.colorbar()
            plt.figure()
            plt.imshow(blurry, cmap=plt.get_cmap("jet"))
            plt.colorbar()
            plt.show()
            exit()
    f_original.close()
    return


if __name__ == "__main__":
    """
    Example:
        python generate_dataset.py --phase train --num_imgs 2000
        python generate_dataset.py --phase test --num_imgs 500
    """
    args = parser.parse_args()

    img_output_fold = args.output_fold + args.phase  # Output Folder

    if not os.path.isdir(img_output_fold):
        os.makedirs(img_output_fold)

    generate_dataset(img_output_fold, args.num_imgs, image_size=args.image_size, is_label=args.is_label)

    if args.is_label:
        f_train = open(img_output_fold + "_instance_names.txt", "r")
        train_data_name = f_train.readlines()
        f_train.close()
        print("Number of instances = {}".format(len(train_data_name)))

    # bean = generate_bean(bean_size=3, sigma=0.2, M=50)
    # cv2.imwrite("bean_size3.png", bean)

