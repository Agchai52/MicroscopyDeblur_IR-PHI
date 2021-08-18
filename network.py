import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)


class BlurModel(nn.Module):
    def __init__(self, args, device='cpu'):
        super(BlurModel, self).__init__()

        self.batch_size = args.batch_size
        self.device = device
        self.args = args  # (31, 31)

    def kernel_fit(self, loc, level=1.0):
        """
        Estimated psf of laser
        :param loc: (x, y)
        :param level: level of image
        :return: z
        """
        x, y = loc
        scale = 25 * level
        sigma = 3.6433  # IR-PHI: 160.5586; Fluoresce0: 2.2282; Fluoresce1: 3.6433
        a = 1.8155  # IR-PHI: 65.51; Fluoresce0: 1174.6063; Fluoresce1: 1.8155
        x, y = scale * x, scale * y
        z = np.sqrt(np.log(2) / np.pi) * a / sigma * np.exp(-np.log(2) * (x * x + y * y) / (sigma * sigma))
        return z

    def kernel_fit_fluor(self, loc, level=1.0):
        """
        Estimated psf of laser
        :param loc: (x, y)
        :return: z
        """
        x, y = loc
        scale = 25 * level
        sigma_x = 181.63153641101883  # Fluoresce2: 181.63153641101883 (vertical)
        sigma_y = 221.2152478747844  # Fluoresce2: 221.2152478747844 (horizontal)
        a = 1.0345  # Fluoresce2: 1.0345
        x, y = scale * x, scale * y
        # z = np.sqrt(np.log(2)/np.pi) * a / sigma * np.exp(-np.log(2) * (x * x / sigma_x ** 2 + y * y / sigma_y ** 2))
        z = a * np.exp(-np.log(2) * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2))
        return z

    def get_kernel(self, level=1.0):
        """
        Compute cropped blur kernel
        :param level: level of image
        :return: blur kernel
        """
        M = 61
        X, Y = np.meshgrid(np.linspace(-30, 31, M), np.linspace(-30, 31, M))
        d = np.dstack([X, Y])
        Z = np.zeros((M, M))
        for i in range(len(d)):
            for j in range(len(d[0])):
                x, y = d[i][j]
                Z[i][j] = self.kernel_fit_fluor((x, y), level)  # IR-PHI: self.kernel_fit((x, y), level)

        Z = Z.reshape(M, M)
        img_Z = np.asarray(Z)
        crop_size = 15
        crop_Z = img_Z[crop_size:M - crop_size, crop_size:M - crop_size]
        kernel = crop_Z / np.float(np.sum(crop_Z))
        return kernel

    def forward(self, x):
        # x : (B, 1, H, W)
        _, _, h, w = x.shape
        # weight :
        level = self.args.load_size / h
        kernel = torch.FloatTensor(self.get_kernel(level))
        kernel_size = kernel.shape[0]
        pad_size = (kernel_size - 1) // 2
        kernel = kernel.expand(1, 1, kernel_size, kernel_size)  # (1, 1, 31, 31)
        kernel = kernel.flip(-1).flip(-2).to(self.device)

        # Padding
        x = nn.ZeroPad2d(pad_size)(x)  # (B, 1, H+2p, W+2p)

        # Convolution
        blur_img = F.conv2d(x, kernel)
        return blur_img

    def __call__(self, x):
        b, c, h, w = x.shape
        # x1 = x
        # x2 = F.interpolate(x, (h // 2, w // 2), mode="bilinear")
        # x3 = F.interpolate(x, (h // 4, w // 4), mode="bilinear")
        # y1 = self.forward(x1)
        # y2 = self.forward(x2)
        # y3 = self.forward(x3)
        y1 = self.forward(x)
        y2 = F.interpolate(y1, (h // 2, w // 2), mode="bilinear")
        y3 = F.interpolate(y1, (h // 4, w // 4), mode="bilinear")
        return list([y3, y2, y1])


class Generator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Generator, self).__init__()

        self.input_nc = args.input_nc
        self.ngf = args.ngf
        self.device = device
        self.loss = nn.MSELoss()
        self.load_size = args.load_size

        self.e1 = nn.Sequential(ConvBlock(self.input_nc, self.ngf * 1),
                                ConvBlock(self.ngf * 1, self.ngf * 1)
                                )  # (B, 64, H, W)
        self.e2 = nn.Sequential(
                               ConvBlock(self.ngf * 1, self.ngf * 2, stride=2),
                               ConvBlock(self.ngf * 2, self.ngf * 2)
                                )  # (B, 128, H/2, W/2)
        self.e3 = nn.Sequential(
                                ConvBlock(self.ngf * 2, self.ngf * 4, stride=2),
                                ConvBlock(self.ngf * 4, self.ngf * 4),
                                ConvBlock(self.ngf * 4, self.ngf * 4),  # (B, 256, H/4, W/4)
                                nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=3, stride=2, padding=1,
                                                   output_padding=1))  # (B, 128, H/2, W/2)

        # Decoder
        self.d1 = nn.Sequential(ConvBlock(self.ngf * 4, self.ngf * 2),
                                ConvBlock(self.ngf * 2, self.ngf * 2),  # (B, 128, H/2, W/2)
                                nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, kernel_size=3, stride=2, padding=1,
                                                   output_padding=1))  # (B, 64, H, W)
        self.d2 = nn.Sequential(ConvBlock(self.ngf * 2, self.ngf * 1),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(self.ngf * 1, self.input_nc, kernel_size=3, stride=1, padding=0,
                                          padding_mode='circular'),  # (B, 1, H, W)
                                nn.Tanh())

    def forward(self, img):
        # Encoder
        e_layer1 = self.e1(img)
        e_layer2 = self.e2(e_layer1)
        e_layer3 = self.e3(e_layer2)

        # Decoder
        e_layer3 = torch.cat([e_layer2, e_layer3], 1)
        d_layer1 = self.d1(e_layer3)

        d_layer1 = torch.cat([e_layer1, d_layer1], 1)
        d_layer2 = self.d2(d_layer1)
        return d_layer2

    def __call__(self, x):
        b, c, h, w = x.shape
        x1 = self.forward(x)
        y1 = F.interpolate(x1, (h * 2, w * 2), mode="bilinear")
        x2 = self.forward(y1)
        y2 = F.interpolate(x2, (h * 4, w * 4), mode="bilinear")
        x3 = self.forward(y2)
        return list([x1, x2, x3])


class Classifier(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Classifier, self).__init__()

        def down(c_in, c_out, k=3, s=2, p=0, d=1):
            return nn.Sequential(
                nn.ReflectionPad2d([0, 1, 0, 1]),
                nn.Conv2d(c_in, c_out, k, s, p, d),
                nn.InstanceNorm2d(c_out),
                nn.ReLU(inplace=True),
                Channel_Att(c_out),
            )

        def up(c_in, c_out, k=3, s=2):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=1, output_padding=1),
                nn.InstanceNorm2d(c_out),
                nn.ReLU(inplace=True),
                Channel_Att(c_out)
            )

        self.input_nc = args.input_nc
        self.ndf = args.ndf
        self.load_size = args.load_size
        self.device = device
        self.classes = args.classes

        self.e_1 = nn.Sequential(
            down(self.input_nc, self.ndf * 1),  # (B, 32 * 1, H/2, W/2)
            down(self.ndf * 1, self.ndf * 2),   # (B, 32 * 2, H/4, W/4)
            down(self.ndf * 2, self.ndf * 4),  # (B, 32 * 4, H/8, W/8)
            down(self.ndf * 4, self.ndf * 8),  # (B, 32 * 8, H/16, W/16)
            up(self.ndf * 8, self.ndf * 4),    # (B, 32 * 4, H/8, W/18)
            up(self.ndf * 4, self.ndf * 2),    # (B, 32 * 2, H/4, W/4)
            up(self.ndf * 2, self.ndf * 1),    # (B, 32 * 1, H/2, W/2)
            up(self.ndf * 1, self.ndf * 1),    # (B, 32 * 1, H, W)
            nn.Conv2d(self.ndf * 1, self.input_nc, 1, 1),  # (B, 2, H/32, W/32)
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(nn.Linear(self.ndf * 4, self.ndf * 2),
                                nn.ReLU(),
                                nn.Linear(self.ndf * 2, self.classes),
                                nn.Softmax()
                                )

    def forward(self, img):
        mask_map = self.e_1(img)
        return mask_map


class Channel_Att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Att, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, cha_att=True, k_size=3, stride=1, pad=0):
        super(ConvBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if stride == 1:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                          padding_mode='circular'),
                nn.InstanceNorm2d(self.c_out),
                nn.ReLU(inplace=True),
                Channel_Att(self.c_out),
            )
        elif stride == 2:
            if cha_att:
                self.model = nn.Sequential(
                    nn.ReflectionPad2d((0, 1, 0, 1)),
                    nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                              padding_mode='circular'),
                    nn.InstanceNorm2d(self.c_out),
                    nn.ReLU(inplace=True),
                    Channel_Att(self.c_out),
                )
            else:
                self.model = nn.Sequential(
                    nn.ReflectionPad2d((0, 1, 0, 1)),
                    nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                              padding_mode='circular'),
                    nn.InstanceNorm2d(self.c_out),
                    nn.ReLU(inplace=True),
                )
        else:
            raise Exception("stride size = 1 or 2")

    def forward(self, maps):
        return self.model(maps)


class GradientLoss(nn.Module):
    def __init__(self, kernel_size=3, device="cpu", is_regular=False):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)
        self.device = device
        self.is_regular = is_regular

    def forward(self, x):
        """
        Sobel Filter
        :param x:
        :return: dh, dv
        """
        # x : (B, 3, H, W)
        x = (x + 1.0) / 2.0

        # Compute a gray-scale image by averaging
        x = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        # weight :
        filter_h = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).expand(1, 1, 3, 3)
        filter_v = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).expand(1, 1, 3, 3)

        filter_h = filter_h.flip(-1).flip(-2)
        filter_v = filter_v.flip(-1).flip(-2)

        filter_h = filter_h.to(self.device)
        filter_v = filter_v.to(self.device)
        # Convolution
        gradient_h = F.conv2d(x, filter_h)
        gradient_v = F.conv2d(x, filter_v)

        return gradient_h, gradient_v

    def __call__(self, fake, real):
        fake_grad_h, fake_grad_v = self.forward(fake)
        if self.is_regular:
            real_grad_h, real_grad_v = torch.zeros_like(fake_grad_h), torch.zeros_like(fake_grad_v)
        else:
            real_grad_h, real_grad_v = self.forward(real)
        h_map = self.loss(real_grad_h, fake_grad_h)
        v_map = self.loss(real_grad_v, fake_grad_v)

        return (h_map + v_map)
