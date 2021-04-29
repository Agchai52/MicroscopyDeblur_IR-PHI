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

        def kernel_fit(loc):
            """
            Estimated psf of laser
            :param loc: (x, y)
            :return: z
            """
            x, y = loc
            scale = 50  # 50
            sigma = 160.5586
            x, y = scale * x, scale * y
            z = np.exp(-np.log(2) * (x * x + y * y) / (sigma * sigma)) * 255
            return z

        def get_kernel():
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
                    Z[i][j] = kernel_fit((x, y))

            Z = Z.reshape(M, M)
            img_Z = np.asarray(Z)
            crop_size = 15
            crop_Z = img_Z[crop_size:M - crop_size, crop_size:M - crop_size]
            kernel = crop_Z / np.float(np.sum(crop_Z))
            return kernel

        self.batch_size = args.batch_size
        self.device = device
        self.kernel = torch.FloatTensor(get_kernel())  # (31, 31)
        self.loss = nn.MSELoss()
        self.kernel_size = self.kernel.shape[0]
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)

    def forward(self, x):
        # x : (B, 1, H, W)
        # Padding
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)

        # weight :
        kernel = self.kernel.expand(1, 1, self.kernel_size, self.kernel_size)  # (1, 1, 31, 31)
        kernel = kernel.flip(-1).flip(-2).to(self.device)

        # Convolution
        blur_img = F.conv2d(x, kernel)
        return blur_img

    def __call__(self, x):
        b, c, h, w = x.shape
        x1 = x
        x2 = F.interpolate(x, (h // 2, w // 2), mode="bilinear")
        x3 = F.interpolate(x, (h // 4, w // 4), mode="bilinear")

        y1 = self.forward(x1)
        y2 = self.forward(x2)
        y3 = self.forward(x3)
        return list([y3, y2, y1])


class Generator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Generator, self).__init__()

        def down(c_in, c_out, k=3, s=2, p=0, d=1):
            return nn.Sequential(
                nn.ReflectionPad2d([0, 1, 0, 1]),
                nn.Conv2d(c_in, c_out, k, s, p, d), nn.SELU(inplace=True),
                Channel_Att(c_out)
            )

        def up(c_in, c_out, k=2, s=2):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s),
                nn.SELU(inplace=True),
                Channel_Att(c_out)
            )

        self.input_nc = args.input_nc
        self.ngf = args.ngf
        self.device = device
        self.loss = nn.MSELoss()
        self.load_size = args.load_size

        self.e1 = nn.Sequential(ConvBlock(self.input_nc, self.ngf * 1),
                                ConvBlock(self.ngf * 1, self.ngf * 1))  # (B, 64, H, W)
        self.e2 = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                ConvBlock(self.ngf * 1, self.ngf * 2),
                                ConvBlock(self.ngf * 2, self.ngf * 2))  # (B, 128, H/2, W/2)
        self.e3 = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                ConvBlock(self.ngf * 2, self.ngf * 4),
                                ConvBlock(self.ngf * 4, self.ngf * 4),
                                ConvBlock(self.ngf * 4, self.ngf * 4),  # (B, 256, H/4, W/4)
                                nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=3, stride=2, padding=1,
                                                   output_padding=1))  # (B, 128, H/2, W/2)

        # Decoder
        self.d1 = nn.Sequential(ConvBlock(self.ngf * 4, self.ngf * 2),
                                ConvBlock(self.ngf * 2, self.ngf * 2),  # (B, 128, H/2, W/2)
                                nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, kernel_size=3, stride=2, padding=1,
                                                   output_padding=1))  # (B, 64, H, W)
        # self.d2 = nn.Sequential(ConvBlock(self.ngf * 1, self.ngf * 1),
        #                         ConvBlock(self.ngf * 1, self.ngf * 1),  # (B, 128, H/2, W/2)
        #                         nn.ConvTranspose2d(self.ngf * 1, self.ngf * 1, kernel_size=3, stride=2, padding=1,
        #                                            output_padding=1))  # (B, 64, H, W)
        # self.d3 = nn.Sequential(ConvBlock(self.ngf * 1, self.ngf * 1),
        #                         ConvBlock(self.ngf * 1, self.ngf * 1),  # (B, 128, H/2, W/2)
        #                         nn.ConvTranspose2d(self.ngf * 1, self.ngf * 1, kernel_size=3, stride=2, padding=1,
        #                                            output_padding=1))  # (B, 64, H, W)
        self.d4 = nn.Sequential(ConvBlock(self.ngf * 2, self.ngf * 1),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(self.ngf * 1, self.input_nc, kernel_size=3, stride=1, padding=0,
                                          padding_mode='circular'),  # (B, 1, H, W)
                                nn.Tanh())

        # self.in_net1 = nn.Sequential(
        #     nn.Conv2d(self.input_nc, self.ngf, 3, 1, 1),
        #     nn.SELU(inplace=True),
        # )
        #
        # self.in_net2 = down(self.ngf * 1, self.ngf * 2)
        # self.in_net3 = down(self.ngf * 2, self.ngf * 4)
        #
        # self.res_net1 = ResBlock(self.ngf * 1, self.ngf * 1)
        # self.res_net2 = ResBlock(self.ngf * 2, self.ngf * 2)
        # self.res_net3 = ResBlock(self.ngf * 2, self.ngf * 2)
        # self.res_net4 = ResBlock(self.ngf * 1, self.ngf * 1)
        #
        # self.up_net1 = up(self.ngf * 4, self.ngf * 2)
        # self.up_net2 = up(self.ngf * 2, self.ngf * 1)
        #
        # self.end_net = nn.Sequential(nn.Conv2d(self.ngf * 1, self.input_nc, 1, 1, 0), nn.Tanh())

    def forward(self, img):
        # # Encode
        # e1 = self.in_net1(x)   # (B, 64*1, 256, 256)
        # # e1 = self.res_net1(e1)
        # e2 = self.in_net2(e1)  # (B, 64*2, 128, 128)
        # # e2 = self.res_net2(e2)
        # e3 = self.in_net3(e2)  # (B, 64*4, 64, 64)
        # # Attention
        # # y = self.att_net(e3)   # (B, 64*4, 64, 64)
        #
        # # Decode
        # d1 = self.up_net1(e3)   # (B, 64*2, 128, 128)
        # # d1 = torch.cat([e2, d1], dim=1)  # (B, 64*4, 128, 128)
        # # d1 = self.res_net3(d1)
        #
        # d2 = self.up_net2(d1)  # (B, 64*1, 256, 256)
        # # d2 = torch.cat([e1, d2], dim=1)  # (B, 64*2, 256, 256)
        # # d2 = self.res_net4(d2)
        #
        # y = self.end_net(d2)

        # Encoder
        e_layer1 = self.e1(img)
        e_layer2 = self.e2(e_layer1)
        e_layer3 = self.e3(e_layer2)

        # Decoder
        e_layer3 = torch.cat([e_layer2, e_layer3], 1)
        d_layer1 = self.d1(e_layer3)

        d_layer1 = torch.cat([e_layer1, d_layer1], 1)
        # d_layer2 = self.d2(d_layer1)
        #
        # d_layer3 = self.d3(d_layer2)
        d_layer4 = self.d4(d_layer1)
        return d_layer4

    def __call__(self, x):
        b, c, h, w = x.shape
        x1 = self.forward(x)
        x1 = F.interpolate(x1, (h * 2, w * 2), mode="bilinear")
        x2 = self.forward(x1)
        x2 = F.interpolate(x2, (h * 4, w * 4), mode="bilinear")
        x3 = self.forward(x2)
        return list([x1, x2, x3])


class Discriminator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Discriminator, self).__init__()
        self.input_nc = args.input_nc
        self.ndf = args.ndf
        self.device = device
        self.d_1 = nn.Sequential(ConvBlock(self.input_nc, self.ndf * 1, stride=2),  # (B, 64, H/2, W/2)
                                 ConvBlock(self.ndf * 1, self.ndf * 2, stride=2),   # (B, 128, H/4, W/4)
                                 ConvBlock(self.ndf * 2, self.ndf * 4, stride=2),   # (B, 256, H/8, W/8)
                                 ConvBlock(self.ndf * 4, self.ndf * 8, stride=2),   # (B, 512, H/16, W/16)
                                 nn.Conv2d(self.ndf * 8, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
                                 )                                                  # (B, 1, H/16, W/16)

        self.d_2 = nn.Sequential(ConvBlock(self.input_nc, self.ndf * 1, stride=2),  # (B, 64, H/4, W/4)
                                 ConvBlock(self.ndf * 1, self.ndf * 2, stride=2),   # (B, 128, H/8, W/8)
                                 ConvBlock(self.ndf * 2, self.ndf * 4, stride=2),   # (B, 256, H/16, W/16)
                                 ConvBlock(self.ndf * 4, self.ndf * 8, stride=2),   # (B, 512, H/32, W/32)
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.Conv2d(self.ndf * 8, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
                                 )                                                  # (B, 1, H/32, W/32)

        self.d_3 = nn.Sequential(ConvBlock(self.input_nc, self.ndf * 1, stride=2),  # (B, 64, H/8, W/8)
                                 ConvBlock(self.ndf * 1, self.ndf * 2, stride=2),   # (B, 128, H/16, W/16)
                                 ConvBlock(self.ndf * 2, self.ndf * 4, stride=2),   # (B, 256, H/32, W/32)
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 ConvBlock(self.ndf * 4, self.ndf * 8, stride=2),   # (B, 512, H/32, W/32)
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.Conv2d(self.ndf * 8, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
                                 )                                                  # (B, 1, H/32, W/32)

    def forward(self, img):
        def random_crop(img, crop_shape):
            b, _, h, w = img.shape
            h1 = int(np.ceil(np.random.uniform(1e-2, h - crop_shape[0])))
            w1 = int(np.ceil(np.random.uniform(1e-2, w - crop_shape[1])))
            crop = img[:, :, h1:h1+crop_shape[0], w1:w1+crop_shape[1]]
            return crop

        b, _, h, w = img.shape
        out1 = self.d_1(img)
        img = random_crop(img, (h // 2, w // 2))
        out2 = self.d_2(img)
        img = random_crop(img, (h // 4, w // 4))
        out3 = self.d_3(img)
        return out1, out2, out3


class Attention(nn.Module):
    def __init__(self, ch):
        super(Attention, self).__init__()
        self.ch = ch
        self.conv2d_f = nn.Conv2d(self.ch, self.ch // 2, 1, 1)
        self.conv2d_g = nn.Conv2d(self.ch, self.ch // 2, 1, 1)
        self.conv2d_h = nn.Conv2d(self.ch, self.ch, 1, 1)
        self.gamma = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)

    def forward(self, x):
        b, _, height, width = x.shape
        f = self.conv2d_f(x).view(b, self.ch // 2, -1)   # (b, ch/2, h*w)
        g = self.conv2d_g(x).view(b, self.ch // 2, -1)
        g = g.permute(0, 2, 1)                           # (b, h*w, ch/2)
        h = self.conv2d_h(x).view(b, self.ch, -1)
        h = h.permute(0, 2, 1)                           # (b, h*w, ch)

        s = torch.matmul(g, f)                           # (b, h*w, h*w)
        beta = F.softmax(s, dim=-1)                      # (b, h*w, h*w)
        o = torch.matmul(beta, h)                        # (b, h*w, ch)
        o = o.permute(0, 2, 1).view(b, self.ch, height, width)
        o = self.gamma * o + x
        return o


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


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3):
        super(ResBlock, self).__init__()
        self.res_net = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(c_in, c_out, kernel_size=k_size, stride=1),
            nn.SELU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(c_in, c_out, kernel_size=k_size, stride=1),
            nn.SELU(inplace=True),
        )

    def forward(self, x):
        x1 = self.res_net(x)
        x2 = x + x1
        return x2


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, pad=0):
        super(ConvBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if stride == 1:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                          padding_mode='circular'),
                nn.InstanceNorm2d(self.c_out),
                nn.ReLU(inplace=True))
        elif stride == 2:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((0, 1, 0, 1)),
                nn.Conv2d(self.c_in, self.c_out, kernel_size=k_size, stride=stride, padding=pad,
                          padding_mode='circular'),
                nn.InstanceNorm2d(self.c_out),
                nn.ReLU(inplace=True))
        else:
            raise Exception("stride size = 1 or 2")

    def forward(self, maps):
        return self.model(maps)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.MSELoss()

    def get_target_tensor(self, image, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(image)

    def __call__(self, img, target_is_real):
        img1, img2, img3 = img
        target_tensor1 = self.get_target_tensor(img1, target_is_real)
        target_tensor2 = self.get_target_tensor(img2, target_is_real)
        target_tensor3 = self.get_target_tensor(img3, target_is_real)

        loss = (self.loss(img1, target_tensor1) + self.loss(img2, target_tensor2) + self.loss(img3, target_tensor3)) / 3
        return loss


class DarkChannelLoss(nn.Module):
    def __init__(self, kernel_size=15):
        super(DarkChannelLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)

    def forward(self, x):
        # x : (B, 3, H, W), in [-1, 1]
        x = (x + 1.0) / 2.0
        H, W = x.size()[2], x.size()[3]

        # Minimum among three channels
        x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        x = self.unfold(x)  # (B, k*k, H*W)
        x = x.unsqueeze(1)  # (B, 1, k*k, H*W)

        # Minimum in (k, k) patch
        dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
        x = dark_map.view(-1, 1, H, W)

        # Count Zeros
        #y0 = torch.zeros_like(x)
        #y1 = torch.ones_like(x)
        #x = torch.where(x < 0.1, y0, y1)
        #x = torch.sum(x)
        #x = int(H * W - x)
        return x.clamp(min=0.0, max=1.0)

    def __call__(self, real, fake):
        real_map = self.forward(real)
        fake_map = self.forward(fake)
        return self.loss(real_map, fake_map)


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

        return 0.5 * (h_map + v_map)
