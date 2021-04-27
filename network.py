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
            x, y = x * 50, y * 50
            z = np.exp(-np.log(2) * (x * x + y * y) / (160.5586 * 160.5586)) * 255
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


class Generator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Generator, self).__init__()

        def down(c_in, c_out, k=3, s=2, p=0, d=1):
            return nn.Sequential(
                nn.ReflectionPad2d([0, 1, 0, 1]),
                nn.Conv2d(c_in, c_out, k, s, p, d), nn.SELU(inplace=True)
            )

        def up(c_in, c_out, k=2, s=2):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s),
                nn.SELU(inplace=True)
            )

        self.input_nc = args.input_nc
        self.ngf = args.ngf
        self.device = device
        self.loss = nn.MSELoss()
        self.load_size = args.load_size

        self.in_net1 = nn.Sequential(
            nn.Conv2d(self.input_nc, self.ngf, 3, 1, 1),
            nn.SELU(inplace=True),
        )

        self.in_net2 = down(self.ngf * 1, self.ngf * 2)
        self.in_net3 = down(self.ngf * 2, self.ngf * 4)

        self.att_net = Attention(self.ngf * 4)  # (B, 16, 64, 64)

        self.res_net1 = ResBlock(self.ngf * 4, self.ngf * 4)
        self.res_net2 = ResBlock(self.ngf * 2, self.ngf * 2)

        self.up_net1 = up(self.ngf * 4, self.ngf * 2)
        self.up_net2 = up(self.ngf * 2, self.ngf * 2)
        self.up_net3 = up(self.ngf * 6, self.ngf * 2)
        self.up_net4 = up(self.ngf * 4, self.ngf * 1)

        self.max_pool = nn.MaxPool2d(4)

        self.end_net = nn.Sequential(nn.Conv2d(self.ngf * 2, self.input_nc, 1, 1, 0), nn.Tanh())

    def forward(self, x):
        # Encode
        e1 = self.in_net1(x)   # (B, 64, 64, 64)
        e2 = self.in_net2(e1)  # (B, 64*2, 32, 32)
        e3 = self.in_net3(e2)  # (B, 64*4, 16, 16)
        # Attention
        y = self.att_net(e3)   # (B, 64*4, 16, 16)

        # Decode
        d1 = self.up_net1(y)   # (B, 64*2, 32, 32)
        # d1 = self.res_net1(d1)

        d2 = self.up_net2(d1)  # (B, 64*2, 64, 64)
        d2 = torch.cat([self.max_pool(e3), d2])  # (B, 64*6, 64, 64)

        d3 = self.up_net3(d2)  # (B, 64*2, 64, 64)
        d3 = torch.cat([self.max_pool(e2), d3])  # (B, 64*4, 128, 128)

        d4 = self.up_net4(d3)  # (B, 64*1, 64, 64)
        d4 = torch.cat([self.max_pool(e1), d4])  # (B, 64*2, 256, 256)

        y4 = self.end_net(d4)
        return y4


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


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.BCELoss()

    def get_target_tensor(self, image, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(image)

    def __call__(self, img, target_is_real):
        target_tensor = self.get_target_tensor(img, target_is_real)
        return self.loss(img, target_tensor)


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
