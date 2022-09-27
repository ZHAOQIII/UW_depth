import torch
import torch.nn as nn
import functools
import numpy as np

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        layers0 = []
        layers0.append(nn.utils.spectral_norm(nn.Conv2d(6+ c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)))
        layers0.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers0.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*layers0)
        # Down-sampling layers.
        curr_dim = conv_dim
        layers1=[]
        layers1.append(nn.utils.spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)))
        layers1.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        layers1.append(nn.LeakyReLU(0.2))
        curr_dim = curr_dim * 2
        self.down1 = nn.Sequential(*layers1)

        layers2=[]
        layers2.append(nn.utils.spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)))
        layers2.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        layers2.append(nn.LeakyReLU(0.2))
        curr_dim = curr_dim * 2
        self.down2 = nn.Sequential(*layers2)

        # Bottleneck layers.
        layers3=[]
        for i in range(repeat_num):
            layers3.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))#ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.res = nn.Sequential(*layers3)
        # Up-sampling layers.
        layers4=[]
        layers4.append(nn.utils.spectral_norm(nn.ConvTranspose2d(curr_dim*2, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)))
        layers4.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
        layers4.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2
        self.up1 = nn.Sequential(*layers4)

        layers5 = []
        layers5.append(nn.utils.spectral_norm(
            nn.ConvTranspose2d(curr_dim*2, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)))
        layers5.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
        layers5.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2
        self.up2 = nn.Sequential(*layers5)

        layers6=[]
        layers6.append(nn.utils.spectral_norm(nn.Conv2d(curr_dim*2, 3, kernel_size=7, stride=1, padding=3, bias=False)))
        layers6.append(nn.Tanh())
        self.main = nn.Sequential(*layers6)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x=torch.cat((x, c), dim=1)
        x_0=self.conv(x)
        x_down1=self.down1(x_0)
        x_down2=self.down2(x_down1)
        x_res=self.res(x_down2)
        x_up1=self.up1(torch.cat((x_res,x_down2),1))
        x_up2=self.up2(torch.cat((x_up1,x_down1),1))
        return self.main(torch.cat((x_up2,x_0),1))



class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=3, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.utils.spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):

        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class Discriminator1(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Discriminator1, self).__init__()
        layers = []
        layers.append(nn.utils.spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        h = self.main(x)
        out_src = self.conv1(h)
        return out_src

class NLayerDiscriminator1(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator1, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                                 kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                             kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        # if use_sigmoid:
        # sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    def __init__(self,c_dim=3,input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                              bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult * 2, kernel_size=4, stride=2, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult * 2, ndf * nf_mult * 4, kernel_size=4, stride=2, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult * 4, c_dim, kernel_size=4, stride=2, padding=padw, bias=use_bias)),
        )

    def forward(self, input):
        """Standard forward."""
        a = self.model(input)
        b = self.conv1(a)
        c = self.conv2(a)
        return b, c.view(c.size(0), c.size(1))




