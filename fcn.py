import torch
from torch import nn


__all__ = ['FCDense', 'fcdense_tiny', 'fcdense56_nodrop',
           'fcdense56', 'fcdense67', 'fcdense103',
           'fcdense103_nodrop']


class DenseBlock(nn.Module):

    def __init__(self, nIn, growth_rate, depth, drop_rate=0, only_new=False,
                 bottle_neck=False):
        super(DenseBlock, self).__init__()
        self.only_new = only_new
        self.depth = depth
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList([self.get_transform(
            nIn + i * growth_rate, growth_rate, bottle_neck,
            drop_rate) for i in range(depth)])

    def forward(self, x):
        if self.only_new:
            outputs = []
            for i in range(self.depth):
                tx = self.layers[i](x)
                x = torch.cat((x, tx), 1)
                outputs.append(tx)
            return torch.cat(outputs, 1)
        else:
            for i in range(self.depth):
                x = torch.cat((x, self.layers[i](x)), 1)
            return x

    def get_transform(self, nIn, nOut, bottle_neck=None, drop_rate=0):
        if not bottle_neck or nIn <= nOut * bottle_neck:
            return nn.Sequential(
                nn.BatchNorm2d(nIn),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.Conv2d(nIn, nOut, 3, stride=1, padding=1, bias=True)),
                nn.Dropout(drop_rate),
            )
        else:
            nBottle = nOut * bottle_neck
            return nn.Sequential(
                nn.BatchNorm2d(nIn),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.Conv2d(nIn, nBottle, 1, stride=1, padding=0, bias=True)),
                nn.BatchNorm2d(nBottle),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.Conv2d(nBottle, nOut, 3, stride=1, padding=1, bias=True)),
                nn.Dropout(drop_rate),
            )


class FCDense(nn.Module):

    def __init__(self, depths, growth_rates, n_scales=5, n_channel_start=32,
                 n_classes=3, drop_rate=0, bottle_neck=False):
        super(FCDense, self).__init__()
        self.n_scales = n_scales
        self.n_classes = n_classes
        self.n_channel_start = n_channel_start
        self.depths = [depths] * \
            (2 * n_scales + 1) if type(depths) == int else depths
        self.growth_rates = [growth_rates] * (2 * n_scales + 1) if \
            type(growth_rates) == int else growth_rates
        self.drop_rate = drop_rate
        assert len(self.depths) == len(self.growth_rates) == 2 * n_scales + 1
        self.conv_first = nn.utils.spectral_norm(nn.Conv2d(
            3, n_channel_start, 3, stride=1, padding=1, bias=True))
        self.dense_blocks = nn.ModuleList([])
        self.transition_downs = nn.ModuleList([])
        self.transition_ups = nn.ModuleList([])

        nskip = []
        nIn = self.n_channel_start
        for i in range(n_scales):
            self.dense_blocks.append(
                DenseBlock(nIn, self.growth_rates[i], self.depths[i],
                           drop_rate=drop_rate, bottle_neck=bottle_neck))
            nIn += self.growth_rates[i] * self.depths[i]
            nskip.append(nIn)
            self.transition_downs.append(self.get_TD(nIn, drop_rate))

        self.dense_blocks.append(
            DenseBlock(nIn, self.growth_rates[n_scales], self.depths[n_scales],
                       only_new=True, drop_rate=drop_rate,
                       bottle_neck=bottle_neck))
        nIn = self.growth_rates[n_scales] * self.depths[n_scales]

        for i in range(n_scales-1):
            self.transition_ups.append(nn.utils.spectral_norm(nn.ConvTranspose2d(
                nIn, nIn, 4, stride=2, padding=1, bias=True)))
            nIn += nskip.pop()
            self.dense_blocks.append(
                DenseBlock(nIn, self.growth_rates[n_scales + 1 + i],
                           self.depths[n_scales + 1 + i],
                           only_new=True, drop_rate=drop_rate,
                           bottle_neck=bottle_neck))
            nIn = self.growth_rates[n_scales + 1 + i] * \
                self.depths[n_scales + 1 + i]
        # last dense block
        self.transition_ups.append( nn.utils.spectral_norm(nn.ConvTranspose2d(
            nIn, nIn, 4, stride=2, padding=1, bias=True)))
        nIn += nskip.pop()
        self.dense_blocks.append(
            DenseBlock(nIn, self.growth_rates[2 * n_scales],
                       self.depths[2 * n_scales], drop_rate=drop_rate,
                       bottle_neck=bottle_neck))
        nIn += self.growth_rates[2 * n_scales] * \
            self.depths[2 * n_scales]
        self.conv_last = nn.utils.spectral_norm(nn.Conv2d(nIn, 3, 1, bias=True))
        self.logsoftmax = nn.Tanh()
    def forward(self, x):
        input = x
        x = self.conv_first(x)
        skip_connects = []
        # down sample
        for i in range(self.n_scales):
            x = self.dense_blocks[i](x)
            skip_connects.append(x)
            x = self.transition_downs[i](x)
        # bottle neck
        x = self.dense_blocks[self.n_scales](x)
        # up sample
        for i in range(self.n_scales):
            skip = skip_connects.pop()
            TU = self.transition_ups[i]
            # adjust padding
            TU.padding = (((x.size(2) - 1) * TU.stride[0] - skip.size(2)
                           + TU.kernel_size[0] + 1) // 2,
                          ((x.size(3) - 1) * TU.stride[1] - skip.size(3)
                              + TU.kernel_size[1] + 1) // 2)
            x = TU(x, output_size=skip.size())
            x = torch.cat((skip, x), 1)
            x = self.dense_blocks[self.n_scales + 1 + i](x)
        x = self.conv_last(x)
        return self.logsoftmax(x)

    def get_TD(self, nIn, drop_rate):
        layers = [nn.BatchNorm2d(nIn), 
                   nn.ReLU(True), nn.utils.spectral_norm(nn.Conv2d(nIn, nIn, 1, bias=True))]
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


def fcdense_tiny(drop_rate=0):
    return FCDense(2, 6, drop_rate=drop_rate)


def fcdense56_nodrop():
    return FCDense(4, 12, drop_rate=0)

def fcdense56_nodrop1():
    return FCDense1(4, 12, drop_rate=0)

def fcdense56(drop_rate=0.2):
    return FCDense(4, 12, drop_rate=drop_rate)


def fcdense67(drop_rate=0):
    return FCDense(5, 16, drop_rate=drop_rate)


def fcdense103(drop_rate=0.2):
    return FCDense([4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], 16,
                   drop_rate=drop_rate)


def fcdense103_nodrop(drop_rate=0):
    return FCDense([4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], 16,
                   drop_rate=drop_rate)

class FCDense1(nn.Module):

    def __init__(self, depths, growth_rates, n_scales=5, n_channel_start=32,
                 n_classes=3, drop_rate=0, bottle_neck=False):
        super(FCDense1, self).__init__()
        self.n_scales = n_scales
        self.n_classes = n_classes
        self.n_channel_start = n_channel_start
        self.depths = [depths] * \
            (2 * n_scales + 1) if type(depths) == int else depths
        self.growth_rates = [growth_rates] * (2 * n_scales + 1) if \
            type(growth_rates) == int else growth_rates
        self.drop_rate = drop_rate
        assert len(self.depths) == len(self.growth_rates) == 2 * n_scales + 1
        self.conv_first = nn.Conv2d(
            9, n_channel_start, 3, stride=1, padding=1, bias=True)
        self.dense_blocks = nn.ModuleList([])
        self.transition_downs = nn.ModuleList([])
        self.transition_ups = nn.ModuleList([])

        nskip = []
        nIn = self.n_channel_start
        for i in range(n_scales):
            self.dense_blocks.append(
                DenseBlock(nIn, self.growth_rates[i], self.depths[i],
                           drop_rate=drop_rate, bottle_neck=bottle_neck))
            nIn += self.growth_rates[i] * self.depths[i]
            nskip.append(nIn)
            self.transition_downs.append(self.get_TD(nIn, drop_rate))

        self.dense_blocks.append(
            DenseBlock(nIn, self.growth_rates[n_scales], self.depths[n_scales],
                       only_new=True, drop_rate=drop_rate,
                       bottle_neck=bottle_neck))
        nIn = self.growth_rates[n_scales] * self.depths[n_scales]

        for i in range(n_scales-1):
            self.transition_ups.append(nn.ConvTranspose2d(
                nIn, nIn, 4, stride=2, padding=1, bias=True))
            nIn += nskip.pop()
            self.dense_blocks.append(
                DenseBlock(nIn, self.growth_rates[n_scales + 1 + i],
                           self.depths[n_scales + 1 + i],
                           only_new=True, drop_rate=drop_rate,
                           bottle_neck=bottle_neck))
            nIn = self.growth_rates[n_scales + 1 + i] * \
                self.depths[n_scales + 1 + i]
        # last dense block
        self.transition_ups.append(nn.ConvTranspose2d(
            nIn, nIn, 4, stride=2, padding=1, bias=True))
        nIn += nskip.pop()
        self.dense_blocks.append(
            DenseBlock(nIn, self.growth_rates[2 * n_scales],
                       self.depths[2 * n_scales], drop_rate=drop_rate,
                       bottle_neck=bottle_neck))
        nIn += self.growth_rates[2 * n_scales] * \
            self.depths[2 * n_scales]
        self.conv_last = nn.Conv2d(nIn, 3, 1, bias=True)
        self.logsoftmax = nn.Tanh()
        #self.conv1 = nn.Sequential(nn.Conv2d(
           # 1, 64, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        #self.conv2 = nn.Sequential(nn.Conv2d(
           # 128, 128, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        #self.conv3 = nn.Sequential(nn.Conv2d(
            #128, 128, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        #self.conv4 = nn.Sequential(nn.Conv2d(
            #128, 3, kernel_size=5, stride=1, padding=2, bias=False))
    def forward(self, x,c):
        input = x
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = self.conv_first(x)
        skip_connects = []
        # down sample
        for i in range(self.n_scales):
            x = self.dense_blocks[i](x)
            skip_connects.append(x)
            x = self.transition_downs[i](x)
        # bottle neck
        x = self.dense_blocks[self.n_scales](x)
        # up sample
        for i in range(self.n_scales):
            skip = skip_connects.pop()
            TU = self.transition_ups[i]
            # adjust padding
            TU.padding = (((x.size(2) - 1) * TU.stride[0] - skip.size(2)
                           + TU.kernel_size[0] + 1) // 2,
                          ((x.size(3) - 1) * TU.stride[1] - skip.size(3)
                              + TU.kernel_size[1] + 1) // 2)
            x = TU(x, output_size=skip.size())
            x = torch.cat((skip, x), 1)
            x = self.dense_blocks[self.n_scales + 1 + i](x)
        x = self.conv_last(x)
        #x1=0.299*input[:,:1,:,:] + 0.587*input[:,1:2,:,:] + 0.114 *input[:,2:3,:,:]
        #x2=self.conv1(x1)
        #x3=self.conv2(torch.cat((x,x2),dim=1))
        #x4=self.conv3(x3)
        #out=self.conv4(x4)
        return self.logsoftmax(x)

    def get_TD(self, nIn, drop_rate):
        layers = [nn.BatchNorm2d(nIn), nn.ReLU(
            True), nn.Conv2d(nIn, nIn, 1, bias=True)]
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
