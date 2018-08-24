import torch
from torch import nn
from torchvision.models import vgg19
from torch.autograd import Variable


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, n_output=64, k_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_output, n_output, k_size, stride, padding),
            nn.BatchNorm2d(n_output),
            nn.PReLU(),
            nn.Conv2d(n_output, n_output, k_size, stride, padding),
            nn.BatchNorm2d(n_output),
        )

    def forward(self, x):
        return x + self.model(x)

class ShuffleBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size=3, stride=1, padding=1):
        super(ShuffleBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_input, n_output, k_size, stride, padding),  # N, 256, H, W
            nn.PixelShuffle(2),  # N, 64, 2H, 2W
            nn.PReLU(),
        )
        '''
        Input: :math:`(N, C * upscale_factor^2, H, W)`
        Output: :math:`(N, C, H * upscale_factor, W * upscale_factor)`
        '''

    def forward(self, x):
        return self.model(x)


# n_fmap = number of feature maps,
# B = number of cascaded residual blocks.
class Generator(nn.Module):
    def __init__(self, n_input=3, n_output=3, n_fmap=64, B=16):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(n_input, n_fmap, 9, 1, 4),
            nn.PReLU(),
        )

        # A cascaded of B residual blocks.
        self.R = []
        for _ in range(B):
            self.R.append(ResidualBlock(n_fmap))
        self.R = nn.Sequential(*self.R)

        self.l2 = nn.Sequential(
            nn.Conv2d(n_fmap, n_fmap, 3, 1, 1),
            nn.BatchNorm2d(n_fmap),
        )

        self.px = nn.Sequential(
            ShuffleBlock(64, 256),
            ShuffleBlock(64, 256),
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(64, n_output, 9, 1, 4),
            nn.Tanh(),
        )


    def forward(self, img_in):
        out_1 = self.l1(img_in)
        out_2 = self.R(out_1)
        out_3 = out_1 + self.l2(out_2)
        out_4 = self.px(out_3)
        return self.conv_final(out_4)


class Discriminator(nn.Module):
    def __init__(self, lr_channels=3):
        super(Discriminator, self).__init__()

        def convblock(n_input, n_output, k_size=3, stride=1, padding=1, bn=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.conv = nn.Sequential(
            *convblock(lr_channels, 64, 3, 1, 1, bn=False),
            *convblock(64, 64, 3, 2, 1),
            *convblock(64, 128, 3, 1, 1),
            *convblock(128, 128, 3, 2, 1),
            *convblock(128, 256, 3, 1, 1),
            *convblock(256, 256, 3, 2, 1),
            *convblock(256, 512, 3, 1, 1),
            *convblock(512, 512, 3, 2, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out_1 = self.conv(img)
        out_1 = out_1.view(img.size(0), -1)
        out_2 = self.fc(out_1)
        return out_2


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        model = vgg19(pretrained=True)

        children = list(model.features.children())
        max_pool_indices = [index for index, m in enumerate(children) if isinstance(m, nn.MaxPool2d)]
        target_features = children[:max_pool_indices[4]]
        '''
          We use vgg-5,4 which is the layer output after 5th conv 
          and right before the 4th max pool.
        '''
        self.features = nn.Sequential(*target_features)
        for p in self.features.parameters():
            p.requires_grad = False

        '''
        # VGG means and stdevs on pretrained imagenet
        mean = -1 + Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        std = 2*Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # This is for cuda compatibility.
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        '''

    def forward(self, input):
        # input = (input - self.mean) / self.std
        output = self.features(input)
        return output