import torch
from torch import nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

'''
Trying to make it compact, I'm afraid
this u-net implementation became very
messy. In future, simplicity before 
compactnesss.
'''

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        ##### Encoder #####
        def convblock(n_input, n_output, k_size=4, stride=2, padding=1, normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        g_layers = [64, 128, 256, 512, 512, 512, 512, 512]  # Generator feature maps (ref: pix2pix paper)
        self.encoder = nn.ModuleList(convblock(in_channels, g_layers[0], normalize=False))  # 1st layer is not normalized.
        for iter in range(1, len(g_layers)):
            self.encoder += convblock(g_layers[iter-1], g_layers[iter])
        # Ex.: Why use ModuleList instead of python lists?

        ##### Decoder #####
        def convdblock(n_input, n_output, k_size=4, stride=2, padding=1, dropout=0.0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output)
            ]
            if dropout:  # Mixing BN and Dropout is not recommended in general.
                block.append(nn.Dropout(dropout))
            block.append(nn.ReLU(inplace=True))
            return block

        d_layers = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.decoder = nn.ModuleList(convdblock(g_layers[-1], d_layers[0]))
        for iter in range(1, len(d_layers)-1):
            self.decoder += convdblock(d_layers[iter], d_layers[iter+1]//2)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(d_layers[-1], out_channels, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, z):
        # Encode.
        e = [self.encoder[1](self.encoder[0](z))]
        module_iter = 2
        for iter in range(1, 8):  # we have 8 downsampling layers
            result = self.encoder[module_iter+0](e[iter-1])
            result = self.encoder[module_iter+1](result)
            result = self.encoder[module_iter+2](result)
            e.append(result)
            module_iter += 3
        # Decode.
        # First d-layer.
        d1 = self.decoder[2](self.decoder[1](self.decoder[0](e[-1])))
        d1 = torch.cat((d1, e[-2]), dim=1)
        d = [d1]
        module_iter = 3
        for iter in range(1, 7):  # we have 7 upsampling layers
            result = self.decoder[module_iter+0](d[iter-1])
            result = self.decoder[module_iter+1](result)
            result = self.decoder[module_iter+2](result)
            result = torch.cat((result, e[-(iter+2)]), dim=1)  # Concating n-i^th layer with i^th.
            d.append(result)
            module_iter += 3

        # Pass the decoder output through a "flattening" layer to get the image.
        img = self.model(d[-1])
        return img


class Discriminator(nn.Module):
    def __init__(self, a_channels=3, b_channels=3):
        super(Discriminator, self).__init__()

        def convblock(n_input, n_output, k_size=4, stride=2, padding=1, normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *convblock(a_channels + b_channels, 64, normalize=False),
            *convblock(64, 128),
            *convblock(128, 256),
            *convblock(256, 512),
        )

        self.l1 = nn.Linear(512 * 16 * 16, 1)

    def forward(self, img_A, img_B):
        img = torch.cat((img_A, img_B), dim=1)
        conv_out = self.model(img)
        conv_out = conv_out.view(img_A.size(0), -1)
        prob = self.l1(conv_out)
        prob = F.sigmoid(prob)
        return prob