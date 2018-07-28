from torch import nn
import torch.nn.functional as F
# from torch.legacy.nn import Identity

# Residual network.
# CT-GAN basically uses the same architecture WGAN-GP uses,
# but with a different loss.
class MeanPoolConv(nn.Module):
    def __init__(self, n_input, n_output, k_size, kaiming_init=True):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        if kaiming_init:
            nn.init.kaiming_uniform_(conv1.weight, mode='fan_in', nonlinearity='relu')
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4.0
        out = self.model(out)
        return out

class ConvMeanPool(nn.Module):
    def __init__(self, n_input, n_output, k_size, kaiming_init=True):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)

        if kaiming_init:
            nn.init.kaiming_uniform_(conv1.weight, mode='fan_in', nonlinearity='relu')

        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = self.model(x)
        out = (out[:,:,::2,::2] + out[:,:,1::2,::2] + out[:,:,::2,1::2] + out[:,:,1::2,1::2]) / 4.0
        return out

class UpsampleConv(nn.Module):
    def __init__(self, n_input, n_output, k_size, kaiming_init=True):
        super(UpsampleConv, self).__init__()

        conv_layer = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        if kaiming_init:
            nn.init.kaiming_uniform_(conv_layer.weight, mode='fan_in', nonlinearity='relu')

        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            conv_layer,
        )

    def forward(self, x):
        x = x.repeat((1, 4, 1, 1)) # Weird concat of WGAN-GPs upsampling process.
        out = self.model(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size, kaiming_init=True)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size, kaiming_init=True)
            self.out_dim = n_output

            # Weight initialization.
            nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size, kaiming_init=True)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size, kaiming_init=True)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim] # Define the dimensions for layer normalization.

            nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = None # Identity
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]

            nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

        self.model = nn.Sequential(
            nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv1,
            nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv2,
        )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)

class DiscBlock1(nn.Module):
    def __init__(self, n_output):
        super(DiscBlock1, self).__init__()

        self.conv1 = nn.Conv2d(3, n_output, 3, padding=(3-1)//2)
        self.conv2 = ConvMeanPool(n_output, n_output, 1, kaiming_init=True)
        self.conv_shortcut = MeanPoolConv(3, n_output, 1, kaiming_init=False)

        # Weight initialization:
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2
        )

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(                     # 128 x 1 x 1
            nn.ConvTranspose2d(128, 128, 4, 1, 0),      # 128 x 4 x 4
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 8 x 8
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 16 x 16
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=(3-1)//2),     # 3 x 32 x 32 (no kaiming init. here)
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        n_output = 128
        '''
        This is a parameter but since we experiment with a single size
        of 3 x 32 x 32 images, it is hardcoded here.
        '''

        self.DiscBlock1 = DiscBlock1(n_output)                      # 128 x 16 x 16
        self.block1 = nn.Sequential(
            ResidualBlock(n_output, n_output, 3, resample='down', bn=False, spatial_dim=16),  # 128 x 8 x 8
        )
        self.block2 = nn.Sequential(
            ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=8),    # 128 x 8 x 8
        )
        self.block3 = nn.Sequential(
            ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=8),    # 128 x 8 x 8
        )

        self.l1 = nn.Sequential(nn.Linear(128, 1))                  # 128 x 1

    def forward(self, x, dropout=0.0, intermediate_output=False):
        # x = x.view(-1, 3, 32, 32)
        y = self.DiscBlock1(x)
        y = self.block1(y)
        y = F.dropout(y, training=True, p=dropout)
        y = self.block2(y)
        y = F.dropout(y, training=True, p=dropout)
        y = self.block3(y)
        y = F.dropout(y, training=True, p=dropout)
        y = F.relu(y)
        y = y.view(x.size(0), 128, -1)
        y = y.mean(dim=2)
        critic_value = self.l1(y).unsqueeze_(1).unsqueeze_(2) # or *.view(x.size(0), 128, 1, 1, 1)

        if intermediate_output:
            return critic_value, y # y is the D_(.), intermediate layer given in paper.

        return critic_value