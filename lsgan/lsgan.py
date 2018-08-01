'''
Using Eqn. 9 of the lsgan paper as the loss function.
Figure 2 of paper's v1 is used as the gen/discr. combo.
Note that there is a v3 (as of 2018, July).
A significant design difference is the use of BN before ReLU
unlike the regular GANs (ReLU > BN), (BN > ReLU).
'''

import argparse

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--sample_interval', type=int, default=64, help='interval betwen image samples')
opt = parser.parse_args()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


try:
    import visdom
    vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True) # Create vis env.
except ImportError:
    vis = None
else:
    vis.close(None) # Clear all figures.


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def convlayer(n_input, n_output, k_size=5, stride=2, padding=0, output_padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False, output_padding=output_padding),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True),
            ]
            return block

        # MNIST (not constantinople.. sorry not LSUN)
        self.model = nn.Sequential(
            *convlayer(opt.latent_dim, 128, 7, 1, 0),                       # Fully connected, 128 x 7 x 7
            *convlayer(128, 64, 5, 2, 2, output_padding=1),                 # 64 x 14 x 14
            *convlayer(64, 32, 5, 2, 2, output_padding=1),                  # 32 x 28 x 28
            nn.ConvTranspose2d(32, opt.channels, 5, 1, 2),                  # 1 x 28 x 28
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, normalize=True, dilation=1):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False, dilation=dilation),]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.conv_block = nn.Sequential(                                # 1 x 28 x 28
            *convlayer(opt.channels, 32, 5, 2, 2, normalize=False),     # 32 x 14 x 14
            *convlayer(32, 64, 5, 2, 2),                                # 64 x 7 x 7
        )

        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img):
        conv_out = self.conv_block(img)
        conv_out = conv_out.view(img.size(0), 64 * 7 * 7)
        l2_value = self.fc_block(conv_out)
        l2_value = l2_value.unsqueeze_(dim=2).unsqueeze_(dim=3)
        return l2_value


def mnist_data():
    compose = transforms.Compose(
        [transforms.Resize(opt.img_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    output_dir = './data/mnist'
    return datasets.MNIST(root=output_dir, train=True,
                          transform=compose, download=True)

mnist = mnist_data()
batch_iterator = DataLoader(mnist, shuffle=True, batch_size=opt.batch_size) # List, NCHW format.


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
gan_loss = nn.MSELoss() # LSGAN's loss, a shortcut to torch.mean( (x-y) ** 2 )

generator = Generator()
discriminator = Discriminator()

optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Loss record.
g_losses = []
d_losses = []
epochs = []
loss_legend = ['Discriminator', 'Generator']

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

noise_fixed = Variable(Tensor(25, opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False) # To track the progress of the GAN.

for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch, _) in enumerate(batch_iterator):

        real = Variable(Tensor(batch.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)

        imgs_real = Variable(batch.type(Tensor), requires_grad=False)
        noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False)
        imgs_fake = generator(noise) # Variable(generator(noise), requires_grad=False)


        # == Discriminator update == #
        optimizer_D.zero_grad()

        d_loss = 0.5*gan_loss(discriminator(imgs_real), real) + \
                 0.5*gan_loss(discriminator(imgs_fake.data), fake)

        d_loss.backward()
        optimizer_D.step()


        # == Generator update == #
        noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False)
        imgs_fake = generator(noise)

        optimizer_G.zero_grad()

        g_loss = 0.5*gan_loss(discriminator(imgs_fake), real)

        g_loss.backward()
        optimizer_G.step()

        if vis:
            batches_done = epoch * len(batch_iterator) + i
            if batches_done % opt.sample_interval == 0:

                # Keep a record of losses for plotting.
                epochs.append(epoch + i/len(batch_iterator))
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                # Generate images for a given set of fixed noise
                # so we can track how the GAN learns.
                imgs_fake_fixed = generator(noise_fixed).detach().data
                imgs_fake_fixed = imgs_fake_fixed.add_(1).div_(2) # To normalize and display on visdom.

                # Display results on visdom page.
                vis.line(
                    X=torch.stack([Tensor(epochs)] * len(loss_legend), dim=1),
                    Y=torch.stack((Tensor(d_losses), Tensor(g_losses)), dim=1),
                    opts={
                        'title': 'loss over time',
                        'legend': loss_legend,
                        'xlabel': 'epoch',
                        'ylabel': 'loss',
                        'width': 512,
                        'height': 512
                },
                    win=1)
                vis.images(
                    imgs_fake_fixed,
                    nrow=5, win=2,
                    opts={
                        'title': 'GAN output [Epoch {}]'.format(epoch),
                        'width': 512,
                        'height': 512,
                    }
                )
