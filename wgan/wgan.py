'''
Wasserstein DCGAN.

Q: Does lower loss indicate better images, i.e.,
does convergence really mean anything? In regular GANs, it doesn't.
A: Kind of. I would say having a steadily decreasing loss is nice, but it's not always a 1-1 mapping.
(related: Many Paths to Equilibrium: GANs Do Not Need to Decrease a Divergence At Every Step)

Q: Is it possible to train a stable network without batch normalization, as claimed?
A: Sort of, batch norm still makes things way faster. Also, there is this practice of
not using BN at the first layer of discriminator which I found to not matter.
'''

import argparse
import os

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset_loader import load_mnist, load_cifar10

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | mnist')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='adam: learning rate')
parser.add_argument('--c', type=float, default=0.01, help='clipping value for the gradients')
parser.add_argument('--n_critic', type=int, default=5, help='number of discriminator updates per each generator update')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--sample_interval', type=int, default=256, help='interval betwen image samples')
opt = parser.parse_args()


try:
    import visdom
    vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True) # Create vis env.
except ImportError:
    vis = None
else:
    vis.close(None) # Clear all figures.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

img_dims = (opt.channels, opt.img_size, opt.img_size)
n_features = opt.channels * opt.img_size * opt.img_size

# DCGAN network.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def convblock(n_input, n_output, k_size=4, stride=2, padding=0, normalize=True):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.ReLU(inplace=True))
            return block

        self.project = nn.Sequential(
            nn.Linear(opt.latent_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.model = nn.Sequential(
            *convblock(opt.latent_dim, 256, 4, 1, 0),
            *convblock(256, 128, 4, 2, 1),
            *convblock(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1),
            nn.Tanh()
        )
        # Tanh > Image values are between [-1, 1]

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), *img_dims)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def convblock(n_input, n_output, kernel_size=4, stride=2, padding=1, normalize=True):
            block =  [nn.Conv2d(n_input, n_output, kernel_size, stride, padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *convblock(opt.channels, 64, 4, 2, 1, normalize=False), # 32-> 16, (32+2p-3-1)/2 + 1 = 16, p = 1, apparently BN at 1st layer is detrimental for WGANs.
            *convblock(64, 128, 4, 2, 1),
            *convblock(128, 256, 4, 2, 1),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False), # FC with Conv.
            nn.Sigmoid()
        )

    def forward(self, img):
        prob = self.model(img)
        return prob


assert (opt.dataset == 'cifar10' or opt.dataset == 'mnist'), 'Unknown dataset! Only cifar10 and mnist are supported.'

if opt.dataset == 'cifar10':
    batch_iterator = DataLoader(load_cifar(opt.img_size), shuffle=True, batch_size=opt.batch_size) # List, NCHW format.
elif opt.dataset == 'mnist':
    batch_iterator = DataLoader(load_mnist(opt.img_size), shuffle=True, batch_size=opt.batch_size) # List, NCHW format.

# Save a batch of real images for reference.
os.makedirs('./out', exist_ok=True)
save_image(next(iter(batch_iterator))[0][:25, ...], './out/real_samples.png', nrow=5, normalize=True)


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
gan_loss = nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

optimizer_D = optim.RMSprop(discriminator.parameters(), lr=opt.lr)
optimizer_G = optim.RMSprop(generator.parameters(), lr=opt.lr)

# Loss record.
g_losses = []
d_losses = []
epochs = []
loss_legend = ['Discriminator', 'Generator']

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

generator.apply(weights_init)
discriminator.apply(weights_init)

noise_fixed = Variable(Tensor(25, opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False) # To track the progress of the GAN.

for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch, _) in enumerate(batch_iterator):

        # == Discriminator update == #
        for iter in range(opt.n_critic):
            # Sample real and fake images.
            imgs_real = Variable(batch.type(Tensor))
            noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1))
            imgs_fake = Variable(generator(noise), requires_grad=False)

            # Update discriminator (or critic, since we don't output probabilities anymore).
            optimizer_D.zero_grad()

            # WGAN utility, we ascend on this hence the loss will be the negative.
            d_loss = -torch.mean(discriminator(imgs_real) - discriminator(imgs_fake))

            d_loss.backward()
            optimizer_D.step()

            # Detrimental clipping of the generator weights,
            # which will be fixed in a few months from this paper.
            for params in discriminator.parameters():
                params.data.clamp_(-opt.c, +opt.c)

        # == Generator update == #
        noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1))
        imgs_fake = generator(noise)

        optimizer_G.zero_grad()

        g_loss = -torch.mean(discriminator(imgs_fake))

        g_loss.backward()
        optimizer_G.step()

        if vis:
            batches_done = epoch * len(batch_iterator) + i
            if batches_done % opt.sample_interval == 0:

                # Keep a record of losses for plotting.
                epochs.append(epoch + i/len(batch_iterator))
                g_losses.append(-g_loss.item()) # Negative because the loss is actually maximized in WGAN.
                d_losses.append(-d_loss.item())

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