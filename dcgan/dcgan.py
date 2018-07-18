import argparse
import os

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--sample_interval', type=int, default=64, help='interval betwen image samples')
opt = parser.parse_args()

# Just trying if these different initializations
# matter that much.
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

try:
    import visdom
    vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True) # Create vis env.
except ImportError:
    vis = None
else:
    vis.close(None) # Clear all figures.


img_dims = (opt.channels, opt.img_size, opt.img_size)
n_features = opt.channels * opt.img_size * opt.img_size

# Model is taken from Fig. 1 of the DCGAN paper.
# BN before ReLU? https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
# I tried BN -> ReLU but got poor results.
# Here I skip one layer of DCGAN because I use CIFAR10 (32x32)
# instead of LSUN (64x64). LSUN is too large to load and train (40 GB for a single bedroom category)

'''
UPDATE: Both disc. and generator made 
fully convolutional, unlike the original
version of the paper. The code is commented 
out for reference. This is the standard practice
in most of the networks today.
'''

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(n_output),
            ]
            return block

        self.project = nn.Sequential(
            nn.Linear(opt.latent_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # pytorch doesn't have padding='same' as keras does.
        # hence we have to pad manually to get the desired behavior.
        # from the manual:
        # o = (i−1)∗s−2p+k, i/o = in/out-put, k = kernel size, p = padding, s = stride.

        self.model = nn.Sequential(
            *convlayer(opt.latent_dim, 256, 4, 1, 0), # Fully connected layer via convolution.
            *convlayer(256, 128, 4, 2, 1), # 4->8, (4-1)*2-2p+4 = 8, p = 1
            *convlayer(128, 64, 4, 2, 1), # 8->16, p = 1
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1),
            nn.Tanh()
        )
        # Tanh > Image values are between [-1, 1]

    def forward(self, z):
        # p = self.project(z)
        # p = p.view(-1, 256, 4, 4) # Project and reshape (notice that pytorch uses NCHW format)
        img = self.model(z)
        img = img.view(z.size(0), *img_dims)
        return img

# Discriminator mirrors the generator, like and encoder-decoder in reverse.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Again, we must ensure the same padding.
        # From the manual:
        # o = (i+2p−d∗(kernel_size−1)−1)/s+1, d = dilation (default = 1).

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels, 64, 4, 2, 1, bias=False), # 32-> 16, (32+2p-3-1)/2 + 1 = 16, p = 1
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(64), # IS IT CRUCIAL TO SKIP BN AT FIRST LAYER OF DISCRIMINATOR?
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False), # FC with Conv.
            nn.Sigmoid()
        )

        # we used 3 convolutional blocks, hence 2^3 = 8 times downsampling.
        semantic_dim = opt.img_size // (2**3)

        self.l1 = nn.Sequential(
            nn.Linear(256 * semantic_dim**2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        prob = self.model(img)
        # prob = self.l1(prob.view(prob.size(0), -1))
        return prob


# Beware: CIFAR10 download is slow.
def custom_data(CIFAR_CLASS = 5): # Only working with dogs.
    compose = transforms.Compose(
        [transforms.Resize(opt.img_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])
    output_dir = './data/cifar10'
    cifar = datasets.CIFAR10(root=output_dir, download=True, train=True,
                          transform=compose)
    # return cifar

    # Our custom cifar with a single category.
    Tensor__ = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    features = Tensor__(5000, 3, 32, 32) # 5k samples in each category.
    targets =  Tensor__(5000, 1)

    # Go through CIFAR to pick only the desired class..
    it = 0
    for i in range(cifar.__len__()):
        sample = cifar.__getitem__(i)
        if sample[1] == CIFAR_CLASS:
            features[it, ...] = sample[0]
            targets[it] = sample[1]
            it += 1
    return features, targets

features, targets = custom_data(CIFAR_CLASS=1) # Cars.
batch_iterator = DataLoader(torch.utils.data.TensorDataset(features, targets), shuffle=True, batch_size=opt.batch_size) # List, NCHW format.

# Save a batch of real images for reference.
os.makedirs('./out', exist_ok=True)
save_image(next(iter(batch_iterator))[0][:25, ...], './out/real_samples.png', nrow=5, normalize=True)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
gan_loss = nn.BCELoss()

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

generator.apply(weights_init_xavier)
discriminator.apply(weights_init_xavier)

noise_fixed = Variable(Tensor(25, opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False) # To track the progress of the GAN.

for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch, _) in enumerate(batch_iterator):

        real = Variable(Tensor(batch.size(0), 1, 1, 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(batch.size(0), 1, 1, 1).fill_(0), requires_grad=False)

        imgs_real = Variable(batch.type(Tensor), requires_grad=False)
        noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False)
        imgs_fake = Variable(generator(noise), requires_grad=False)


        # == Discriminator update == #
        optimizer_D.zero_grad()

        # A small reminder: given classes c, prob. p, - c*logp - (1-c)*log(1-p) is the BCE/GAN loss.
        # Putting D(x) as p, x=real's class as 1, (..and same for fake with c=0) we arrive to BCE loss.
        # This is intuitively how well the discriminator can distinguish btw real & fake.
        d_loss = gan_loss(discriminator(imgs_real), real) + \
                 gan_loss(discriminator(imgs_fake), fake)

        d_loss.backward()
        optimizer_D.step()


        # == Generator update == #
        noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False)
        imgs_fake = generator(noise)

        optimizer_G.zero_grad()

        # Minimizing (1-log(d(g(noise))) is less stable than maximizing log(d(g)) [*].
        # Since BCE loss is defined as a negative sum, maximizing [*] is == minimizing [*]'s negative.
        # Intuitively, how well does the G fool the D?
        g_loss = gan_loss(discriminator(imgs_fake), real)

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
