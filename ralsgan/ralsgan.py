'''
To run this script, you must have the CATS dataset.
I put the necessary code under this
folder. Steps to follow (ref: @AlexiaJM's github)
1. Download: Cat Dataset (http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd)
(I used a version uploaded by @simoninithomas hence I have
a slightly different version of cats .)
2. Run setting_up_script.sh in same folder as preprocess_cat_dataset.py
and your CAT dataset (open and run manually)
3. mv cats_bigger_than_256x256 "cats/0"
(Imagefolder class requires a subfolder under the given
folder (indicating class)
'''


'''
Relativistic GAN paper (https://arxiv.org/abs/1807.00734)
is very inventive and experimental, hence comes with a
bunch of different versions of the same underlying
idea. I used Ralsgan (relativistic average least squares)
for demonstration as it seems to give the most promising
results for the high-definition outputs with a single shot
network.
A really interesting behavior I observe with this model is 
that although around 5k iterations I see some "catlike" 
images, the images look like noise before that for thousands
of iterations, which is unlike the other GAN losses, where the
quality improvement is more linear. Also note that the proper
cat generations don't come before 10s of thousands of iterations.
'''

import argparse
import numpy

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--n_iters', type=int, default=100e3, help='number of iterations of training')
parser.add_argument('--batch_size', type=int, default=48, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
parser.add_argument('--input_folder', type=str, default="cats", help='source images folder')
parser.add_argument('--img_size', type=int, default=256, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--sample_interval', type=int, default=64, help='interval betwen image samples')
opt = parser.parse_args()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
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


# Appendix D.4., DCGAN for 0.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(opt.latent_dim, 1024, 4, 1, 0), # Fully connected layer via convolution.
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            *convlayer(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, opt.channels, 4, 2, 1),
            nn.Tanh()
        )
        '''
        There is a slight error in v2 of the relativistic gan paper, where
        the architecture goes from 128>64>32 but then 64>3.
        '''


    def forward(self, z):
        z = z.view(-1, opt.latent_dim, 1, 1)
        img = self.model(z)
        return img

# PACGAN2.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *convlayer(opt.channels * 2, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            *convlayer(256, 512, 4, 2, 1, bn=True),
            *convlayer(512, 1024, 4, 2, 1, bn=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),  # FC with Conv.
        )

    def forward(self, imgs):
        critic_value = self.model(imgs)
        critic_value  = critic_value.view(imgs.size(0), -1)
        return critic_value


'''
Worst part of the data science: data (prep). 
I shamelessly copy these from @AlexiaJM's code.
(https://github.com/AlexiaJM/RelativisticGAN) 
'''

transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])
data = datasets.ImageFolder(root=os.path.join(os.getcwd(), opt.input_folder), transform=transform)


def generate_random_sample():
    while True:
        random_indexes = numpy.random.choice(data.__len__(), size=opt.batch_size * 2, replace=False)
        batch = [data[i][0] for i in random_indexes]
        yield torch.stack(batch, 0)


random_sample = generate_random_sample()

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
gan_loss = mse_loss

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

noise_fixed = Variable(Tensor(25, opt.latent_dim).normal_(0, 1), requires_grad=False)

for it in range(int(opt.n_iters)):
    print('Iter. {}'.format(it))

    batch = random_sample.__next__()

    imgs_real = Variable(batch.type(Tensor))
    imgs_real = torch.cat((imgs_real[0:opt.batch_size, ...], imgs_real[opt.batch_size:opt.batch_size * 2, ...]), 1)
    real = Variable(Tensor(batch.size(0)//2, 1).fill_(1.0), requires_grad=False)

    noise = Variable(Tensor(opt.batch_size * 2, opt.latent_dim).normal_(0, 1))
    imgs_fake = generator(noise)
    imgs_fake = torch.cat((imgs_fake[0:opt.batch_size, ...], imgs_fake[opt.batch_size:opt.batch_size * 2, ...]), 1)

    # == Discriminator update == #
    optimizer_D.zero_grad()

    c_xr = discriminator(imgs_real)
    c_xf = discriminator(imgs_fake.detach())

    d_loss = gan_loss(c_xr, torch.mean(c_xf) + real) + gan_loss(c_xf, torch.mean(c_xr) - real)

    d_loss.backward()
    optimizer_D.step()

    # == Generator update == #
    batch = random_sample.__next__()

    imgs_real = Variable(batch.type(Tensor))
    imgs_real = torch.cat((imgs_real[0:opt.batch_size, ...], imgs_real[opt.batch_size:opt.batch_size * 2, ...]), 1)

    noise = Variable(Tensor(opt.batch_size * 2, opt.latent_dim).normal_(0, 1))
    imgs_fake = generator(noise)
    imgs_fake = torch.cat((imgs_fake[0:opt.batch_size, ...], imgs_fake[opt.batch_size:opt.batch_size * 2, ...]), 1)

    c_xr = discriminator(imgs_real)
    c_xf = discriminator(imgs_fake)
    real = Variable(Tensor(batch.size(0)//2, 1).fill_(1.0), requires_grad=False)

    optimizer_G.zero_grad()

    g_loss = gan_loss(c_xf, torch.mean(c_xr) + real) + gan_loss(c_xr, torch.mean(c_xf) - real)

    g_loss.backward()
    optimizer_G.step()

    if vis:
        if it % opt.sample_interval == 0:

            # Keep a record of losses for plotting.
            epochs.append(it)
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
                    'xlabel': 'iteration',
                    'ylabel': 'loss',
                    'width': 512,
                    'height': 512
                },
                win=1)
            vis.images(
                imgs_fake_fixed,
                nrow=5, win=2,
                opts={
                    'title': 'GAN output [Iteration {}]'.format(it),
                    'width': 512,
                    'height': 512,
                }
            )