'''
To run, use
bash download_dataset.sh cityscapes
to download the data into the same
directory as the pix2pix.py.
'''

'''
pix2pix paper suggests using a noise
vector along with the real image (to generate
translated images). However, I noticed from
separate experiments that the noise is simply
filtered out, i.e. it does not provide any diversity
for a given A -> B conversion, hence omitted
here. In fact, there is a GAN called BicycleGAN which
addresses the very same issue by employing
autoencoders.
'''

import argparse

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from models import Generator, Discriminator, weights_init_normal
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--lambda_l1', type=int, default=100, help='penalty term for deviation from original images')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--data_folder', type=str, default="cityscapes", help='where the training image pairs are located')
parser.add_argument('--img_size', type=int, default=256, help='size of the input images')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--sample_interval', type=int, default=64, help='interval betwen image samples')
opt = parser.parse_args()


try:
    import visdom
    vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True) # Create vis env.
except ImportError:
    vis = None
else:
    vis.close(None) # Clear all figures.

# Cityscapes dataset loader.
transform_image = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
data_train = Dataset(opt.data_folder, transform_image, stage='train')
data_val = Dataset(opt.data_folder, transform_image, stage='val')

params = {'batch_size': opt.batch_size, 'shuffle': True}
dataloader_train = DataLoader(data_val, **params)

params = {'batch_size': 5, 'shuffle': True}
dataloader_val = DataLoader(data_val, **params)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
gan_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

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
    gan_loss = gan_loss.cuda()
    l1_loss = l1_loss.cuda()


generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch_A, batch_B) in enumerate(dataloader_train):

        real = Variable(Tensor(batch_A.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(batch_A.size(0), 1).fill_(0), requires_grad=False)

        imgs_real_A = Variable(batch_A.type(Tensor))
        imgs_real_B = Variable(batch_B.type(Tensor))

        # == Discriminator update == #
        optimizer_D.zero_grad()

        imgs_fake = Variable(generator(imgs_real_A.detach()))

        d_loss = gan_loss(discriminator(imgs_real_A, imgs_real_B), real) + gan_loss(discriminator(imgs_real_A, imgs_fake), fake)

        d_loss.backward()
        optimizer_D.step()

        # == Generator update == #
        imgs_fake = generator(imgs_real_A)

        optimizer_G.zero_grad()

        g_loss = gan_loss(discriminator(imgs_real_A, imgs_fake), real) + opt.lambda_l1 * l1_loss(imgs_fake, imgs_real_B)

        g_loss.backward()
        optimizer_G.step()

        if vis:
            batches_done = epoch * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:

                # Keep a record of losses for plotting.
                epochs.append(epoch + i/len(dataloader_train))
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())



                # Generate 5 images from the validation set.
                batch_A, batch_B = next(iter(dataloader_val))

                imgs_val_A = Variable(batch_A.type(Tensor))
                imgs_val_B = Variable(batch_B.type(Tensor))
                imgs_fake_B = generator(imgs_val_A).detach().data

                # For visualization purposes.
                imgs_val_A = imgs_val_A.add_(1).div_(2)
                imgs_fake_B = imgs_fake_B.add_(1).div_(2)
                fake_val = torch.cat((imgs_val_A, imgs_fake_B), dim=2)


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
                    fake_val,
                    nrow=5, win=2,
                    opts={
                        'title': 'GAN output [Epoch {}]'.format(epoch),
                        'width': 512,
                        'height': 512,
                    }
                )
