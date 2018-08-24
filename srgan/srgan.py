'''
Literally, the 90% of my time was spent
looking for a HD dataset that I can use
for this GAN. It turns out not a lot of
such sets exist. If you happen to have
a nice custom dataset of say size 1024x
1024, it would fit perfectly for this
code. I settled for "celeba align"
dataset that isn't particularly high
definition, but the memory constraints
forced my hand on this. It is accessible:
https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8
'''

import argparse

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torchvision import transforms
from torch.utils.data import DataLoader

from models import Generator, Discriminator, VGGFeatures, weights_init_normal
from datasets import Dataset

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--img_size', type=int, default=256, help='high resolution image output size')
parser.add_argument('--data_folder', type=str, default="img_align_celeba", help='where the training image pairs are located')
parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--sample_interval', type=int, default=16, help='interval betwen image samples')
opt = parser.parse_args()



try:
    import visdom
    vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True) # Create vis env.
except ImportError:
    vis = None
else:
    vis.close(None) # Clear all figures.


# Celeba dataset loader.
transform_image_hr = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
transform_image_lr = [
    transforms.Resize((opt.img_size//4, opt.img_size//4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
data_train = Dataset(opt.data_folder, transform_image_lr, transform_image_hr, stage='train')
data_val = Dataset(opt.data_folder, transform_image_lr, transform_image_hr, stage='val')

params = {'batch_size': opt.batch_size, 'shuffle': True}
dataloader_train = DataLoader(data_val, **params)

params = {'batch_size': 5, 'shuffle': True}
dataloader_val = DataLoader(data_val, **params)


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

gan_loss = nn.BCELoss()
content_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()
vgg = VGGFeatures()

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
    vgg = vgg.cuda()
    gan_loss = gan_loss.cuda()
    content_loss = content_loss.cuda()


generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch_lr, batch_hr) in enumerate(dataloader_train):

        real = Variable(Tensor(batch_lr.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(batch_lr.size(0), 1).fill_(0), requires_grad=False)

        imgs_real_lr = Variable(batch_lr.type(Tensor))
        imgs_real_hr = Variable(batch_hr.type(Tensor))

        # == Discriminator update == #
        optimizer_D.zero_grad()

        imgs_fake_hr = Variable(generator(imgs_real_lr.detach()))

        d_loss = gan_loss(discriminator(imgs_real_hr), real) + gan_loss(discriminator(imgs_fake_hr), fake)

        d_loss.backward()
        optimizer_D.step()

        # == Generator update == #
        imgs_fake_hr = generator(imgs_real_lr)

        optimizer_G.zero_grad()

        g_loss = (1/12.75) * content_loss(vgg(imgs_fake_hr), vgg(imgs_real_hr.detach())) + 1e-3 * gan_loss(discriminator(imgs_fake_hr), real)

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
                batch_lr, batch_hr = next(iter(dataloader_val))

                imgs_val_lr = Variable(batch_lr.type(Tensor))
                imgs_val_hr = Variable(batch_hr.type(Tensor))
                imgs_fake_hr = generator(imgs_val_lr).detach().data

                # For visualization purposes.
                imgs_val_lr = torch.nn.functional.upsample(imgs_val_lr, size=(imgs_fake_hr.size(2), imgs_fake_hr.size(3)), mode='bilinear')

                imgs_val_lr = imgs_val_lr.mul_(Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)).add_(Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                imgs_val_hr = imgs_val_hr.mul_(Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)).add_(Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                imgs_fake_hr = imgs_fake_hr.add_(torch.abs(torch.min(imgs_fake_hr))).div_(torch.max(imgs_fake_hr)-torch.min(imgs_fake_hr))
                fake_val = torch.cat((imgs_val_lr, imgs_val_hr, imgs_fake_hr), dim=2)

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