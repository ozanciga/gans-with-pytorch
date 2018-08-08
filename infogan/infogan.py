'''
IMO, the presentation of this paper is really poor and
unnecessarily convoluted. For example, the paper clearly
states " lRELU. stride 2. batchnorm" which I take it to be
' relu > batch norm'. However, in OpenAI implementation,
the network has been constructed as BN > R.
'''

'''
Note to self: Learn the use cases of keywords such as
variable or requires_grad, .detach(), .data ...
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
parser.add_argument('--d_lr', type=float, default=0.0002, help='adam: learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=0.001, help='adam: learning rate for generator')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=74, help='dimensionality of the latent space (including one-hot encoding of digits and the continuous latent code variables c_1, c_2)')
parser.add_argument('--lambda_c', type=float, default=0.1, help='continous loss penalty, set it so the discrete, continuous and gan losses are at the same scale')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
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

def to_categorical(y):
    """ 1-hot encodes a tensor """
    # return Tensor(torch.eye(10)[y])
    y = LongTensor(y).view(-1, 1)
    y_onehot = Tensor(y.size(0), 10)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot

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

        self.conv_block = nn.Sequential(
            *convlayer(opt.latent_dim, 1024, 1, 1),                     # 1024 x 1 x 1
            *convlayer(1024, 128, 7, 1, 0),                             # 128 x 7 x 7
            *convlayer(128, 64, 4, 2, 1),                               # 64 x 14 x 14
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1, bias=False),  # 1 x 28 x 28
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, opt.latent_dim, 1, 1)
        img = self.conv_block(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.1, inplace=True))
            return block

        self.model = nn.Sequential(                                     # 1 x 28 x 28
            *convlayer(opt.channels, 64, 4, 2, 1, normalize=False),     # 64 x 14 x 14
            *convlayer(64, 128, 4, 2, 1),                               # 128 x 7 x 7
            *convlayer(128, 1024, 7, 1, 0),                             # 1024 x 1 x 1
        )

        # Regular probability of pertaining to real distribution.
        self.d_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        # Continuous.
        self.q_head_C = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2)
        )

        # Discrete (digits).
        self.q_head_D = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

        '''self.q_head_C_mu = nn.Sequential(
            nn.Linear(128, 2)
        )
        self.q_head_C_std = nn.Sequential(
            nn.Linear(128, 2)
        )'''


    def forward(self, img):
        conv_out = self.model(img)
        conv_out = conv_out.squeeze(dim=3).squeeze(dim=2)
        prob = self.d_head(conv_out)
        # Continuous output parameters.
        q = self.q_head_C(conv_out)
        # mu, std = self.q_head_C_mu(q), self.q_head_C_std(q).exp()
        # Discrete outputs.
        digit_probs = self.q_head_D(conv_out)

        return prob, digit_probs, q # mu, std



class GaussianLoss():  # loss pertaining to q(c|x)
    def __call__(self, mu_, std_, x):
        eps = 1e-6
        epsilon = ((x - mu_) / (std_ + eps)) ** 2
        loss = (std_.log() + 0.5 * epsilon)
        loss = torch.mean(loss)

        return loss

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
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

gan_loss = nn.BCELoss()
discrete_loss = nn.NLLLoss()
continuous_loss = nn.MSELoss() # GaussianLoss()
'''
 MSE instead of GaussianLoss is used for reason described in the blog post 
 (http://aiden.nibali.org/blog/2016-12-01-implementing-infogan/) 
 However, I left Gaussian here too since the paper uses that.
'''

generator = Generator()
discriminator = Discriminator()

optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
optimizer_G = optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))

# Loss record.
g_losses = []
d_losses = []
epochs = []
loss_legend = ['Discriminator', 'Generator']

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    gan_loss = gan_loss.cuda()
    discrete_loss = discrete_loss.cuda()
    continuous_loss = continuous_loss.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

noise = Variable(Tensor(100, opt.latent_dim - 10 - 2).normal_(0, 1), requires_grad=False)
digits = Variable(to_categorical([i for i in range(10) for j in range(10)]), requires_grad=False)
c1 = Variable(Tensor(torch.linspace(-1, 1, 10).repeat(10,).data.numpy()).unsqueeze_(1), requires_grad=False)
c2 = Variable(Tensor(100, 1).fill_(0), requires_grad=False)
z_fixed0 = torch.cat((noise, digits, c2, c2), dim=-1)
z_fixed1 = torch.cat((noise, digits, c1, c2), dim=-1)
z_fixed2 = torch.cat((noise, digits, c2, c1), dim=-1)



for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch, _) in enumerate(batch_iterator):

        real = Variable(Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch.size(0), 1).fill_(0.0), requires_grad=False)

        imgs_real = Variable(batch.type(Tensor))


        # == Discriminator update == #
        optimizer_D.zero_grad()

        noise = Variable(Tensor(batch.size(0), opt.latent_dim - 10 - 2).normal_(0, 1))
        digits = to_categorical(list(torch.randint(0, 9, (batch.size(0), ))))
        cis = Variable(Tensor(batch.size(0), 2).uniform_(-1, 1))  # c_i's (continuous variables)

        z = torch.cat((noise, digits, cis), dim=1)  # [62 noise parameters | 10 one-hot encoding params | c_1 | c_2]

        imgs_fake = generator(z)

        prob_real, _, _ = discriminator(imgs_real)
        prob_fake, _, _ = discriminator(imgs_fake.data)

        d_loss = 0.5*gan_loss(prob_real, real) + 0.5*gan_loss(prob_fake, fake)

        d_loss.backward()
        optimizer_D.step()

        # == Generator update == #
        optimizer_G.zero_grad()

        d_labels = torch.randint(0, 9, (batch.size(0), )).data.numpy()   # Discrete labels (0..9).
        c_labels = Variable(Tensor(batch.size(0), 2).uniform_(-1, 1), requires_grad=False)
        d_targets = Variable(LongTensor(d_labels), requires_grad=False)   # Discrete targets.

        noise = Variable(Tensor(batch.size(0), opt.latent_dim - 10 - 2).normal_(0, 1))
        digits = Variable(to_categorical(d_labels))

        z = torch.cat((noise, digits, c_labels), dim=1)  # [62 noise parameters | 10 one-hot encoding params | c_1 | c_2]

        imgs_fake = generator(z)
        prob_fake, logits, q = discriminator(imgs_fake)

        # Mutual information added loss of InfoGAN.
        g_vanilla_loss = gan_loss(prob_fake, real)
        g_discrete_loss = discrete_loss(logits, d_targets)  # labels, logits.. takes me back... to the horrific tensorflow days.
        g_continuous_loss = continuous_loss(q, c_labels)
        g_loss = g_vanilla_loss + g_discrete_loss + opt.lambda_c*g_continuous_loss

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
                imgs_fake_fixed0 = generator(z_fixed0).data.add_(1).div_(2)
                imgs_fake_fixed1 = generator(z_fixed1).data.add_(1).div_(2)
                imgs_fake_fixed2 = generator(z_fixed2).data.add_(1).div_(2)

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
                    imgs_fake_fixed1,
                    nrow=10, win=2,
                    opts={
                        'title': '(Only vary c1) GAN output [Epoch {}]'.format(epoch),
                        'width': 512,
                        'height': 512,
                    }
                )
                vis.images(
                    imgs_fake_fixed2,
                    nrow=10, win=3,
                    opts={
                        'title': '(Only vary c2) GAN output [Epoch {}]'.format(epoch),
                        'width': 512,
                        'height': 512,
                    }
                )
                vis.images(
                    imgs_fake_fixed0,
                    nrow=10, win=4,
                    opts={
                        'title': '(Nothing is varied) GAN output [Epoch {}]'.format(epoch),
                        'width': 512,
                        'height': 512,
                    }
                )
