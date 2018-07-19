from torchvision import transforms, datasets

def load_cifar10(img_size):
    compose = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])
    output_dir = './data/cifar10'
    cifar = datasets.CIFAR10(root=output_dir, download=True, train=True,
                             transform=compose)
    return cifar

def load_mnist(img_size):
    compose = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    output_dir = './data/mnist'
    return datasets.MNIST(root=output_dir, train=True,
                          transform=compose, download=True)
