from PIL import Image
from torch.utils import data
import glob
import torchvision.transforms as transforms

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_folder, transform, stage):
        'Initialization'
        self.images = glob.glob('{}/{}/*'.format(data_folder, stage))
        self.transform = transforms.Compose(transform)
        'Get image size'
        _, self.img_size = Image.open(self.images[0]).size

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        X = Image.open(self.images[index])
        B = X.crop((0, 0, self.img_size, self.img_size))  # (left, upper, right, lower)
        A = X.crop((self.img_size, 0, self.img_size+self.img_size, self.img_size))

        return self.transform(A), self.transform(B)
