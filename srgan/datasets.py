from PIL import Image
from torch.utils import data
import glob
import torchvision.transforms as transforms
import numpy as np

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_folder, transform_lr, transform_hr, stage):
        'Initialization'
        file_list = glob.glob('{}/*'.format(data_folder))
        n = len(file_list)
        train_size = np.floor(n * 0.8).astype(np.int)
        self.images = file_list[:train_size] if stage is 'train' else file_list[train_size:]

        self.transform_lr = transforms.Compose(transform_lr)
        self.transform_hr = transforms.Compose(transform_hr)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        hr = Image.open(self.images[index])

        return self.transform_lr(hr), self.transform_hr(hr)