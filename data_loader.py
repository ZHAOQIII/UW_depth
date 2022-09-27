from torch.utils import data
import transform as T
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import random
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def get_loader(image_dir, crop_size=128, image_size=128,
               batch_size=8, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        #transform.append(T.RandomHorizontalFlip())
        #transform.append(T.RandomCrop(crop_size))
        transform.append(T.RandomCrop(crop_size))
    #transform.append(T.Resize(crop_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    if mode == 'train':
        datasetA = ImageFolder(image_dir+'/trainA', transform)
        datasetB = ImageFolder(image_dir+'/trainB', transform)
        data_loaderA = data.DataLoader(dataset=datasetA,
                                  batch_size=batch_size,drop_last=True,
                                  shuffle=('train' == mode),
                                  )#num_workers=num_workers
        data_loaderB = data.DataLoader(dataset=datasetB,
                                  batch_size=batch_size,drop_last=True,
                                  shuffle=('train' == mode),
                                  )#num_workers=num_workers
    else:
        datasetA = ImageFolder(image_dir, transform)
        data_loaderA = data.DataLoader(dataset=datasetA,
                                       batch_size=batch_size,
                                       )#num_workers=num_workers
        data_loaderB = []
    return data_loaderA,data_loaderB

