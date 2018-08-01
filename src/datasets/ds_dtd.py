import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class DataSetDTD(object):
    """
    Class manage DTD data-set
    """

    def __init__(self,
                 path_data,
                 num_dunkeys=4,
                 batch_size_train=100,
                 batch_size_val=100,
                 fin_scale=32):

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        init_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Resize(fin_scale),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean,
                                 std=imagenet_std)
        ])

        self.transforms = {
            'train': init_transform,
            'val': init_transform
        }

        self.dataset = {
            'train': dsets.ImageFolder(root=os.path.join(path_data, 'train'),
                                       transform=self.transforms['val'],
                                       target_transform=None),
            'val': dsets.ImageFolder(root=os.path.join(path_data, 'test'),
                                     transform=self.transforms['val'],
                                     target_transform=None)
        }

        self.loader = {
            'train': torch.utils.data.DataLoader(dataset=self.dataset['train'],
                                                 batch_size=batch_size_train,
                                                 shuffle=True,
                                                 num_workers=num_dunkeys),
            'val': torch.utils.data.DataLoader(dataset=self.dataset['val'],
                                               batch_size=batch_size_val,
                                               shuffle=False,
                                               num_workers=num_dunkeys)
        }
