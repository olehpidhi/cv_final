import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from  torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class DataSetPlantVillage(object):

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
                                       target_transform=None)
        }

        num_train = len(self.dataset["train"])
        indices = list(range(num_train))

        validation_idx = np.random.choice(indices, size=int(0.2 * num_train), replace=False)
        train_idx = list(set(indices) - set(validation_idx))

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)

        self.loader = {
            'train': torch.utils.data.DataLoader(dataset=self.dataset['train'],
                                                 batch_size=batch_size_train,
                                                 shuffle=False,
                                                 num_workers=num_dunkeys,
                                                 sampler=train_sampler),
            'val': torch.utils.data.DataLoader(dataset=self.dataset['train'],
                                               batch_size=batch_size_val,
                                               shuffle=False,
                                               num_workers=num_dunkeys,
                                               sampler=validation_sampler)
        }
