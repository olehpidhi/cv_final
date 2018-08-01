import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class DataSetCifar10(object):
    """
    Class manage CIFAR10 data-set
    """

    def __init__(self,
                 path_data,
                 num_dunkeys=4,
                 batch_size_train=100,
                 batch_size_val=100,
                 download=False):
        im_mean = [0.5, 0.5, 0.5]
        im_std = [0.5, 0.5, 0.5]

        init_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=im_mean,
                                 std=im_std)
        ])

        self.transforms = {
            'train': init_transform,
            'val': init_transform
        }

        self.dataset = {
            'train': dsets.CIFAR10(root=path_data,
                                   train=True,
                                   download=download,
                                   transform=self.transforms['train']),
            'val': dsets.CIFAR10(root=path_data,
                                 train=False,
                                 download=download,
                                 transform=self.transforms['val'])
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

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
