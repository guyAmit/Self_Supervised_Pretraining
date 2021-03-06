import os
import random

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Dataset


class ViewsDataset(Dataset):
    def __init__(self, dataset, img_size, args):
        '''
        dataset: a pytorh dataset without a transform
        transform: transform sequanace
        '''
        self.dataset = dataset
        self.n_views = args.n_views
        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=2),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4,
                                       hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.__getitem__(idx)
        imgs = [self.transform(img) for _ in range(self.n_views)]
        return imgs


class InPaintingDataset(Dataset):
    def __init__(self, dataset, img_size, args):
        '''
        dataset: a pytorh dataset without a transform
        img_size: the size of the images
        mask_size: int - the size of the mask
        transform: transform sequanace
        '''
        self.dataset = dataset
        self.img_size = img_size
        self.mask_size = args.mask_size
        self.h_mask_size = args.mask_size//2
        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=2),
            transforms.RandomApply([transforms.ColorJitter(hue=.03,
                                                           brightness=.5)],
                                   p=6),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
        ])
        self.convert2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        center = random.sample(range(self.h_mask_size+1,
                                     self.img_size - self.h_mask_size-1), 2)
        img, label = self.dataset.__getitem__(idx)
        img = self.transform(img)
        img = np.array(img)
        masked_area = np.copy(img[center[0] - self.h_mask_size:
                                  center[0] + self.h_mask_size,
                                  center[1] - self.h_mask_size:
                                  center[1] + self.h_mask_size, :])
        assert masked_area.shape == (self.mask_size, self.mask_size, 3)
        img[center[0] - self.h_mask_size: center[0] + self.h_mask_size,
            center[1] - self.h_mask_size: center[1] + self.h_mask_size, :] = 0
        img = self.convert2tensor(img)
        masked_area = self.convert2tensor(masked_area)
        assert (masked_area.shape[1] == masked_area.shape[2]
                and masked_area.shape[1] == self.mask_size)
        return img, masked_area


def create_image_dataset(args, split):
    image_dataset = ImageFolder(os.path.join('./data', args.dataset,
                                             split), transform=None)
    return image_dataset


def get_dataset(args):
    if args.dataset == 'cifar10':
        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True,
                                                     transform=None)

        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True,
                                                    transform=None)
        return cifar10_train, cifar10_test, 32

    elif args.dataset == 'stl10':
        stl10_train = torchvision.datasets.STL10(root='./data',
                                                 split='unlabeled',
                                                 transform=None,
                                                 download=True)

        stl10_test = torchvision.datasets.STL10(root='./data',
                                                split='test',
                                                transform=None,
                                                download=True)
        return stl10_train, stl10_test, 96

    elif os.path.isdir(os.path.join('./data', args.dataset)):
        custom_train = create_image_dataset(args, 'train')
        custom_test = create_image_dataset(args, 'test')
        image_size = custom_train.__getitem__(0)[0].size(0)
        return custom_train, custom_test, image_size
    else:
        print('Not implemented')
        exit(-1)


def create_dataloader(dataset, args):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers)
    return loader


def get_dataloaders(args):
    train, test, image_size = get_dataset(args)
    if args.type in ['SimCLR', 'VICReg', 'SimSiam']:
        if args.type in ['VICReg', 'SiamSiam']:
            assert args.n_views == 2
        train_data = ViewsDataset(train, args)
        valid_data = ViewsDataset(test, args)
    elif args.type == 'InPainting':
        train_data = InPaintingDataset(train, image_size,  args)
        valid_data = InPaintingDataset(test, image_size, args)
    else:
        print('Not Implemented')
        quit(-1)
    trainloader = create_dataloader(train_data, args)
    validloader = create_dataloader(valid_data, args)
    return trainloader, validloader
