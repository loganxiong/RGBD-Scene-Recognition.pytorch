import pathlib
import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms
from sampler import ImbalancedDatasetSampler


import augmentations
from PIL import Image, ImageEnhance

import transforms

import cv2
import random

import torchvision.transforms.functional as F

RGBD = True

from nyu import NYU

class RGBDCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
          
    def __call__(self, img, depth):  
        for t in self.transforms:
            img, depth = t(img, depth)
        return img, depth

    def __repr__(self): 
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n' 
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RGBDRandomCrop(torchvision.transforms.RandomCrop):
    def __call__(self, img, depth):
        if not (img.size == depth.size):
            print(img.size)
            print(depth.size)
        assert img.size == depth.size
        if self.padding > 0:
            img = F.pad(img, self.padding)
            depth = F.pad(depth, self.padding)
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            depth = F.pad(depth, (int((1 + self.size[1] - depth.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            depth = F.pad(depth, (0, int((1 + self.size[0] - depth.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)
        return F.crop(img, i, j, h, w), F.crop(depth, i, j, h, w)

class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        pass
                
class RGBDRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, depth):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(rgb), F.hflip(depth)
        return rgb, depth

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RGBDNormalize(object):
    def __init__(self, mean, std, mean_d, std_d):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.mean_d = np.array(mean_d)
        self.std_d = np.array(std_d)

    def __call__(self, image, depth):
        image = np.asarray(image).astype(np.float32) / 255.
        depth = np.asarray(depth).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        
        return image, depth

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RGBDToTensor(object):
    def __call__(self, data, depth):
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data]), tuple([self._to_tensor(image) for image in depth])
        else:
            return self._to_tensor(data), self._to_tensor(depth)

    def _to_tensor(self, data):
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))

class Dataset:
    def __init__(self, config):
        self.config = config
        #dataset_rootdir = pathlib.Path('~/.torchvision/datasets').expanduser()
        dataset_rootdir = pathlib.Path('/backup')
        self.dataset_dir = dataset_rootdir / config['dataset']

        self._train_transforms = []
        if RGBD:
        	self.train_transform = self._get_train_transformRGBD()
        	self.test_transform = self._get_test_transformRGBD()
        else:
        	self.train_transform = self._get_train_transform()
        	self.test_transform = self._get_test_transform()

    def get_datasets(self):
        train_dataset = NYU(
            self.dataset_dir,
            train=True,
            transform=self.train_transform,
            download=True)
        test_dataset = NYU(
            self.dataset_dir,
            train=False,
            transform=self.test_transform,
            download=True)
        return train_dataset, test_dataset
    

    def _add_random_crop(self):
        transform = torchvision.transforms.RandomCrop(
            self.size, padding=self.config['random_crop_padding'])
        self._train_transforms.append(transform)
    
    def _add_random_cropRGBD(self):
        transform = RGBDRandomCrop(
            self.size, padding=self.config['random_crop_padding'])
        self._train_transforms.append(transform)

    def _add_horizontal_flip(self):
        self._train_transforms.append(
            torchvision.transforms.RandomHorizontalFlip())
    
    def _add_horizontal_flipRGBD(self):
        self._train_transforms.append(
            RGBDRandomHorizontalFlip())

    def _add_normalization(self):
        self._train_transforms.append(
            transforms.Normalize(self.mean, self.std))
            
    def _add_normalizationRGBD(self):
        self._train_transforms.append(
            RGBDNormalize(self.mean, self.std, self.mean_d, self.std_d))

    def _add_to_tensor(self):
        self._train_transforms.append(transforms.ToTensor())
        
    def _add_to_tensorRGBD(self):
        self._train_transforms.append(RGBDToTensor())

    def _add_random_erasing(self):
        transform = augmentations.random_erasing.RandomErasing(
            self.config['random_erasing_prob'],
            self.config['random_erasing_area_ratio_range'],
            self.config['random_erasing_min_aspect_ratio'],
            self.config['random_erasing_max_attempt'])
        self._train_transforms.append(transform)
        
    def _add_random_erasingRGBD(self):
        transform = augmentations.random_erasing.RGBDRandomErasing(
            self.config['random_erasing_prob'],
            self.config['random_erasing_area_ratio_range'],
            self.config['random_erasing_min_aspect_ratio'],
            self.config['random_erasing_max_attempt'])
        self._train_transforms.append(transform)

    def _add_cutout(self):
        transform = augmentations.cutout.Cutout(self.config['cutout_size'],
                                                self.config['cutout_prob'],
                                                self.config['cutout_inside'])
        self._train_transforms.append(transform)
    
    def _add_cutoutRGBD(self):
        transform = augmentations.cutout.RGBDCutout(self.config['cutout_size'],
                                                self.config['cutout_prob'],
                                                self.config['cutout_inside'])
        self._train_transforms.append(transform)

    def _add_dual_cutout(self):
        transform = augmentations.cutout.DualCutout(
            self.config['cutout_size'], self.config['cutout_prob'],
            self.config['cutout_inside'])
        self._train_transforms.append(transform)

    def _get_train_transform(self):
        if self.config['use_random_crop']:
            self._add_random_crop()
        if self.config['use_horizontal_flip']:
            self._add_horizontal_flip()
        self._add_normalization()
        if self.config['use_random_erasing']:
            self._add_random_erasing()
        if self.config['use_cutout']:
            self._add_cutout()
        elif self.config['use_dual_cutout']:
            self._add_dual_cutout()
        self._add_to_tensor()
        return torchvision.transforms.Compose(self._train_transforms)
        
    def _get_train_transformRGBD(self):
        if self.config['use_random_crop']:
            self._add_random_cropRGBD()
        if self.config['use_horizontal_flip']:
            self._add_horizontal_flipRGBD()
        self._add_normalizationRGBD()
        if self.config['use_random_erasing']:
            self._add_random_erasingRGBD()
        if self.config['use_cutout']:
            self._add_cutoutRGBD()
        self._add_to_tensorRGBD()
        return RGBDCompose(self._train_transforms)

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            transforms.Normalize(self.mean, self.std),
            transforms.ToTensor(),
        ])
        return transform
        
    def _get_test_transformRGBD(self):
        transform = RGBDCompose([
            RGBDNormalize(self.mean, self.std, self.mean_d, self.std_d),
            RGBDToTensor(),
        ])
        return transform


class CIFAR(Dataset):
    def __init__(self, config):
        self.size = 32
        if config['dataset'] == 'CIFAR10':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2470, 0.2435, 0.2616])
        elif config['dataset'] == 'CIFAR100':
            self.mean = np.array([0.5071, 0.4865, 0.4409])
            self.std = np.array([0.2673, 0.2564, 0.2762])            
        super(CIFAR, self).__init__(config)


class MNIST(Dataset):
    def __init__(self, config):
        self.size = 28
        if config['dataset'] == 'MNIST':
            self.mean = np.array([0.1307])
            self.std = np.array([0.3081])
        elif config['dataset'] == 'FashionMNIST':
            self.mean = np.array([0.2860])
            self.std = np.array([0.3530])
        elif config['dataset'] == 'KMNIST':
            self.mean = np.array([0.1904])
            self.std = np.array([0.3475])
        super(MNIST, self).__init__(config)

class NYUv2(Dataset):
    def __init__(self, config):
        self.size = (224,224)
        self.mean = np.array([0.4487, 0.3479, 0.3301])
        self.std = np.array([0.2474, 0.2269, 0.2211])
        self.mean_hha = np.array([0.5241,0.3707,0.4655])
        self.std_hha = np.array([0.2272,0.2465,0.1844])
        self.mean_d = np.array([0.4281])
        self.std_d = np.array([0.2720])
        super(NYUv2, self).__init__(config)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_loader(config):
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    use_gpu = config['use_gpu']

    dataset_name = config['dataset']
    assert dataset_name in [
        'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST', 'NYU'
    ]

    if dataset_name in ['CIFAR10', 'CIFAR100']:
        dataset = CIFAR(config)
        train_dataset, test_dataset = dataset.get_datasets()
    elif dataset_name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset = MNIST(config)
        train_dataset, test_dataset = dataset.get_datasets()
    else:
        dataset = NYUv2(config)
        train_dataset, test_dataset = dataset.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        #shuffle=True,
        sampler=ImbalancedDatasetSampler(train_dataset),
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
