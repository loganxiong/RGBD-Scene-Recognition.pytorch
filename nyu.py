from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import codecs
import pathlib
import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

#import augmentations
from PIL import Image, ImageEnhance

import transforms

import cv2

NYUscenetypes = {'bedroom':0,'kitchen':1,'living_room':2,'bathroom':3,'dining_room':4,'office':5,'home_office':6,'classroom':7,'bookstore':8,'others':9};

RGBD = True

class NYU(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = '/data/shadow/NYUv2/'
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.input_shape = 256 

        if self.train:
            self.train_data, self.train_hha, self.train_labels = self.get_data(train)
            print('Loading trainset........')
        else:
            self.test_data, self.test_hha, self.test_labels = self.get_data(train)
            print('Loading testset.........')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if RGBD:
                img, depth, target = self.train_data[index], self.train_hha[index], self.train_labels[index]
            else:
                img, _, target = self.train_data[index], self.train_hha[index], self.train_labels[index]
        else:
            if RGBD:
                img, depth, target = self.test_data[index], self.test_hha[index], self.test_labels[index]
            else:
                img, _, target = self.test_data[index], self.test_hha[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        if RGBD:
            depth = Image.fromarray(np.uint8(depth)).convert('RGB')
        #img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            if RGBD:
                #print(self.transform)
                img, depth = self.transform(img, depth)
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if RGBD:
            return img, depth, target
        else:
            return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
            
    def get_data(self, train):
        if self.train:
            scene_train_file =  self.root + 'nyu_scene_split_train.txt'
            with open(scene_train_file,'r') as f:
                train_num = len(f.readlines())
            rgb_train = np.zeros([train_num,self.input_shape,self.input_shape,3])
            label_train = np.zeros([train_num]).astype(np.long)
            hha_train = np.zeros([train_num,self.input_shape,self.input_shape,3])
            self.read_lines(scene_train_file, rgb_train, hha_train, label_train, 'train')
            return rgb_train, hha_train, label_train 
        else:
            scene_test_file = self.root + 'nyu_scene_split_val.txt'
            with open(scene_test_file,'r') as f:
                test_num = len(f.readlines())
            rgb_test = np.zeros([test_num,self.input_shape,self.input_shape,3])
            hha_test = np.zeros([test_num,self.input_shape,self.input_shape,3])
            label_test = np.zeros([test_num]).astype(np.long)
            if 'test' in scene_test_file:
                self.read_lines(scene_test_file, rgb_test, hha_test, label_test, 'test')
            else:
                self.read_lines(scene_test_file, rgb_test, hha_test, label_test, 'val')
            return rgb_test, hha_test, label_test
	  
    def read_lines(self, file_name, save_array, save_hha, save_label, mode):
        with open(file_name) as train_f:
            train_lines = train_f.readlines()
            num_item = save_array.shape[0]
            for i in (range(num_item)):
                train_name, class_name = train_lines[i].split(' ')
                train_img_name = self.root + mode +'/images/' + train_name + '.png'
                train_hha_name = self.root + mode + '/hhas/' + train_name + '.png'
                class_label = NYUscenetypes[class_name[:-1]]
                train_img = cv2.imread(train_img_name)
                train_hha = cv2.imread(train_hha_name)
                train_img = cv2.resize(train_img,(self.input_shape,self.input_shape))
                train_hha = cv2.resize(train_hha,(self.input_shape,self.input_shape))
                save_array[i,...] = train_img
                save_hha[i,...] = train_hha
                save_label[i] = int(class_label)
