from __future__ import print_function


import os
from os import listdir
from os import path
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm

import torch.utils.data as data
from torchvision.datasets import ImageFolder

import config as CONFIG

class DataSet(data.Dataset):
    def __init__(self, file_path, transform=None, n_channels=3):
        """
        Follow conventional pytorch dataset format for images

        Parameters
        ----------
        file_path: str
            location of hdf5 file of shots
        transform: torchvision.transforms
            torchvision ndarray to tensor transformations

        Returns
        -------
        DataSet
        """
        self.file_path = file_path
        self.imgs = listdir(self.file_path)
        self.transform = transform
        self.n_channels = n_channels

    def __getitem__(self, i):
        img_name = self.imgs[i]
        img_path = path.join(self.file_path, img_name)
        img = Image.open(img_path)
        img = img.resize((28,28), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
            size = img.size()

        return img, img_name

    def __len__(self):
        return len(self.imgs)


def get_data(imgs_path, img_size=(3,256,256), test_split=0.2, n_channels=3):
    """
    Loads all shot images into numpy arrays

    Parameters
    ----------
    imgs_path: str
        location of folder to read images from
    img_size: tuple
        img width and height sizes
    test_split: float
        indicator of test and train size

    Returns
    -------
    train_test: tuple
        ndarrays of train and test images
    """

    images = listdir(imgs_path)
    shots = []

    for i in tqdm(range(len(images))):
        image = images[i]
        img_path = path.join(imgs_path, image)
        shot = cv2.imread(img_path,n_channels)
        shot = cv2.resize(shot,img_size)
        shots.append(shot)

    X = np.array(shots)

    return train_x, test_x

def create_hdf5(width=256, height=256, channels=3):
    """
    Create hdf5 files to store np array information of images

    Parameters
    ----------
    width: int
        img width
    height: int
        img height
    channels: int
        num of channels in image (RGB)

    Returns
    -------
    """
    # get image array data
    train_x, test_x = get_data(img_size=(256,256))
    train_len = train_x.shape[0]
    test_len = test_x.shape[0]

    # initialize hdf5 file and datasets
    train_path = CONFIG.hdf5.dir + '/' + 'shots_train.h5py'
    if os.path.exists(train_path):
        os.remove(train_path)

    h5_file = h5py.File(train_path)
    imgs = h5_file.create_dataset('imgs', (train_len, width, height, channels))

    for i in tqdm(range(0,train_len)):
        imgs[i] = train_x[i]

    h5_file.close()

    test_path = CONFIG.hdf5.dir + '/' + 'shots_test.h5py'
    if os.path.exists(test_path):
        os.remove(test_path)

    h5_file = h5py.File(test_path)
    imgs = h5_file.create_dataset('imgs', (test_len, width, height, channels))

    for i in tqdm(range(0,test_len)):
        imgs[i] = test_x[i]

    h5_file.close()
