from __future__ import print_function

import cv2
import h5py
import os
from os import listdir
from os import path
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.utils.data as data

import config as CONFIG

class DataSet(data.Dataset):
    def __init__(self, file_path, transform=None):
        """
        Follow conventional pytorch dataset format for h5 files,

        Parameters
        ----------
        file_path: str
            location of hdf5 file of shots
        transform: torchvision.transforms
            torchvision ndarray to tensor transformations
        """
        self.data = h5py.File(file_path, 'r')
        self.imgs = self.data['imgs']
        self.transform = transform
        self._length = len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]

        if self.transform is not None:
            img = self.transform(img)

        return (img)

    def __len__(self):
        return self._length


def get_data(img_size=(256,256), test_split=0.2):
    """
    Loads all shot images into numpy arrays

    Parameters
    ----------
    img_size: tuple
        img width and height sizes
    test_split: float
        indicator of test and train size

    Returns
    -------
    train_test: tuple
        ndarrays of train and test images
    """

    images = listdir(CONFIG.shots.dir)
    shots = []

    for i in tqdm(range(len(images))):
        image = images[i]
        img_path = path.join(CONFIG.shots.dir, image)
        shot = cv2.imread(img_path,3)
        shot = cv2.resize(shot,img_size)
        shot_sm = cv2.pyrMeanShiftFiltering(shot, 20, 45, 3)
        shots.append(shot_sm)

    X = np.array(shots)
    train_x, test_x = train_test_split(X, test_size=test_split)

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


if __name__ == '__main__':
    create_hdf5()
