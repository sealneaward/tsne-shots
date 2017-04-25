from __future__ import print_function

import cv2
import config as CONFIG
from os import listdir
from os import path
from sklearn.model_selection import train_test_split
import numpy as np

def get_data(img_size=(28,28), test_split=0.2):
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

    for image in images:
        img_path = path.join(CONFIG.shots.dir, image)
        shot = cv2.imread(img_path,0)
        shot = cv2.resize(shot,img_size)
        shots.append(shot)

    X = np.array(shots)
    train_x, test_x = train_test_split(X, test_size=test_split)

    return train_x, test_x


if __name__ == '__main__':
    train_x, test_x = get_data()

    print('Train Size: %s' % (str(train_x.shape)))
    print('Test Size: %s' % (str(test_x.shape)))
