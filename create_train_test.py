from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from sklearn.model_selection import train_test_split

import config as CONFIG

def create_train_test_folders():
    """
    Check and create directories to put files in
    """
    paths = [CONFIG.train.dir, CONFIG.val.dir, CONFIG.train.img.dir, CONFIG.val.img.dir]
    for path in paths:
        # check if directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        # remove folders if exist
        else:
            os.system('rm -rf  ' + path)
            os.makedirs(path)

def split_data(input_dir, split_size):
    """
    Use train_test_split from sklearn to create lists
    of training data and test data to move to respective folders for dataset creation.

    Parameters
    ----------
    input_dir: str
        CONFIG entry of where to list files from
    split_size: float
        split size for train_test_split

    Returns
    -------
    train_list: str list
        list of file names
    val_list: str list
        list of file names
    """
    files = listdir(input_dir)
    for i, file_path in enumerate(files):
        file_path = input_dir + '/' + file_path
        files[i] = file_path

    train_list, val_list = train_test_split(files, test_size=split_size)
    return train_list, val_list


def move_files(input_list, output_type):
    """
    Move files into proper directory for consumption for training

    Parameters
    ----------
    input_list: string list
        list of files to move
    output_type: string
        one of many different combinations of where to  move the files
        into from the input txt file

    Returns
    -------
    """
    output_path = ''

    if output_type == 'train':
        output_path = CONFIG.train.img.dir
    elif output_type == 'val':
        output_path = CONFIG.val.img.dir

    print('Copying to %s \n' % (output_path))

    for file_path in input_list:
        try:
            img_name = file_path.split('/')[-1]

            # copy to directory
            print('Copying image %s' % (img_name))
            copyfile(file_path, output_path + '/' + img_name)

        except Exception as err:
            print('Error copying line %s' % (file_path))
            print('Error: %s' % (err))


if __name__ == "__main__":
    create_train_test_folders()

    input_dir = CONFIG.shots.dir
    train_list, val_list = split_data(input_dir=input_dir, split_size=0.2)

    move_files(train_list, 'train')
    move_files(val_list, 'val')
