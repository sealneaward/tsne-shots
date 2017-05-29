from __future__ import print_function

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from data import DataSet
from vae import VAE
import config as CONFIG

def reconstruction(data_loader, model, n_images=5):
    """
    Plot n images from dataset to show the images in
    original form and in reconstructed form

    Parameters
    ----------
    data_loader: torch.DataSet
        loader of data
    model: pytorch model
        VAE model from checkpoint
    n_images: int
        number of images to display in two rows

    Returns
    -------
    """
    # TODO: https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/404_autoencoder.py

    # get image data for n images
    for batch_idx, (images, _) in enumerate(data_loader):
        data = images
        orig_size = data.size()
        vutils.save_image(data, CONFIG.plots.dir + '/original_images.png',nrow=5)

        # run sample batch through model to get reconstructed images
        data = Variable(data)
        data = data.cuda()
        model.train()
        recon_batch, mu, logvar = model(data)
        recon_batch.data = recon_batch.data.view(n_images,3,28,28)
        recon_size = recon_batch.data.size()
        vutils.save_image(recon_batch.data, CONFIG.plots.dir + '/reconstructed_images.png',nrow=5)

        break

def save_encoding(data_loader, model):
    """
    Load all encoding representations of an image given a model and store the flattened arrays.

    Parameters
    ----------
    data_loader: torch.DataSet
        loader of data
    model: pytorch model
        VAE model from checkpoint

    """
    # get image data for all images
    columns = range(50)
    columns = ["{:02d}".format(x) for x in columns]
    columns.extend(['player_id', 'season_id'])
    images_data = pd.DataFrame(columns=columns)

    for batch_idx, (img, img_name) in enumerate(data_loader):
        img_name = img_name[0].split('.')[0]
        img = Variable(img)
        img = img.cuda()

        # create encoding through reparametrization
        recon_batch, mu, logvar = model(img)
        size = logvar.size()
        logvar = logvar.data.view(-1, size[0] * size[1])

        row = logvar.cpu().numpy().tolist()[0]
        player_id = int(img_name.split('_')[0])
        season_id = img_name.split('_')[1]
        row.append(player_id)
        row.append(season_id)
        row = [row]
        image_data = pd.DataFrame(data=row, columns=columns)
        images_data = images_data.append(image_data)

    images_data.to_csv(CONFIG.data.dir + '/encoded_data.csv', index=False)



if __name__ == '__main__':
    # params for visualizations
    n_images = 10

    transformers = transforms.Compose([
        transforms.ToTensor()
    ])

    img_path = CONFIG.shots.dir
    dataset = DataSet(img_path, transform=transformers)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=n_images, shuffle=True
    )

    model = VAE()
    model.cuda()
    resume_path = CONFIG.model.dir + '/model.checkpoint.tar'
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, checkpoint['epoch']))

        reconstruction(data_loader=loader, model=model, n_images=n_images)
        loader.batch_size = 1
        # save_encoding(data_loader=loader, model=model)

    else:
        print("=> no checkpoint found at '{}'".format(resume_path))
