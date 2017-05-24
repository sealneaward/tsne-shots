from __future__ import print_function

import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from data import DataSet
from vae import VAE
import config as CONFIG

# Parameters
parser = argparse.ArgumentParser(description='VAE train')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args, unknown = parser.parse_known_args()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Setup
model = VAE()
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if args.resume:
    resume_path = CONFIG.model.dir + '/model.checkpoint.tar'
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def save_checkpoint(state, is_best, filename):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    transformers = transforms.Compose([
        transforms.ToTensor()
    ])
    model_path = CONFIG.model.dir
    train_path = CONFIG.train.img.dir
    test_path = CONFIG.val.img.dir
    train_dataset = DataSet(train_path, transform=transformers)
    test_dataset = DataSet(test_path, transform=transformers)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
        **kwargs
    )

    # train and test model
    best_loss = 50000
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test(epoch)

        # remember best loss and save checkpoint
        is_best = best_loss > test_loss
        best_loss = min(best_loss, test_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, filename=CONFIG.model.dir + '/model.checkpoint.tar')
