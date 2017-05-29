import torch
import torch.nn as nn
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(2352, 1000)
        self.fc21 = nn.Linear(1000, 50)
        self.fc22 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 1000)
        self.fc4 = nn.Linear(1000, 2352)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        size = x.size()
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_() # cuda
        # eps = torch.FloatTensor(std.size()).normal_() # no cuda
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        size = x.size()
        x = x.view(-1, 2352)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
