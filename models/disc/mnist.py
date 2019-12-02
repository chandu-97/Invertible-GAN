import torch.nn as nn

from models.disc.spectral_loss import SpectralNorm

class MNIST_Linear(nn.Module):
    def __init__(self, img_shape):
        super(MNIST_Linear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        return self.model(img_flat)


class MNIST_Linear_Spectral(nn.Module):
    def __init__(self, img_shape):
        super(MNIST_Linear_Spectral, self).__init__()
        self.model = nn.Sequential(
            SpectralNorm(nn.Linear(int(img_shape), 512)),
            SpectralNorm(nn.LeakyReLU(0.2, inplace=True)),
            SpectralNorm(nn.Linear(512, 256)),
            SpectralNorm(nn.LeakyReLU(0.2, inplace=True)),
            SpectralNorm(nn.Linear(256, 1)),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        return self.model(img_flat)
