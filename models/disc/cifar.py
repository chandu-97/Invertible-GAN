import torch.nn as nn
from spectral_loss import SpectralNorm

leak = 0.1

class CIFAR_Disc(nn.Module):
    def __init__(self):
        super(CIFAR_Disc, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4*4*256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 4*4*256)
        x = self.fc(x)
        return x


class CIFAR_Disc_Spectral(nn.Module):
    def __init__(self):
        super(CIFAR_Disc_Spectral, self).__init__()
        self.cnn = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(leak),
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(leak),
            SpectralNorm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            nn.LeakyReLU(leak),
        )
        self.fc = nn.Sequential(
            SpectralNorm(nn.Linear(4*4*256, 128)),
            nn.LeakyReLU(leak),
            SpectralNorm(nn.Linear(128, 1))
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 4*4*256)
        x = self.fc(x)
        return x