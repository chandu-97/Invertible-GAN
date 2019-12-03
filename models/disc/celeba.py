import torch.nn as nn
from models.disc.spectral_loss import SpectralNorm


class CelebA_64_Disc(nn.Module):
    def __init__(self, leak, use_spectral):
        super(CelebA_64_Disc, self).__init__()
        if use_spectral:
            self.features = nn.Sequential(
                SpectralNorm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(leak),
                SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.LeakyReLU(leak),
                SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.LeakyReLU(leak),
                SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.LeakyReLU(leak),
                SpectralNorm(nn.Conv2d(512, 1, 4, 1))
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(leak),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(leak),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(leak),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(leak),
                nn.Conv2d(512, 1, 4, 1)
            )

    def forward(self, x):
        return self.features(x).view(-1)


class CelebA_32_Disc(nn.Module):
    def __init__(self, leak, use_spectral):
        super(CelebA_32_Disc, self).__init__()
        if use_spectral:
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
                SpectralNorm(nn.Linear(128, 1)),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.LeakyReLU(leak),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.LeakyReLU(leak),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.LeakyReLU(leak),
                nn.MaxPool2d(2, 2)
            )
            self.fc = nn.Sequential(
                nn.Linear(4*4*256, 128),
                nn.LeakyReLU(leak),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
            )


    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 4*4*256)
        x = self.fc(x)
        return x