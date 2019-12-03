import torch.nn as nn

from models.disc.spectral_loss import SpectralNorm

class MNIST_Linear(nn.Module):
    def __init__(self, img_shape, leak, use_spectral):
        super(MNIST_Linear, self).__init__()
        if use_spectral:
            self.model = nn.Sequential(
                SpectralNorm(nn.Linear(int(img_shape), 512)),
                nn.LeakyReLU(leak),
                SpectralNorm(nn.Linear(512, 256)),
                nn.LeakyReLU(leak),
                SpectralNorm(nn.Linear(256, 1)),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(int(img_shape), 512),
                nn.LeakyReLU(leak),
                nn.Linear(512, 256),
                nn.LeakyReLU(leak),
                nn.Linear(256, 1),
            )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        return self.model(img_flat)

