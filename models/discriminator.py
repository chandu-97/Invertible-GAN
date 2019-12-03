from models.disc.mnist import MNIST_Linear
from models.disc.cifar import CIFAR_Disc
from models.disc.celeba import *

def all_discriminator(args):
		if args.dataset == "MNIST":
			img_shape = 28*28*1
			return MNIST_Linear(img_shape, args.leak, args.is_spectral)
		elif args.dataset == "CIFAR":
			return CIFAR_Disc(args.leak, args.is_spectral)
		elif args.dataset == "CelebA32":
			return CelebA_32_Disc(args.leak, args.is_spectral)
		elif args.dataset == "CelebA64":
			return CelebA_64_Disc(args.leak, args.is_spectral)
		else:
			raise
