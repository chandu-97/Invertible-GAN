import torch
import torch.nn as nn

from models.gen.real_nvp import RealNVP
from models.gen.mnist import MNIST_Gen
from models.gen.cifar import CIFAR_Gen
from models.gen.celeba import CelebA_32_Gen, CelebA_64_Gen

# from models.iresnet.InvertibleResnet import InvertibleResnetConv
from models.irevnet.iRevNet import iRevNet

class RealNVP_GAN(nn.Module):
	def __init__(self, num_scales, in_channels, mid_channels, num_blocks):
		super(RealNVP_GAN, self).__init__()
		self.real_nvp = RealNVP(num_scales, in_channels, mid_channels, num_blocks)

	def forward(self, x, reverse=False):
		if reverse:
			x = (x + 1.0)/2
			x = torch.clamp(x, min=0.0, max=1.0)
			return self.real_nvp(x, reverse)
		else:
			output, _ = self.real_nvp(x, reverse)
			return 2*output - 1.0


def all_generator(args):
	assert not(args.is_realnvp ^ args.is_realnvp)
	if args.is_realnvp:
		if (args.dataset == "CIFAR") or (args.dataset == "CelebA32") or (args.dataset == "CelebA64"):
			return RealNVP_GAN(num_scales=args.realnvp_num_scales,
							   in_channels=3,
							   mid_channels=args.realnvp_num_mid_channels,
							   num_blocks=args.realnvp_num_num_blocks)
		elif args.dataset == "MNIST":
			return RealNVP_GAN(num_scales=args.realnvp_num_scales,
							   in_channels=1,
							   mid_channels=args.realnvp_num_mid_channels,
							   num_blocks=args.realnvp_num_num_blocks)
		else:
			raise
	elif args.is_irevnet:
		if args.dataset == "MNIST":
			nClasses = 10
			in_shape = [4,8,8]
			bottleneck_mult = 4
			init_ds = 0
			nBlocks = [18, 18, 18]
			nStrides = [1, 2, 2]
			nChannels = [8, 32, 128]
			return iRevNet(nBlocks=nBlocks, nStrides=nStrides,
							nChannels=nChannels, nClasses=nClasses,
							init_ds=init_ds, dropout_rate=0.0, affineBN=True,
							in_shape=in_shape, mult=bottleneck_mult)
		elif args.dataset == "CIFAR":
			nClasses = 10
			in_shape = [8,8,8]
			bottleneck_mult = 4
			init_ds = 0
			nBlocks = [18, 18, 18]
			nStrides = [1, 2, 2]
			nChannels = [8*3, 32*3, 128*3]
			return iRevNet(nBlocks=nBlocks, nStrides=nStrides,
							nChannels=nChannels, nClasses=nClasses,
							init_ds=init_ds, dropout_rate=0.0, affineBN=True,
							in_shape=in_shape, mult=bottleneck_mult)
		elif args.dataset == "CelebA32":
			nClasses = 10
			in_shape = [8,8,8]
			bottleneck_mult = 4
			init_ds = 0
			nBlocks = [18, 18, 18]
			nStrides = [1, 2, 2]
			nChannels = [8*3, 32*3, 128*3]
			return iRevNet(nBlocks=nBlocks, nStrides=nStrides,
							nChannels=nChannels, nClasses=nClasses,
							init_ds=init_ds, dropout_rate=0.0, affineBN=True,
							in_shape=in_shape, mult=bottleneck_mult)
		elif args.dataset == "CelebA64":
			nClasses = 10
			in_shape = [8,16,16]
			bottleneck_mult = 4
			init_ds = 0
			nBlocks = [18, 18, 18]
			nStrides = [1, 2, 2]
			nChannels = [8*3, 32*3, 128*3]
			return iRevNet(nBlocks=nBlocks, nStrides=nStrides,
							nChannels=nChannels, nClasses=nClasses,
							init_ds=init_ds, dropout_rate=0.0, affineBN=True,
							in_shape=in_shape, mult=bottleneck_mult)
		else:
			raise
	else:
		if args.dataset == "MNIST":
			return MNIST_Gen()
		elif args.dataset == "CIFAR":
			return CIFAR_Gen()
		elif args.dataset == "CelebA32":
			return CelebA_32_Gen()
		elif args.dataset == "CelebA64":
			return CelebA_64_Gen()
		else:
			raise
