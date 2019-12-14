import torch
import torch.nn as nn

from models.gen.real_nvp import RealNVP
from models.gen.mnist import MNIST_Gen
from models.gen.cifar import CIFAR_Gen
from models.gen.celeba import CelebA_32_Gen, CelebA_64_Gen


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
