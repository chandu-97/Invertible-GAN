from models.gen.real_nvp import RealNVP

def all_generator(args):
	if args.dataset == "CIFAR":
		return RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
	else:
		raise