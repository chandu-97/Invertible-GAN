import argparse
import datetime
import torch
import os

from models.real_nvp import RealNVP

def parse_args():
	parser = argparse.ArgumentParser(description="Invertible GAN(RealNVP and Spectral Norm loss)")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--dataset', type=str, default="MNIST")
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--checkpoint_dir', type=str, default="")
	parser.add_argument('--is_cuda', type=bool, default=True)
	args = parser.parse_args()
	if args.checkpoint_dir == "":
		args.checkpoint_dir = "checkpoint_" + f"num_epochs_{args.num_epochs}_" + \
							  f"dataset_{args.dataset}_" + f"batch_size_{args.batch_size}_"\
							  "time_" + str(datetime.datetime.now().time())
	return args


def train():
	pass


def test():
	pass


def save_models(args, epoch, discriminator, generator):
	torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, f"discriminator_{epoch}"))
	torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f"generator_{epoch}"))


def main(args):
	discriminator = None
	generator = None
	for epoch in range(args.num_epochs):
		train(args)
		test(args)
		save_models(args, epoch, discriminator, generator)


if __name__=="__main__":
	args = parse_args()
	main(args)