import argparse
import datetime
import torch
import os

from models.discriminator import discriminator
from models.generator import generator

# GLOBAL VARS
discriminator = None
generator = None

def parse_args():
	parser = argparse.ArgumentParser(description="Invertible GAN(RealNVP and Spectral Norm loss)")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--dataset', type=str, default="MNIST")
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--checkpoint_dir', type=str, default="")
	parser.add_argument('--is_cuda', type=bool, default=True)
	parser.add_argument('--is_spectral_loss', type=bool, default=True)
	args = parser.parse_args()
	if args.checkpoint_dir == "":
		args.checkpoint_dir = "checkpoint_" + f"num_epochs_{args.num_epochs}_" + \
							  f"dataset_{args.dataset}_" + f"batch_size_{args.batch_size}_"\
							  "time_" + str(datetime.datetime.now().time())
	return args


def train(args):
	global discriminator, generator


def test(args):
	global discriminator, generator


def save_models(args, epoch):
	torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, f"discriminator_{epoch}"))
	torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f"generator_{epoch}"))


def main(args):
	global discriminator, generator
	discriminator = discriminator(args)
	generator = generator(args)
	for epoch in range(args.num_epochs):
		train(args)
		test(args)
		save_models(args, epoch, discriminator, generator)


if __name__=="__main__":
	args = parse_args()
	main(args)