import torch
import argparse
import scipy.misc as scmisc
import os
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--count', type=int, default=50000)
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--dataset', type=str, default="CIFAR")
	parser.add_argument('--generator_path', type=str, default="model_path")
	parser.add_argument('--output_dir', type=str, default="images")
	parser.add_argument('--is_realnvp', type=int, default=0)
	return parser.parse_args()

def ldim(dataset, is_realnvp):
	if is_realnvp:
		if dataset == "CIFAR":
			return (3, 32, 32)
		elif dataset == "MNIST":
			return (1, 28, 28)
		elif dataset == "CelebA32":
			return (3, 32, 32)
		elif dataset == "CelebA64":
			return (3, 64, 64)
		else:
			raise
	else:
		return (100, 1, 1)

def main(args):
	latent_dim = ldim(args.dataset, args.is_realnvp)
	model = torch.load(args.generator_path)
	print(model)
	z = torch.randn(args.batch_size, *latent_dim)
	image_num = 1
	if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
	for _ in range(args.count//args.batch_size):
		out, _ = model(z, reverse=True)
		out = out.to('cpu').data.numpy()
		out = (out+1.0)/2.0
		out = np.clip(out,0.0,1.0)
		for sample in out:
			out_file = os.path.join(args.output_dir,f"{image_num}.png")
			if sample.shape[0] == 1: sample = sample[0]
			scmisc.toimage(sample, cmin=0.0, cmax=1.0).save(out_file)
			image_num += 1

if __name__ == '__main__':
	args = parse_args()
	main(args)