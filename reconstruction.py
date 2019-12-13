import torch
import argparse
import scipy.misc as scmisc
import os
import numpy as np
from utils import data_loader

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--count', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--dataset', type=str, default="MNIST")
	parser.add_argument('--generator_path', type=str, default="model_path")
	parser.add_argument('--output_dir', type=str, default="images")
	parser.add_argument('--is_realnvp', type=int, default=1)
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
	model.eval()
	print(model)
	real_image_num = 1
	reconstruct_image_num = 1
	_, test_loader = data_loader(args)
	if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
	for batch_idx, (data, target) in enumerate(train_loader):
		if data.size()[0] != args.batch_size:
			continue
		data, target = torch.tensor(data).to(args.device), torch.tensor(target).to(args.device)
		z, _ = model(data,reverse=False)
		z=z.to(args.device)
		out, _ = model(data, reverse=True)
		out = out.to('cpu').data.numpy()
		out = (out+1.0)/2.0
		out = np.clip(out,0.0,1.0)
		for sample in out:
			out_file = os.path.join(args.output_dir,f"{reconstruct_image_num}.png")
			if sample.shape[0] == 1: sample = sample[0]
			scmisc.toimage(sample, cmin=0.0, cmax=1.0).save(out_file)
			reconstruct_image_num += 1

		data = data.to('cpu').data.numpy()
		data = (data+1.0)/2.0
		data = np.clip(data,0.0,1.0)
		for sample in data:
			out_file = os.path.join(args.output_dir,f"{real_image_num}.png")
			if sample.shape[0] == 1: sample = sample[0]
			scmisc.toimage(sample, cmin=0.0, cmax=1.0).save(out_file)
			real_image_num += 1

if __name__ == '__main__':
	args = parse_args()
	main(args)
