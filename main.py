import argparse
import datetime
import torch
import os

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.discriminator import all_discriminator
from models.generator import all_generator
from utils import data_loader

# GLOBAL VARS
discriminator = None
generator = None


def parse_args():
	parser = argparse.ArgumentParser(description="Invertible GAN(RealNVP and Spectral Norm loss)")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--dataset', type=str, default="CIFAR")
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--checkpoint_dir', type=str, default="")
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--disc_iters', type=int, default=5)
	parser.add_argument('--loss', type=str, default="hinge")
	args = parser.parse_args()
	if args.device == 'cuda':
		args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if args.checkpoint_dir == "":
		args.checkpoint_dir = "checkpoint_" + f"num_epochs_{args.num_epochs}_" + \
							  f"dataset_{args.dataset}_" + f"batch_size_{args.batch_size}_"\
							  "time_" + str(datetime.datetime.now().time())
	return args


def train(args, train_loader, optim_disc, optim_gen, latent_dim):
	global discriminator, generator
	for batch_idx, (data, target) in enumerate(train_loader):
		if data.size()[0] != args.batch_size:
			continue
		data, target = torch.tensor(data).to(args.device), torch.tensor(target).to(args.device)

		# update discriminator
		for _ in range(args.disc_iters):
			z = torch.randn(args.batch_size, *latent_dim, requires_grad=True).to(args.device)
			optim_disc.zero_grad()
			optim_gen.zero_grad()
			gen_data, _ = generator(z, reverse=True)
			if args.loss == 'hinge':
				disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(gen_data)).mean()
			elif args.loss == 'wasserstein':
				disc_loss = -discriminator(data).mean() + discriminator(gen_data).mean()
			else:
				disc_loss = nn.BCEWithLogitsLoss()(discriminator(data),
												   torch.ones(args.batch_size, 1, requires_grad=True).to(args.device)) + \
							nn.BCEWithLogitsLoss()(discriminator(gen_data),
												   torch.zeros(args.batch_size, 1, requires_grad=True).to(args.device))
			disc_loss.backward()
			optim_disc.step()

		z = torch.randn(args.batch_size, *latent_dim, requires_grad=True).to(args.device)
		# update generator
		optim_disc.zero_grad()
		optim_gen.zero_grad()
		gen_data, _ = generator(z, reverse=True)
		if args.loss == 'hinge' or args.loss == 'wasserstein':
			gen_loss = -discriminator(gen_data).mean()
		else:
			gen_loss = nn.BCEWithLogitsLoss()(discriminator(gen_data),
											  torch.ones(args.batch_size, 1, requires_grad=True).to(args.device))
		gen_loss.backward()
		optim_gen.step()

		if batch_idx % 100 == 0:
			print('disc loss', disc_loss.data, 'gen loss', gen_loss.data)

def test(args, fixed_latent, epoch):
	global discriminator, generator
	samples, _ = generator(fixed_latent)
	samples = samples.cpu().data.numpy()[:64]
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(8, 8)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)

	if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)
	output_folder = os.path.join(args.checkpoint_dir, 'out')
	if not os.path.exists(output_folder): os.makedirs(output_folder)
	plt.savefig(os.path.join(output_folder, '/{}.png').format(str(epoch).zfill(3)), bbox_inches='tight')
	plt.close(fig)


def save_models(args, epoch):
	torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, f"discriminator_{epoch}"))
	torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f"generator_{epoch}"))


def main(args):
	global discriminator, generator
	discriminator = all_discriminator(args).to(args.device)
	generator = all_generator(args).to(args.device)
	optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()))
	optim_gen = optim.Adam(generator.parameters())

	latent_dim = (3,32,32)
	fixed_latent = torch.randn(args.batch_size, *latent_dim, requires_grad=True).to(args.device)
	train_loader, _ = data_loader(args)
	for epoch in range(args.num_epochs):
		train(args, train_loader, optim_disc, optim_gen, latent_dim)
		test(args, fixed_latent, epoch)
		save_models(args, epoch)


if __name__=="__main__":
	args = parse_args()
	main(args)