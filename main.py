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
scheduler_d = None
scheduler_g = None

def parse_args():
	parser = argparse.ArgumentParser(description="Invertible GAN(RealNVP and Spectral Norm loss)")
	parser.add_argument('--b1', type=float, default=0.0)
	parser.add_argument('--b2', type=float, default=0.9)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--dataset', type=str, default="CIFAR")
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--disc_iters', type=int, default=1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--checkpoint_dir', type=str, default="")
	parser.add_argument('--is_realnvp', type=int, default=0)
	parser.add_argument('--is_spectral', type=int, default=0)
	parser.add_argument('--leak', type=float, default=0.1)
	parser.add_argument('--loss', type=str, default="hinge")
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--realnvp_num_scales', type=int, default=2)
	parser.add_argument('--realnvp_num_mid_channels', type=int, default=64)
	parser.add_argument('--realnvp_num_num_blocks', type=int, default=8)
	args = parser.parse_args()
	if args.device == 'cuda':
		args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if args.checkpoint_dir == "":
		args.checkpoint_dir = f"{args.dataset}_" + f"batch_size_{args.batch_size}_" \
							  "checkpoint_" + f"num_epochs_{args.num_epochs}_" + f"loss_{args.loss}_"\
							  "time_" + str(datetime.datetime.now().time())
		if args.is_spectral: args.checkpoint_dir = "Spectral_" + args.checkpoint_dir
		if args.is_realnvp:
			args.checkpoint_dir = f"RealNVP_{args.realnvp_num_scales}_{args.realnvp_num_mid_channels}" \
								  f"_{args.realnvp_num_num_blocks}" + args.checkpoint_dir
	return args


def sample(args, fixed_latent, epoch, batch):
	global discriminator, generator, scheduler_g, scheduler_d
	samples, _ = generator(fixed_latent, reverse=True)
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
		if sample.shape[0]!=1:
			plt.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)
		else:
			plt.imshow(sample[0] * 0.5 + 0.5)

	if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)
	output_folder = os.path.join(args.checkpoint_dir, 'out')
	if not os.path.exists(output_folder): os.makedirs(output_folder)
	file_name = os.path.join(output_folder, '{}_{}.png'.format(str(epoch).zfill(3), str(batch).zfill(3)))
	print(file_name)
	plt.savefig(file_name, bbox_inches='tight')
	plt.close(fig)


def train(args, train_loader, optim_disc, optim_gen, latent_dim, epoch, fixed_latent):
	global discriminator, generator, scheduler_g, scheduler_d
	cumulative_gen_loss = 0.0
	cumulative_disc_loss = 0.0
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
			elif args.loss == 'bce':
				disc_loss = nn.BCEWithLogitsLoss()(discriminator(data),
													torch.ones(args.batch_size, 1, requires_grad=True).to(args.device)) + \
							nn.BCEWithLogitsLoss()(discriminator(gen_data),
													torch.zeros(args.batch_size, 1, requires_grad=True).to(args.device))
			else:
				raise
			disc_loss.backward()
			optim_disc.step()

		z = torch.randn(args.batch_size, *latent_dim, requires_grad=True).to(args.device)
		# update generator
		optim_disc.zero_grad()
		optim_gen.zero_grad()
		gen_data, _ = generator(z, reverse=True)
		if args.loss == 'hinge' or args.loss == 'wasserstein':
			gen_loss = -discriminator(gen_data).mean()
		elif args.loss == 'bce':
			gen_loss = nn.BCEWithLogitsLoss()(discriminator(gen_data),
											  torch.ones(args.batch_size, 1, requires_grad=True).to(args.device))
		else:
			raise
		gen_loss.backward()
		optim_gen.step()

		cumulative_disc_loss += disc_loss.item()
		cumulative_gen_loss += gen_loss.item()
		if batch_idx%20 == 0:
			sample(args, fixed_latent, epoch, batch_idx)
			print("[Epoch:{}] ".format(epoch) + "[Batch : {}/{}]".format(batch_idx+1, len(train_loader)) + \
				  'Discriminator loss: ', disc_loss.item(), 'Generator loss: ', gen_loss.item())
			print("Cumulative Generator Loss : {}, Cumulative Discriminator Loss : {}".format(cumulative_gen_loss/(batch_idx+1),
																							  cumulative_disc_loss/(batch_idx+1)))
	scheduler_d.step()
	scheduler_g.step()
	return cumulative_gen_loss/len(train_loader), cumulative_gen_loss/len(train_loader)

def test(args, fixed_latent, epoch):
	global discriminator, generator, scheduler_g, scheduler_d
	pass

def save_models(args, epoch):
	torch.save(discriminator, os.path.join(args.checkpoint_dir, f"discriminator_{epoch}.model"))
	torch.save(generator, os.path.join(args.checkpoint_dir, f"generator_{epoch}.model"))


def main(args):
	global discriminator, generator, scheduler_g, scheduler_d
	print(args)
	discriminator = all_discriminator(args).to(args.device)
	generator = all_generator(args).to(args.device)
	optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), 
							lr=args.lr,
							betas=(args.b1, args.b2))
	optim_gen = optim.Adam(generator.parameters(),
							lr=args.lr,
							betas=(args.b1, args.b2))
	scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
	scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

	if args.is_realnvp:
		if args.dataset == "CIFAR":
			latent_dim = (3,32,32)
		elif args.dataset == "MNIST":
			latent_dim = (1,28,28)
		elif args.dataset == "CelebA32":
			latent_dim = (3,32,32)
		elif args.dataset == "CelebA64":
			latent_dim = (3,64,64)
		else:
			raise
	else:
		latent_dim = (100,1,1)
	fixed_latent = torch.randn(args.batch_size, *latent_dim, requires_grad=True).to(args.device)
	train_loader, _ = data_loader(args)
	for epoch in range(1, args.num_epochs+1):
		train_gen, train_dis = train(args, train_loader, optim_disc, optim_gen, latent_dim, epoch, fixed_latent)
		print('*'*70)
		print("Epoch : {}, Generator Loss : {}, Discriminator Loss : {}".format(epoch,
																				train_gen,
																				train_dis))
		test(args, fixed_latent, epoch)
		save_models(args, epoch)
		print("Saved models after Epoch : {}".format(epoch))

if __name__=="__main__":
	args = parse_args()
	main(args)
