from models.disc.mnist import MNIST_Linear
from models.disc.cifar import CIFAR_Disc

def all_discriminator(args):
	if args.dataset=="MNIST":
		img_shape = 32*32*3
		return MNIST_Linear(img_shape)
	elif args.dataset=="CIFAR":
		return CIFAR_Disc()