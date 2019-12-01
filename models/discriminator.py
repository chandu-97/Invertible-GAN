from discriminator.mnist import MNIST_Disc_Linear
from discriminator.cifar import CIFAR_Disc

def discriminator(args):
	if args.dataset=="MNIST":
		img_shape = 32*32*3
		return MNIST_Disc_Linear(img_shape)
	elif args.dataset=="CIFAR":
		return CIFAR_Disc()