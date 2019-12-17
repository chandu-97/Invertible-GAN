import argparse
import torch
import os
import torchvision
import torch.utils.data as data

from torchvision import datasets
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.utils import save_image


def parseargs():
    parser = argparse.ArgumentParser(description="Invertible GAN(RealNVP and Spectral Norm loss)")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="celeba")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--model",type=str,default="realnvp")
    parser.add_argument("--num_samples",type=int,default=20,help="number of samples to output")
    parser.add_argument("--a",type=int,default=0,help="attribute")
    args = parser.parse_args()
    return args



def reconstruct(args,generator,dataloader):
    for i, (imgs, labels) in enumerate(dataloader):	
        n=imgs.shape[0]
        imgs = imgs.reshape((n,768,4,4))
        real_imgs = Variable(imgs.type(Tensor))
        z = generator.inverse(real_imgs)
        _, reconstructed=generator(z)

        save_image(real_imgs.data[:25].reshape(25,3,64,64), "reconstruct/{}/{}-original.png".format(args.dataset,i), nrow=5)
        save_image(reconstructed.data[:25].reshape(25,3,64,64), "reconstruct/{}/{}-reconstructed.png".format(args.dataset,i), nrow=5,normalize=True)

        if i==args.num_samples:
            break




def main(args):
    transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    trainset = torchvision.datasets.CelebA(root='data', split="train", download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=1)

    generator = torch.load("/home/chandu/Desktop/courses/bayesian/Invertible-GAN/celeba.model",
                            map_location=torch.device('cpu')).eval()

    reconstruct(args, generator, trainloader)


if __name__=="__main__":
    args = parseargs()
    os.makedirs("reconstruct/{}".format(args.dataset), exist_ok=True)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    z_avg=torch.load('savedzvectors/zavg')
    z0=z_avg[args.a][1]-z_avg[args.a][0]
    z0=z0

    main(args)