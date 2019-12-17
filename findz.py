import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import pickle
import os
import torch.utils.data as data


def main():
    transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    trainset = torchvision.datasets.CelebA(root='data', split="train", download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=1)


    generator = torch.load("/home/chandu/Desktop/courses/bayesian/Invertible-GAN/celeba.model",
                            map_location=torch.device('cpu'))

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(trainloader):
            print(i, len(trainloader))
            print('*'*50)
            n=imgs.shape[0]
            # print(imgs.shape)
            imgs = imgs.reshape((n,768,4,4))
            # print(imgs.shape)
            z = generator.inverse(imgs)
            # print(z.shape)
            for j in range(n):
                for l in range(40):
                    if labels[j][l]==1:
                        numeachcat[l][1]+=1
                        zavg[l][1]+=z[j]
                    else:
                        numeachcat[l][0]+=1
                        zavg[l][0]+=z[j]
						
				
    torch.save(zavg,'./savedzvectors/zsum')
    for x in range(40):
        for y in range(2):
            zavg[x][y]=zavg[x][y]/numeachcat[x][y]


    print(numeachcat)

if __name__ == '__main__':
    os.makedirs('./savedzvectors',exist_ok=True)
    numeachcat=[[0,0] for i in range(40)]
    zavg=[[torch.zeros(8,16,16,requires_grad=False) for i in range(2)] for j in range(40)]
    main()
    torch.save(numeachcat,'./savedzvectors/numeachcat')
    torch.save(zavg,'./savedzvectors/zavg')