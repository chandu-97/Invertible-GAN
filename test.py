import torch
from models.irevnet.iRevNet import iRevNet

# MNIST
nClasses = 10
in_shape = [4,8,8]
bottleneck_mult = 4
init_ds = 0
nBlocks = [18, 18, 18]
nStrides = [1, 2, 2]
nChannels = [8, 32, 128]

# CIFAR and CELEBA32
nClasses = 10
in_shape = [8,8,8]
bottleneck_mult = 4
init_ds = 0
nBlocks = [18, 18, 18]
nStrides = [1, 2, 2]
nChannels = [8*3, 32*3, 128*3]

# CELEBA64
nClasses = 10
in_shape = [8,16,16]
bottleneck_mult = 4
init_ds = 0
nBlocks = [18, 18, 18]
nStrides = [1, 2, 2]
nChannels = [8*3, 32*3, 128*3]

model = iRevNet(nBlocks=nBlocks, nStrides=nStrides,
				nChannels=nChannels, nClasses=nClasses,
				init_ds=init_ds, dropout_rate=0.1, affineBN=True,
				in_shape=in_shape, mult=bottleneck_mult)
model.eval()
# print(model)
shape = [15]
shape.extend(in_shape)
shape = tuple(shape)
test_inp = torch.randn(*shape)
a,b = model(test_inp)
c = model.inverse(b)

print(b.shape)
print(c.shape)
# print(c==test_inp)
print(torch.sum(torch.abs(c-test_inp)))
print(torch.equal(c,test_inp))
b = b.reshape(15,3,64,64)
print(b.shape)