import torch
from models.irevnet.iRevNet import iRevNet


nClasses = 10
in_shape = [4,8,8]
bottleneck_mult = 4
init_ds = 0
nBlocks = [18, 18, 18]
nStrides = [1, 2, 2]
nChannels = [8, 32, 128]

model = iRevNet(nBlocks=nBlocks, nStrides=nStrides,
				nChannels=nChannels, nClasses=nClasses,
				init_ds=init_ds, dropout_rate=0.1, affineBN=True,
				in_shape=in_shape, mult=bottleneck_mult)
model.eval()
# print(model)
test_inp = torch.randn(15,4,8,8)
a,b = model(test_inp)
c = model.inverse(b)

print(b.shape)
print(c.shape)
# print(c==test_inp)
print(torch.sum(torch.abs(c-test_inp)))
print(torch.equal(c,test_inp))
b.reshape(15,32,32)