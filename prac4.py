from torchvision.datasets import STL10,CIFAR10

dt = CIFAR10(root='~/',train=True,download=True)

inputs = dt.data
x = 0
for i in inputs:
    print(type(i),i.shape)

