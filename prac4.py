import torch


# x = torch.randn(5,)
#
# idx = torch.randint(0,4,(5,))
#
#
# print(x)
# print(idx)
#
# print(torch.index_select(x,0,idx))
import torch.nn.functional as F


x = torch.randn(5,2)

y= F.linear(x,x)

print(y)
for i in range(len(y)):
    y[i,i] = 0

print(y)