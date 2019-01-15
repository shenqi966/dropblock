import torch
from torch.distributions import Bernoulli


mask_sizes = [7,7]
bbb = Bernoulli(torch.tensor(0.9)).sample((3, *mask_sizes))

print(bbb)
print(bbb.size())

