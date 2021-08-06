
import torch
import numpy as np
import time
from geomloss import SampleLoss

# if __name__ == "__main__":
#     torch.manual_seed(17)
#     strt_t = time.time()
#     N,C,S = 1,3,64
#     x = torch.rand((N,C,S,S))
#     x /= x.sum(dim=(1,2,3), keepdim=True) + 1e-35
#     y = torch.rand((N,C,S,S))
#     y /= y.sum(dim=(1,2,3), keepdim=True) + 1e-35
#     mid_t = time.time()

#     wass_dist = SampleLoss(loss="sinkhorn", p=2, blur=.05)
#     val = wass_dist(x, y)
#     print(val)

#     print("time: {:.3f}".format(time.time() - strt_t))
#     print("preprocess time: {:.3f}".format(mid_t - strt_t))

import torch
import geomloss  # See also ImagesLoss, VolumesLoss

# Create some large point clouds in 3D
x = torch.randn(100000, 3, requires_grad=True).cuda()
y = torch.randn(200000, 3).cuda()

# Define a Sinkhorn (~Wasserstein) loss between sampled measures
loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=.05)

L = loss(x, y)  # By default, use constant weights = 1/number of samples
g_x, = torch.autograd.grad(L, [x])  # GeomLoss fully supports autograd!