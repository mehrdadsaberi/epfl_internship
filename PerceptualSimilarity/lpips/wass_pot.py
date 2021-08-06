import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import ot
import ot.gromov
import time

import lpips

import matplotlib.pyplot as plt


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace

class Wasserstein_POT(FakeNet):
    def __init__(self, use_gpu=True, colorspace='Lab', cost='L1'):
        super(Wasserstein_POT, self).__init__(use_gpu, colorspace)
        self.cost = cost
        self.is_M = False
        self.M = None

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1
        (N,C,X,Y) = in0.size()
        
        if not self.is_M:
            if self.cost == 'L1':
                # print("\nstart calculating cost matrix...")
                # tmp_time = time.time()
                # self.M = torch.zeros(X * Y, X * Y)
                # for p1 in range(X * Y):
                #     for p2 in range(X * Y):
                #         i1 = p1 // Y
                #         j1 = p1 - i1 * Y
                #         i2 = p2 // Y
                #         j2 = p2 - i2 * Y
                #         self.M[p1,p2] = (i1 - i2) ** 2  + (j1 - j2) ** 2
                # torch.save(self.M,"cmats/l2.pt")
                # print("cost matrix calculation done! time:",time.time() - tmp_time)
                # exit()
                self.M = torch.load("cmats/l2.pt")
                self.M = np.array(self.M)
                self.is_M = True
            else:
                assert(False)

        if(self.colorspace=='RGB'):
            value = torch.zeros(N)
            in0 = in0 * 0.5 + 0.5
            in1 = in1 * 0.5 + 0.5
            in0 = in0.clamp(min=1e-4)
            in1 = in1.clamp(min=1e-4)

            for i in range(C):
                x0 = in0[0,i].view(-1).cpu().numpy().astype(np.float64)
                x1 = in1[0,i].view(-1).cpu().numpy().astype(np.float64)
                #print(np.min(x0))
                x0 /= np.linalg.norm(x0,1) + 1e-35
                x1 /= np.linalg.norm(x1,1) + 1e-35
                x0 = np.array(x0)
                x1 = np.array(x1)
                #print(np.min(x0))
                #if np.isnan(x0) or np.isnan(x1) or abs(np.linalg.norm(x0,1) - 1.) > 1e-12 or abs(np.linalg.norm(x1,1) - 1.) > 1e-12:
                #    print(np.linalg.norm(x0,1), np.linalg.norm(x1,1))
                #value += ot.sinkhorn2(x0, x1, self.M, 1.)
                
                pi = torch.tensor(ot.emd(x0, x1, self.M))
                val = torch.sum(pi * self.M)

                # for lam in [10., 1., 0.1, 0.01, 0.001]:
                #     PI = np.array(ot.sinkhorn(x0, x1, self.M, lam))
                #     val = np.sum(PI * self.M)
                #     print("\nLam = {}".format(lam))
                #     print("X0 diff: {}".format(np.sum(np.abs(x0 - np.sum(PI, axis=1)))))
                #     print("X1 diff: {}".format(np.sum(np.abs(x1 - np.sum(PI, axis=0)))))
                #     print("Images diff: {}".format(np.sum(np.abs(x0 - x1))))
                #     print("Wass dist: {}".format(val))

                #input()

                #val = ot.gromov.gromov_wasserstein2(C1=self.M, C2=self.M, p=x0, q=x1, loss_fun='square_loss', verbose=True, numItermax=10)
                value += val
                #print("DONE")
            if(self.use_gpu):
                value = value.cuda()
            return value
        elif(self.colorspace=='Lab'):
            assert(False)

