import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import ot
import time
import matplotlib.pyplot as plt
import os



# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='RGB'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace

class Wasserstein_Sinkhorn(FakeNet):
    def __init__(self, use_gpu=True, colorspace='RGB', cost='L1', lam=0.1):
        super(Wasserstein_Sinkhorn, self).__init__(use_gpu, colorspace)
        self.cost = cost
        self.is_M = False
        self.M = None
        self.lam = lam
        self.cnt_plot = 0

    def sinkhorn_knopp(self, a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, ret_pi=True, **kwargs):

        N = a.shape[0]
        dim_a = a.shape[1]
        dim_b = b.shape[1]
        #a, b, M = list_to_array(a, b, M)

        u = M.new_ones((N, dim_a)) / dim_a
        v = M.new_ones((N, dim_b)) / dim_b

        K = torch.exp(M / (-reg))

        cpt = 0
        err = 1
        while (err > stopThr and cpt < numItermax):
            uprev = u
            vprev = v


            v = b / torch.mm(u, K)
            u = a / torch.mm(v,K.T)

            

            er_inx = (torch.isnan(u) + torch.isnan(v) + torch.isinf(u) + torch.isinf(v)).sum(dim=1, keepdim=True)

            u = torch.where(er_inx > 0, uprev, u)
            v = torch.where(er_inx > 0, vprev, v)


            if (torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                assert(False)

            cpt = cpt + 1

        if not ret_pi:  # return only loss
            return torch.einsum('bi,ij,bj,ij->b', u, K, v, M)

        else:  # return OT matrix
            return torch.einsum('bi,ij,bj->bij', u, K, v)


    def pi_analyze(self, pi, dist):
        try:
            os.mkdir("pi_analyze/")
        except:
            pass
        pi = pi.mean(dim=0)
        X = int(pi.shape[0] ** (1/2))
        tmp_cnt = [0 for i in range(2 * X)]
        thresh_val = 0

        for k1 in range(pi.shape[0]):
            for k2 in range(pi.shape[1]):
                if k1 == k2:
                    continue
                i1 = k1 // X
                j1 = k1 % X
                i2 = k2 // X
                j2 = k2 % X 
                tmp_cnt[abs(i1 - i2) + abs(j1 - j2)] += pi[k1, k2]
                if abs(i1 - i2) + abs(j1 - j2) > 10:
                    thresh_val += pi[k1, k2]
                        
        plt.bar(range(len(tmp_cnt)), tmp_cnt)
        plt.title("Wass dist: {:.6f}, More than 10: {}".format(dist.mean().item(), thresh_val))
        plt.savefig("pi_analyze/{:03d}.png".format(self.cnt_plot))
        self.cnt_plot += 1
        plt.close()
        

    def forward(self, in0, in1, retPerLayer=None):
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
                #         self.M[p1,p2] = abs(i1 - i2)  + abs(j1 - j2)
                # torch.save(self.M,"cmats/l1.pt")
                # print("cost matrix calculation done! time:",time.time() - tmp_time)
                # exit()
                self.M = torch.load("cmats/l1.pt", map_location=in0.device)
                self.is_M = True
            else:
                assert(False)

        if(self.colorspace=='RGB'):
            in0 = in0 * 0.5 + 0.5
            in1 = in1 * 0.5 + 0.5

            value = self.M.new_zeros(N)
            #s0 = in0.sum(dim=(1,2,3)) + 1e-35
            #s1 = in1.sum(dim=(1,2,3)) + 1e-35
            #print(s0, s1)
            #value += 0.1 * torch.norm(in0 - in1, p=2, dim=(1,2,3)) # 1. * (torch.max(s0 / s1, s1 / s0) - 1.0)
            #print(value)

            #in0 = in0.mean(dim=1)
            #in1 = in1.mean(dim=1)
            #in0 = in0.view(N, X * Y).clamp(min=3e-3)
            #in1 = in1.view(N, X * Y).clamp(min=3e-3)

            in0 = in0.view(N * C, X * Y).clamp(min=3e-4)
            in1 = in1.view(N * C, X * Y).clamp(min=3e-4)

            in0 /= in0.sum(dim=1, keepdim=True) + 1e-35
            in1 /= in1.sum(dim=1, keepdim=True) + 1e-35

            PI = self.sinkhorn_knopp(in0, in1, self.M, self.lam)

            dist = (PI * self.M).sum(dim=(1,2))
            self.pi_analyze(PI, dist)
            #log_pi = torch.log(PI)
            #log_pi[log_pi == -float("inf")] = 0
            #dist += lam * (PI * log_pi).sum(dim=(1,2))
            value += dist.view(N, C).sum(dim=1)
            #print(value)
            
            #print((in0 - PI.sum(dim=2)).abs().sum(dim=1))
            #print((in1 - PI.sum(dim=1)).abs().sum(dim=1))
            #input()
            #print(value)
            #exit()
            return value
        elif(self.colorspace=='Lab'):
            assert(False)

