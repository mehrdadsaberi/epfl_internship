import torch
import torch.nn.functional as F
import numpy as np
import math
import ot

import time
import sys


MAX_FLOAT = 1e38# 1.7976931348623157e+308

def any_nan(X): 
    return (X != X).any().item()
def any_inf(X): 
    return (X == float('inf')).any().item()


# batch dot product
def _bdot(X,Y): 
    return torch.matmul(X.unsqueeze(-2), Y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

def _unfold(x, kernel_size, padding=None): 
    # this is necessary because unfold isn't implemented for multidimensional batches
    size = x.size()
    if len(size) > 4: 
        x = x.contiguous().view(-1, *size[-3:])
    out = F.unfold(x, kernel_size, padding=kernel_size//2)
    if len(size) > 4: 
        out = out.view(*size[:-3], *out.size()[1:])
    return out


def _mm2(A, x, y, shape): 
    kernel_size = A.size(-1)
    nfilters = shape[1]
    unfolded = _unfold(x, kernel_size, padding=kernel_size//2).transpose(-1,-2)
    unfolded = _expand(unfolded, (A.size(-3),A.size(-2)*A.size(-1))).transpose(-2,-3)
    unfolded = unfolded * y.view(*shape[:-2],-1).unsqueeze(-1)
    out = torch.matmul(unfolded, collapse2(A.contiguous()).unsqueeze(-1)).squeeze(-1)
    return unflatten2(out)

def _mm(A, x, shape): 
    kernel_size = A.size(-1)
    nfilters = shape[1]
    unfolded = _unfold(x, kernel_size, padding=kernel_size//2).transpose(-1,-2)
    unfolded = _expand(unfolded, (A.size(-3),A.size(-2)*A.size(-1))).transpose(-2,-3)
    out = torch.matmul(unfolded, collapse2(A.contiguous()).unsqueeze(-1)).squeeze(-1)
    return unflatten2(out)

def _cost(A, x, y, shape, C, mx, my):
    kernel_size = A.size(-1)
    nfilters = shape[1]
    unfolded = _unfold(x, kernel_size, padding=kernel_size//2).transpose(-1,-2)
    unfolded = _expand(unfolded, (A.size(-3),A.size(-2)*A.size(-1))).transpose(-2,-3)
    unfolded = unfolded * y.view(*shape[:-2],-1).unsqueeze(-1)
    #print(collapse2(A.contiguous()) * C.view(-1))
    #print(A.size())
    #print(A)
    #exit()
    out = torch.matmul(unfolded, (collapse2(A.contiguous()) * C.view(-1)).unsqueeze(-1)).squeeze(-1)
    out = unflatten2(out)
    out *= torch.exp(mx + my)
    return out.sum(dim=(1,2,3))

def wasserstein_cost(X, p=2, kernel_size=5):
    if kernel_size % 2 != 1: 
        raise ValueError("Need odd kernel size")
        
    center = kernel_size // 2
    C = X.new_zeros(kernel_size,kernel_size)
    for i in range(kernel_size): 
        for j in range(kernel_size): 
            C[i,j] = (abs(i-center)**p + abs(j-center)**p)**(1/p)
    return C

def unsqueeze3(X):
    return X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def collapse2(X): 
    return X.view(*X.size()[:-2], -1)

def collapse3(X): 
    return X.view(*X.size()[:-3], -1)

def _expand(X, shape): 
    return X.view(*X.size()[:-1], *shape)

def unflatten2(X): 
    # print('unflatten2', X.size())
    n = X.size(-1)
    k = int(math.sqrt(n))
    return _expand(X,(k,k))

def _expand_filter(X, nfilters): 
    sizes = list(-1 for _ in range(X.dim()))
    sizes[-3] = nfilters
    return X.expand(*sizes)


def bdot(x,y): 
    return _bdot(collapse3(x), collapse3(y))

def log_sinkhorn(X, Y, C, lam, verbose=False, plan=False,
    maxiters=100, return_objective=False, return_iters=False, termination=None):

    #print("start wass calc")
    batch_sizes = X.size()[:-3]
    nfilters = X.size(-3)

    # size check
    for xd,yd in (zip(reversed(X.size()),reversed(Y.size()))): 
        assert xd == yd or xd == 1 or yd == 1

    # helper functions
    expand3 = lambda x: _expand(x, X.size()[-3:])
    expand_filter = lambda x: _expand_filter(x, X.size(-3))
    mm = lambda A,x: _mm(expand_filter(A),x,X.size())
    mm2 = lambda A,x,y: _mm2(expand_filter(A),x,y,X.size())
    get_cost = lambda A,x,y,mx,my: _cost(expand_filter(A),x,y,X.size(),C, mx, my)
    get_pi_sum = lambda A,x,y,mx,my: _cost(expand_filter(A),x,y,X.size(),torch.ones_like(C), mx, my)
    norm = lambda x: torch.norm(collapse3(x), dim=-1)
    # like numpy
    if termination is None:
        allclose = lambda x,y: (x-y).abs() <= 1e-4 + 1e-4*y.abs()
    else:
        abso = termination[0]
        rela = termination[1]
        allclose = lambda x, y: (x - y).abs() <= abso + rela * y.abs()
        # allclose = lambda x,y: (x-y).abs() <= 1e-4 + 1e-7 * y.abs()
    
    # assert valid distributions
    assert (X>=0).all()
    assert ((collapse3(X).sum(-1) - 1).abs() <= 1e-4).all()
    assert X.dim() == Y.dim()

    size = tuple(max(sx,sy) for sx,sy in zip(X.size(), Y.size()))

    # total dimension size for each example
    m = collapse3(X).size(-1)
    
    alpha = -torch.log(X.new_ones(*size)/m)
    beta = -torch.log(X.new_ones(*size)/m)
    exp_alpha = torch.exp(alpha)
    exp_beta = torch.exp(beta)

    lam_v = X.new_ones(*size[:-3]) * lam
    K = torch.exp(-unsqueeze3(lam_v) * C - 1)

    def eval_obj(alpha, beta, exp_alpha, exp_beta, K, max_alpha, max_beta): 
        return (bdot(torch.clamp(alpha, min=-1e10), X) 
                + bdot(torch.clamp(beta, min=-1e10), Y)
                - get_pi_sum(K, exp_alpha, exp_beta, max_alpha, max_beta))

    old_obj = -float('inf')
    i = 0

    if verbose:
        start_time = time.time()


    with torch.no_grad(): 
        while True: 
          
            max_beta = torch.amax(beta, dim=(2, 3), keepdim=True)
            exp_beta = torch.exp(beta - max_beta).clamp(min=1e-40)
            alpha = (-torch.log(mm(K,exp_beta)) + torch.log(X) - max_beta)
            
            max_alpha = torch.amax(alpha, dim=(2, 3), keepdim=True)
            exp_alpha = torch.exp(alpha - max_alpha).clamp(min=1e-40)
            beta = (-torch.log(mm(K,exp_alpha)) + torch.log(Y) - max_alpha)
            
            #if i % 1 == 0:
            #    print(i, "||||", eval_obj(alpha, beta, exp_alpha, exp_beta, K, max_alpha, max_beta).min().item(), get_cost(K, exp_alpha, exp_beta, max_alpha, max_beta).max().item(), alpha.min().item(), alpha.max().item(), beta.min().item(), beta.max().item())
            #    print("abs(X-PI1):", (X - mm2(K, exp_beta, exp_alpha) * torch.exp(max_alpha + max_beta)).abs().sum(dim=(1,2,3)).max().item(), end=", ")
            #    print("abs(Y-PI^T1):", (Y - mm2(K, exp_alpha, exp_beta) * torch.exp(max_alpha + max_beta)).abs().sum(dim=(1,2,3)).max().item())
                #print("\t", exp_alpha.min(), exp_alpha.max(), exp_beta.min(), exp_beta.max())
                #print("\t", any_nan(exp_alpha), any_nan(exp_beta))
            
            # check for convergence
            obj = 0 #eval_obj(alpha, exp_alpha, beta, exp_beta, K) 
            i += 1
            if i > maxiters:# or allclose(old_obj,obj).all(): 
                if verbose: 
                    print('terminate at iteration: {:5d} max iteration: {:5d} average dual objective: {:15.6f}'.format(i, maxiters, obj.mean().item()))
                    if i > maxiters: 
                        print('warning: took more than {} iters'.format(maxiters))
                break
            old_obj = obj

            # check for overflow
            if any_nan(exp_alpha) or any_inf(exp_alpha) or \
                any_nan(exp_beta) or any_inf(exp_beta):
                print('Overflow error: in logP_sinkhorn')
                return None, None

    #print((X - mm2(K, exp_beta, exp_alpha) * torch.exp(max_alpha + max_beta)).abs().sum(dim=(1,2,3)).max())
    #print((Y - mm2(K, exp_alpha, exp_beta) * torch.exp(max_alpha + max_beta)).abs().sum(dim=(1,2,3)).max())
    #print(get_cost(K, exp_alpha, exp_beta, max_alpha, max_beta))
    #exit()
    #print("end wass calc")
    return get_cost(K, exp_alpha, exp_beta, max_alpha, max_beta)


if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(17)
    strt_t = time.time()
    N,C,S = 1,3,64
    x = torch.rand((N,C,S,S))
    x /= x.sum(dim=(1,2,3), keepdim=True) + 1e-35
    y = torch.rand((N,C,S,S))
    y /= y.sum(dim=(1,2,3), keepdim=True) + 1e-35
    cost = wasserstein_cost(x, p=2, kernel_size = 5)
    mid_t = time.time()
    print(log_sinkhorn(x, y, cost, 1., maxiters=1000))
    print("time: {:.3f}".format(time.time() - strt_t))
    print("preprocess time: {:.3f}".format(mid_t - strt_t))