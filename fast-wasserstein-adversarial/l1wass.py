import torch
import torch.nn as nn
from advertorch.attacks import Attack

from data import str2dataset
from model import str2model
from utils import *

from wasserstein_attack import WassersteinAttack
from lmo import entr_support_func
from projection import dual_capacity_constrained_projection


class L1Wasserstein(Attack):

    def __init__(self,
                 predict, loss_fn,
                 eps, alpha,
                 nb_iter,
                 device="cuda",
                 postprocess=False,
                 verbose=True,
                 ):
        
        super().__init__(predict, loss_fn, clip_min=0., clip_max=1.)
        self.eps = eps

        self.device = device

        """post-processing parameters"""
        self.postprocess = postprocess

        self.cost = None

        """variables supporting sparse matrices operations"""
        self.cost_indices = None
        self.forward_idx = None
        self.backward_idx = None

        """other parameters"""
        self.verbose = verbose

        """
        parameters for the ease of recording experimental results

        group 1:record (total projection/conjugate running time,
                        total projection/conjugate iterations,
                        total projection/conjugate function calls)
        """
        self.run_time = 0.0
        self.num_iter = 0
        self.func_calls = 0


        """group 2: flags for projected Sinkhorn"""
        self.converge = True
        self.overflow = False


        """group 3: loss and accuracy in each batch"""
        self.lst_loss = []
        self.lst_acc = []


        self.nb_iter = nb_iter
        self.alpha = alpha

        self.inf = 1000000

    def initialize_coupling(self, X):
        batch_size, c, h, w = X.size()
        pi = torch.zeros([batch_size, c, h, w, 4], dtype=torch.float, device=self.device)
        return pi

    def coupling2adversarial(self, pi, X):
        x0 = nn.ConstantPad1d((0, 0, 0, 1), 0)(pi[:,:,1:,:,0])
        x1 = nn.ConstantPad1d((1, 0, 0, 0), 0)(pi[:,:,:,:-1,1])
        x2 = nn.ConstantPad1d((0, 0, 1, 0), 0)(pi[:,:,:-1,:,2])
        x3 = nn.ConstantPad1d((0, 1, 0, 0), 0)(pi[:,:,:,1:,3])
        x4 = pi.sum(dim=4)
        return X - x4 + x0 + x1 + x2 + x3
    

    def l1wass_projection(self, G, X, inf, eps, verbose=False):
        batch_size, c, h, w = X.size()

        lam = X.new_zeros(batch_size).requires_grad_(True)
        mu = X.new_zeros((batch_size, c, h, w, 2)).requires_grad_(True)

        #lam_y = X.new_zeros(batch_size)
        #mu_y = X.new_zeros((batch_size, c, h, w, 2))
        #lam_y_prev = X.new_zeros(batch_size)
        #mu_y_prev = X.new_zeros((batch_size, c, h, w, 2))

        def recover(lam, mu):
            gam = X.new_zeros(G.size())
            gam += lam.view(-1, 1, 1, 1, 1)
            gam += mu[...,0:1] - mu[...,1:2]
            mu0 = nn.ConstantPad1d((0, 0, 0, 0, 0, 1), 0)(mu[:,:,1:,:,:])
            mu1 = nn.ConstantPad1d((0, 0, 1, 0, 0, 0), 0)(mu[:,:,:,:-1,:])
            mu2 = nn.ConstantPad1d((0, 0, 0, 0, 1, 0), 0)(mu[:,:,:-1,:,:])
            mu3 = nn.ConstantPad1d((0, 0, 0, 1, 0, 0), 0)(mu[:,:,:,1:,:])
            gam[...,0] += mu2[...,1] - mu2[...,0]
            gam[...,1] += mu3[...,1] - mu3[...,0]
            gam[...,2] += mu0[...,1] - mu0[...,0]
            gam[...,3] += mu1[...,1] - mu1[...,0]
            #print("GAMMA:", gam[1].min().item())
            pi_star = torch.max(torch.zeros_like(G), G - gam)
            return pi_star
        
        def grad(lam, mu, verbose=False, pi_tilde=None):
            if pi_tilde == None:
                pi_tilde = recover(lam, mu)

            d_lam = pi_tilde.sum(dim=(1, 2, 3, 4)) - eps
            #print(d_lam.max().item(), lam.max().item(), lam.min().item())

            adv_example = self.coupling2adversarial(pi_tilde, X)
            
            d_mu = X.new_zeros(mu.size())
            d_mu[...,0] = -adv_example
            d_mu[...,1] = adv_example - 1

            if verbose:
                print(i)
                print("ADV:", adv_example[79].max().item(), adv_example[79].min().item())
                print("X:", X[79].max().item(), X[79].min().item())
                print("EX COST:", (pi_tilde.sum(dim=(1, 2, 3, 4)) - eps)[79].item())
                print("INIT COST:", (G.sum(dim=(1, 2, 3, 4)) - eps)[79].item())
                print("PI:", pi_tilde[79].max().item())
                print("G:", G[79].max().item())
                print(lam[79].item(), mu[79].max().item(), mu[79].min().item())

            return d_lam, d_mu
        
        def calculate_g(lam, mu, pi=None):
            if pi == None:
                pi = recover(lam, mu)
            d_lam, d_mu = grad(lam, mu, pi_tilde=pi)
            #print(pi.size(), G.size(), lam.size(), d_lam.size(), mu.size(), d_mu.size())
            #print(torch.norm(pi - G, p=2, dim=(1,2,3,4)))
            return 1/2 * torch.norm(pi - G, p=2, dim=(1,2,3,4)) + lam * d_lam + (mu * d_mu).sum(dim=(1,2,3,4))

        N_ITER = 40
        #best_lam = torch.zeros_like(lam)
        #best_mu = torch.zeros_like(mu)
        #best_g = torch.ones_like(lam) * (-inf)

        #optimizer_lam = torch.optim.Adam([lam], lr=0.001, weight_decay=1e-5)
        #optimizer_mu = torch.optim.Adam([mu], lr=0.01)

        for i in range(N_ITER):

            pi = recover(lam, mu)
            #g_func = calculate_g(lam, mu, pi)
            
            #ids = torch.where(best_g < g_func, torch.ones_like(best_g), torch.zeros_like(best_g))
            #best_lam += (lam - best_lam) * ids 
            #best_mu += (mu - best_mu) * ids.view(-1,1,1,1,1)
            #best_g += (g_func - best_g) * ids

            d_lam, d_mu = grad(lam, mu, verbose=verbose, pi_tilde=pi)

            #optimizer_lam.zero_grad()
            #optimizer_mu.zero_grad()
            #lam.grad = -d_lam
            #mu.grad = -d_mu
            #optimizer_lam.step()
            #optimizer_mu.step()

            #print(lam.mean().item())

            #print(i, ":", g_func.mean().item())#, best_g.mean().item())
            #print("\t", lam.mean().item(), d_lam.max().item())
            #print("\t", mu.mean().item(), d_mu.max().item(), d_mu.min().item())

            #if i <= N_ITER / 4:
            #    eta = 1e-2
            #elif i <= N_ITER / 2:
            #    eta = 1e-3
            #elif i <= N_ITER * 3 / 4:
            #    eta = 1e-4
            #else :
            #    eta = 1e-5
            #eta = 0.01 / (np.sqrt(i) + 1)
            gamma = 0.9
            
            lam_eta = 0.001 / (np.sqrt(i) + 1)
            mu_eta = 0.5 / (np.sqrt(i) + 1)
            mu = (mu + mu_eta * d_mu).clamp(min=0.)
            lam = (lam + lam_eta * d_lam).clamp(min=0.)
            
            #tmp = mu_y.clone()
            #mu_y = (mu + mu_eta * d_mu).clamp(min=0.)
            #mu = ((1 + gamma) * tmp - gamma * mu_y_prev).clamp(min=0.)
            #tmp = lam_y.clone()
            #lam_y = (lam + lam_eta * d_lam).clamp(min=0.)
            #lam = ((1 + gamma) * tmp - gamma * lam_y_prev).clamp(min=0.)

            #mu_y_prev = mu_y
            #lam_y_prev = lam_y

            #if self.verbose and i % 100 == 0:
            #    print("iter {:5d}".format(i),
            #          "d_lam {:11.8f}".format(d_lam.max().item()),
            #          "d_mu max {:11.8f}".format(d_mu.max().item()),
            #          "d_mu min {:11.8f}".format(d_mu.min().item()),
            #          )
            #    print(i, lam.mean().item(), mu.mean().item())
                #print(G[0].max(), G[0].min(), G[0].sum())

        # up-weight lambda slightly to ensure the transportation cost constraint
        # pi_tilde = recover(lam + 1e-4, mu)

        #mu = mu_y
        #lam = lam_y

        #"""Run bisection method again to ensure strict feasibility of transportation cost constraint"""
        #lam = optimize_lam(mu)

        pi_tilde = recover(lam, mu)

        #adv_example = self.coupling2adversarial(pi_tilde, X)
        #print(lam.max().item(), mu.max().item())
        #print("INIT:", self.coupling2adversarial(G, X).max().item(), self.coupling2adversarial(G, X).min().item(), (G.sum(dim=(1, 2, 3, 4)) - eps).max().item())
        #print("FINAL:", adv_example.max().item(), adv_example.min().item(), (pi_tilde.sum(dim=(1, 2, 3, 4)) - eps).max().item())
        #print(adv_example.view(adv_example.size(0), -1).max(dim=-1))
        #print(eps.max().item())
        #print("ADV MAX MIN:", adv_example.max().item(), adv_example.min().item())
        #print((pi_tilde.sum(dim=(1, 2, 3, 4)) - eps).max().item())

        return pi_tilde, N_ITER


    '''def l1wass_simple_projection1(self, G, X, inf, eps, dual_max_iter, grad_tol, int_tol, verbose=False):
        batch_size, c, h, w = X.size()

        lo = X.new_zeros(batch_size,1,1,1,1)
        hi = X.new_ones(batch_size,1,1,1,1)

        N_ITER = 30
        for i in range(N_ITER):
            mid = (lo + hi) / 2
            pi = torch.min(G,mid)
            tmp = (pi.sum(dim=(1,2,3,4)) <= eps).view(-1,1,1,1,1)
            lo = torch.where(tmp == True, mid, lo)
            hi = torch.where(tmp == False, mid, hi)
          
        return torch.min(G,hi), N_ITER'''

    def l1wass_simple_projection(self, G, X, inf, eps, verbose=False, strict=False):
        batch_size, c, h, w = X.size()

        lo = X.new_zeros(batch_size,1,1,1,1)
        hi = X.new_ones(batch_size,1,1,1,1)

        N_ITER = 15
        G_z = torch.zeros_like(G)
        for i in range(N_ITER):
            mid = (lo + hi) / 2
            pi = torch.max(G_z, G - mid)
            tmp = (pi.sum(dim=(1,2,3,4)) <= eps).view(-1,1,1,1,1)
            lo = torch.where(torch.logical_not(tmp), mid, lo)
            hi = torch.where(tmp, mid, hi)
        if strict:
            pi = torch.max(torch.zeros_like(G), G - hi)
        #print("ZERO BEFORE:", (G == 0.).sum().item() / batch_size)
        #print("ZERO:", (pi == 0.).sum().item() / batch_size)
        #print("ALL:", (pi <= 100).sum().item() / batch_size)
        return pi, N_ITER

    
    def print_info(self, acc):
        print("accuracy under attack ------- {:.2f}%".format(acc))
        print("total dual running time ----- {:.3f}ms".format(self.run_time))
        print("total number of dual iter --- {:d}".format(self.num_iter))
        print("total number of fcall ------- {:d}".format(self.func_calls))

    def save_info(self, acc, save_info_loc):
        torch.save((acc,
                    self.run_time,
                    self.num_iter,
                    self.func_calls,
                    self.overflow,
                    self.converge,
                    self.lst_loss,
                    self.lst_acc,
                    ),
                   save_info_loc)

    
    def check_nonnegativity(self, pi, tol=1e-4, verbose=False):
        diff = pi.clamp(max=0.).abs().sum(dim=(1, 2, 3, 4)).max().item()
        assert diff < tol

    def check_transport_cost(self, pi, tol=1e-4, verbose=False):
        diff = pi.sum(dim=(1, 2, 3, 4)).max().item()
        if verbose:
            print("Max Cost:", diff, ", EPS:", self.eps)
        #assert diff < self.eps + tol
    
    def check_hypercube(self, adv_example, tol=1e-3, verbose=False):
        if verbose:
            print("----------------------------------------------")
            print("num of pixels that exceed exceed one {:d}  ".format((adv_example > 1.).sum(dim=(1, 2, 3)).max().item()))
            print("maximum pixel value                  {:f}".format(adv_example.max().item()))
            print("total pixel value that exceed one    {:f}".format((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)).max().item()))
            print("% of pixel value that exceed one     {:f}%".format(
                100 * ((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)) / adv_example.sum(dim=(1, 2, 3))).max().item()))
            print("----------------------------------------------")

        if tol is not None:
            if verbose:
                print("ADV Min Max:", adv_example.min().item(), adv_example.max().item())
            #assert(adv_example.max().item() < 1 + tol)
            #assert(adv_example.min().item() > -tol)

    
    def perturb(self, X, y):
        batch_size, c, h, w = X.size()

        pi = self.initialize_coupling(X).clone().detach().requires_grad_(True)
        normalization = X.sum(dim=(1, 2, 3), keepdim=True)
        normalization_pi = X.sum(dim=(1, 2, 3)).view(-1, 1, 1, 1, 1)

        for t in range(self.nb_iter):
            #print(pi.sum(), pi.max())
            adv_example = self.coupling2adversarial(pi, X)
            #self.check_hypercube(adv_example, verbose=False)
            #print(self.coupling2adversarial(pi, X / normalization).max().item(), normalization.max().item())
            #print((adv_example-X).abs().max())
            scores = self.predict(adv_example.clamp(min=self.clip_min, max=self.clip_max))

            loss = self.loss_fn(scores, y)
            loss.backward()

            with torch.no_grad():
                self.lst_loss.append(loss.item())
                self.lst_acc.append((scores.max(dim=1)[1] == y).sum().item())

                """Add a small constant to enhance numerical stability"""
                pi.grad /= (tensor_norm(pi.grad, p='inf').view(batch_size, 1, 1, 1, 1) + 1e-35)
                assert (pi.grad == pi.grad).all() and (pi.grad != float('inf')).all() and (pi.grad != float('-inf')).all()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                pi += self.alpha * pi.grad
                pi = pi.clamp(min=0.)

                start.record()

                
                #pi, num_iter = self.l1wass_projection(pi,
                pi, num_iter = self.l1wass_simple_projection(pi,
                                          X,
                                          inf=self.inf,
                                          eps=self.eps * normalization.squeeze(),
                                          verbose=False)
                end.record()

                torch.cuda.synchronize()

                self.run_time += start.elapsed_time(end)
                self.num_iter += num_iter
                self.func_calls += 1

                if self.verbose and (t + 1) % 1 == 0:
                    print("num of iters : {:4d}, ".format(t + 1),
                          "loss : {:12.6f}, ".format(loss.item()),
                          "acc : {:5.2f}%, ".format((scores.max(dim=1)[1] == y).sum().item() / batch_size * 100),
                          "dual iter : {:2d}, ".format(num_iter),
                          "per iter time : {:7.3f}ms".format(start.elapsed_time(end) / num_iter))

                #self.check_nonnegativity(pi / normalization_pi, tol=1e-6, verbose=False)
                #self.check_marginal_constraint(pi / normalization_pi, X / normalization, tol=1e-6, verbose=False)
                #self.check_transport_cost(pi / normalization_pi, tol=1e-3, verbose=False)
    
            pi = pi.clone().detach().requires_grad_(True)


        with torch.no_grad():
            adv_example = self.coupling2adversarial(pi, X)
            #print(torch.sum(0 < (torch.abs(adv_example - X))))
            #print(torch.sum(X > 0))
            #print(torch.mean(torch.abs(normalization - adv_example.sum(dim=(1, 2, 3), keepdim=True))))
            self.check_hypercube(adv_example, verbose=self.verbose)
            self.check_nonnegativity(pi / normalization_pi, tol=1e-5, verbose=self.verbose)
            #self.check_marginal_constraint(pi / normalization_pi, X / normalization, tol=1e-5, verbose=self.verbose)
            self.check_transport_cost(pi / normalization_pi, tol=self.eps * 1e-2, verbose=self.verbose)

        """Do not clip the adversarial examples to preserve pixel mass"""
        return adv_example


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--checkpoint', type=str, default='mnist_vanilla')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_batch', type=int_or_none, default=5)

    parser.add_argument('--eps', type=float, default=0.5, help='the perturbation size')
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--nb_iter', type=int, default=20)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--postprocess', type=str2bool, default=False)

    parser.add_argument('--save_img_loc', type=str_or_none, default=None)
    parser.add_argument('--save_info_loc', type=str_or_none, default=None)

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(args.seed)

    testset, normalize, unnormalize = str2dataset(args.dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = str2model(args.checkpoint, dataset=args.dataset, pretrained=True).eval().to(device)

    for param in net.parameters():
        param.requires_grad = False

    l1_wass = L1Wasserstein(predict=lambda x: net(normalize(x)),
                             loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                             eps=args.eps,
                             alpha=args.alpha,
                             nb_iter=args.nb_iter,
                             device=device,
                             postprocess=args.postprocess,
                             verbose=True)

    acc = test(lambda x: net(normalize(x)),
               testloader,
               device=device,
               attacker=l1_wass,
               num_batch=args.num_batch,
               save_img_loc=args.save_img_loc)

    l1_wass.print_info(acc)

    if args.save_info_loc is not None:
        l1_wass.save_info(acc, args.save_info_loc)
