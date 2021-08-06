import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import set_seed
from data import str2dataset
from model import str2model

from frank_wolfe import FrankWolfe
from l1wass import L1Wasserstein

import setGPU


def train(model, loader, device, lr, epoch, attacker, args):
    model.train()

    lr_schedule = lambda t: np.interp([t], [0, epoch // 2, epoch], [0., lr, 0.])[0]
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    for i in range(args.resume + 1, epoch):
        correct = 0
        total = 0
        total_loss = 0
        cln_correct = 0
        total_cln_loss = 0
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for batch_idx, (cln_data, target) in enumerate(loader):
            cln_data, target = cln_data.to(device), target.to(device)
            
            cur_lr = lr_schedule(i + (batch_idx + 1) / len(loader))
            optimizer.param_groups[0].update(lr=cur_lr)

            for param in model.parameters():
                param.requires_grad = False

            adv_data = attacker.perturb(cln_data, target)

            for param in model.parameters():
                param.requires_grad = True

            optimizer.zero_grad()

            scores = model(adv_data)
            loss = loss_fn(scores, target)
            cln_scores = model(cln_data)
            cln_loss = loss_fn(cln_scores, target)
            loss.backward()

            optimizer.step()

            cur_correct = scores.max(dim=1)[1].eq(target).sum().item()
            cur_cln_correct = cln_scores.max(dim=1)[1].eq(target).sum().item()

            correct += cur_correct
            cln_correct += cur_cln_correct
            total += scores.size(0)
            total_loss += loss.item()
            total_cln_loss += cln_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print("\t batch: {:4d}/{:4d} \t loss: {:.4f} acc: {:3.3f}% \t clean loss: {:.4f} clean acc: {:3.3f}% \t lr: {:.5f}"
                    .format(batch_idx, len(loader), loss.item(), 100. * cur_correct / scores.size(0), cln_loss.item(), 100. * cur_cln_correct / scores.size(0), cur_lr))
            
        end.record()

        print("epoch: {:4d}/{:4d} \t loss: {:.4f} acc: {:3.3f}% \t clean loss: {:.4f} clean acc: {:3.3f}% \t time: {:7.3f}s  lr: {:.5f}"
              .format(i + 1, epoch, total_loss / len(loader), 100. * correct / total, total_cln_loss / len(loader), 100. * cln_correct / total, start.elapsed_time(end) / 1000, cur_lr))
        
        ckpt = {'model_state_dict': model.state_dict()}

        if args.save_model_loc is None:
            torch.save(ckpt, "./checkpoints/{}_adv_training_attack-{}_eps-{}_epoch-{}.pth".format(args.dataset, args.attack, args.eps, i + 1))
        else:
            torch.save(ckpt, args.save_model_loc)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=20)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--nb_iter', type=int, default=40, help='number of attack iterations')

    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--save_model_loc', type=str, default=None)

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(0)

    #testset, normalize, unnormalize = str2dataset(args.dataset, train=False)
    trainset, normalize, unnormalize = str2dataset(args.dataset, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    net = str2model(path=args.save_model_loc, dataset=args.dataset, pretrained=args.resume).eval().to(device)

    if args.attack == "frank":
        attacker = FrankWolfe(predict=lambda x: net(normalize(x)),
                              loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                              eps=args.eps,
                              kernel_size=5,
                              nb_iter=args.nb_iter,
                              entrp_gamma=0.1,
                              dual_max_iter=30,
                              grad_tol=1e-4,
                              int_tol=1e-4,
                              device=device,
                              postprocess=False,
                              verbose=False)
    elif args.attack == "l1wass":
        attacker = L1Wasserstein(predict=lambda x: net(normalize(x)),
                              loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                              eps=args.eps,
                              alpha=args.alpha,
                              nb_iter=args.nb_iter,
                              device=device,
                              postprocess=False,
                              verbose=False)
    
    else:
        assert 0

    train(net, trainloader, device, args.lr, args.epoch, attacker, args)

    ckpt = {'model_state_dict': net.state_dict()}

    if args.save_model_loc is None:
        torch.save(ckpt, "./checkpoints/{}_adv_training_attack-{}{}_eps-{}.pth".format(args.dataset, args.attack, args.epoch, args.eps))
    else:
        torch.save(ckpt, args.save_model_loc)