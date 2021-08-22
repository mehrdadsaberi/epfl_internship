import os
import argparse
import numpy as np
import torch
import models
import pandas as pd
import data
import setGPU

from robustbench.data import load_cifar10, load_cifar10c
from robustbench.utils import clean_accuracy

from resnet import ResNet18


def corr_eval(x_corrs, y_corrs, model):
    model.eval()
    res = np.zeros((5, 15))
    for i in range(1, 6):
        for j, c in enumerate(data.corruptions):
            res[i-1, j] = clean_accuracy(model, x_corrs[i][j].cuda(), y_corrs[i][j].cuda())
            print(c, i, res[i-1, j])

    return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--us_gpu', action='store_true')
    parser.add_argument('--output', default='output.csv', type=str)
    parser.add_argument('--only_clean', action='store_true')
    parser.add_argument('--n_samples', default=1000, type=int)
    return parser.parse_args()

corruptions = ["shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform"]

def main():
    args = get_args()
    device = "cuda"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str("6")
    x_clean, y_clean = load_cifar10(n_examples=args.n_samples, data_dir=args.data_dir)

    
    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

    models_info = [["FrankWolfe", "models/cifar_adv_training_attack-frank_eps-0.005.pth", "dict", None],
#                ["Wass 0.01", "models/cifar_adv_training_attack-l1wass_eps-0.01_epoch-30.pth", "dict", None],
#                ["Wass 0.005", "models/cifar_adv_training_attack-l1wass_eps-0.005_epoch-30.pth", "dict", None],
#                ["Wass 0.002", "models/cifar_adv_training_attack-l1wass_eps-0.002_epoch-30.pth", "dict", None],
                 ["Wass 0.005", "models/cifar_frank_eps-0.005_epoch-30.pth", "dict", None],
                 ["Wass 0.001", "models/cifar_frank_eps-0.001_epoch-30.pth", "dict", None],
                 ["Wass 0.001 (60ep)", "models/cifar_frank_eps-0.001_epoch-60-c.pth", "dict", None],
                 ["Wass 0.001 (150ep)", "models/cifar_frank_eps-0.001_epoch-150.pth", "dict", None],
                 ["Wass 0.00075", "models/cifar_frank_eps-0.00075_epoch-30.pth", "dict", None],
                 ["Wass 0.0005", "models/cifar_frank_eps-0.0005_epoch-30.pth", "dict", None],
                 ["Wass 0.0001", "models/cifar_frank_eps-0.0001_epoch-30.pth", "dict", None],
#                ["Wass 0.0008 (a,51ep)", "models/cifar_l1wass_eps-0.0008_alpha-0.005_epoch-51.pth", "dict", None],
#                ["Wass 0.01 (2)", "models/cifar_adv_training_attack-l1wass_eps-0.01_epoch-20.pth", "dict", None],
#                ["Wass 0.001 (2)", "models/cifar_adv_training_attack-l1wass_eps-0.001_epoch-20.pth", "dict", None],
#                ["Wass 0.0005 (2)", "models/cifar_adv_training_attack-l1wass_eps-0.0005_epoch-20.pth", "dict", None],
                ["L2", "models/l2-at-eps=0.1-cifar10.pt", "pre", None],
                ["Standard", "models/cifar_vanilla.pth", "net", None]]

    def print_info(name, elems):
        print("{} \t".format(name.ljust(20)), end="")
        for i in elems:
            print("{} \t".format(i.ljust(15)), end="")
        print()
    
        
    prnt = []
    for model in models_info:
        if model[2] == "dict":            
            model[-1] = ResNet18().cuda()
            ckpt = torch.load(model[1])
            model[-1].load_state_dict(ckpt["model_state_dict"])
            model[-1].eval()
        elif model[2] == "net":
            model[-1] = ResNet18().cuda()
            ckpt = torch.load(model[1])
            if "net" in ckpt.keys():
                for key in ckpt["net"].keys():
                    assert "module" in key
                ckpt["net"] = dict((key[7:], value) for key, value in ckpt["net"].items())
                model[-1].load_state_dict(ckpt["net"])
            model[-1].eval()
        elif model[2] == "pre":        
            model[-1] = models.PreActResNet18(n_cls=10, model_width=64, cifar_norm=True).cuda()
            model[-1].load_state_dict(torch.load(model[1])['last'])
            model[-1].eval()
        prnt += [model[0]]
    print_info("Corruption", prnt)

    prnt = []
    acc_cln = [0 for i in models_info]
    for i, model in enumerate(models_info):
        net = model[-1]
        if model[2] == "pre":
            acc_cln[i] = clean_accuracy(net, x_clean.cuda(), y_clean.cuda())
        elif model[2] == "dict" or model[2] == "net":
            acc_cln[i] = clean_accuracy(net, ((x_clean.cuda() - mu) / std), y_clean.cuda())
        prnt += ["{:.4f}".format(acc_cln[i])]
    print()
    print_info("Clean", prnt)
    


    mean_acc = [[0 for i in models_info] for j in range(6)]
    tot_mean = [0 for i in models_info]
    cnt = 0
    for corr in corruptions:
        print()
        for i in range(1,6):
            x_corr, y_corr = load_cifar10c(n_examples=args.n_samples, severity=i, corruptions=(corr,), data_dir=args.data_dir)
            cnt += 1

            prnt = []
            acc_corr = [0 for i in models_info]
            for j, model in enumerate(models_info):
                net = model[-1]
                if model[2] == "pre":
                    acc_corr[j] = clean_accuracy(net, x_corr.cuda(), y_corr.cuda())
                elif model[2] == "dict" or model[2] == "net":
                    acc_corr[j] = clean_accuracy(net, ((x_corr.cuda() - mu) / std), y_corr.cuda())
                mean_acc[i][j] += acc_corr[j]
                tot_mean[j] += acc_corr[j]
                prnt += ["{:.4f}".format(acc_corr[j])]
            print_info("{} {}".format(corr, i), prnt)

    for i in range(1,6):
        prnt = []
        for acc in mean_acc[i]:
            prnt += ["{:.4f}".format(acc / cnt * 5)]
        print()
        print_info("Average {}".format(i), prnt)

    prnt = []
    for acc in tot_mean:
        prnt += ["{:.4f}".format(acc / cnt)]
    print()
    print_info("Total Average", prnt)
    


    # if args.only_clean:
    #     return

    # x_corrs, y_corrs, _, _ = data.get_cifar10_numpy()
    
    # corr_res_last = corr_eval(x_corrs, y_corrs, model)
    # corr_data_last = pd.DataFrame({i+1: corr_res_last[i, :] for i in range(0, 5)}, index=data.corruptions)
    # corr_data_last.loc['average'] = {i+1: np.mean(corr_res_last, axis=1)[i] for i in range(0, 5)}
    # corr_data_last['avg'] = corr_data_last[list(range(1,6))].mean(axis=1)
    # corr_data_last.to_csv(args.output)
    # print(corr_data_last)


if __name__ == "__main__":
    main()
