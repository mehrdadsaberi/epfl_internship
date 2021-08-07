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
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    x_clean, y_clean = load_cifar10(n_examples=args.n_samples, data_dir=args.data_dir)

    
    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

    std_model = ResNet18().cuda()
    ckpt = torch.load("models/cifar_vanilla.pth")
    if "net" in ckpt.keys():
        for key in ckpt["net"].keys():
            assert "module" in key
        ckpt["net"] = dict((key[7:], value) for key, value in ckpt["net"].items())
        std_model.load_state_dict(ckpt["net"])
    std_model.eval()

    fw_model = ResNet18().cuda()
    ckpt = torch.load("models/cifar_adv_training_attack-frank_eps-0.005.pth")
    fw_model.load_state_dict(ckpt["model_state_dict"])
    fw_model.eval()

    l1_1_model = ResNet18().cuda()
    ckpt = torch.load("models/cifar_adv_training_attack-l1wass_eps-0.01_epoch-30.pth")
    l1_1_model.load_state_dict(ckpt["model_state_dict"])
    l1_1_model.eval()

    l1_2_model = ResNet18().cuda()
    ckpt = torch.load("models/cifar_adv_training_attack-l1wass_eps-0.005_epoch-30.pth")
    l1_2_model.load_state_dict(ckpt["model_state_dict"])
    l1_2_model.eval()

    l1_3_model = ResNet18().cuda()
    ckpt = torch.load("models/cifar_adv_training_attack-l1wass_eps-0.002_epoch-30.pth")
    l1_3_model.load_state_dict(ckpt["model_state_dict"])
    l1_3_model.eval()

    l1_4_model = ResNet18().cuda()
    ckpt = torch.load("models/cifar_adv_training_attack-l1wass_eps-0.0008_epoch-30.pth")
    l1_4_model.load_state_dict(ckpt["model_state_dict"])
    l1_4_model.eval()

    l2_model = models.PreActResNet18(n_cls=10, model_width=64, cifar_norm=True).cuda()
    l2_model.load_state_dict(torch.load("models/l2-at-eps=0.1-cifar10.pt")['last'])
    l2_model.eval()

    l1_1_acc_cln = clean_accuracy(l1_1_model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    l1_2_acc_cln = clean_accuracy(l1_2_model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    l1_3_acc_cln = clean_accuracy(l1_3_model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    l1_4_acc_cln = clean_accuracy(l1_4_model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    fw_acc_cln = clean_accuracy(fw_model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    l2_acc_cln = clean_accuracy(l2_model, x_clean.cuda(), y_clean.cuda())
    std_acc_cln = clean_accuracy(std_model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    print("Clean | \t Wass 0.01: {:.4f} \t Wass 0.005: {:.4f} \t Wass 0.002: {:.4f} \t Wass 0.0008: {:.4f} \t FrankWolfe: {:.4f} \t L2: {:.4f} \t Standard: {:.4f}".format(l1_1_acc_cln, l1_2_acc_cln, l1_3_acc_cln, l1_4_acc_cln, fw_acc_cln, l2_acc_cln, std_acc_cln))

    for corr in corruptions:
        for i in range(1,6):
            x_corr, y_corr = load_cifar10c(n_examples=args.n_samples, severity=i, corruptions=(corr,), data_dir=args.data_dir)

            l1_1_acc = clean_accuracy(l1_1_model, ((x_corr.cuda() - mu) / std), y_corr.cuda())
            l1_2_acc = clean_accuracy(l1_2_model, ((x_corr.cuda() - mu) / std), y_corr.cuda())
            l1_3_acc = clean_accuracy(l1_3_model, ((x_corr.cuda() - mu) / std), y_corr.cuda())
            l1_4_acc = clean_accuracy(l1_4_model, ((x_corr.cuda() - mu) / std), y_corr.cuda())
            fw_acc = clean_accuracy(fw_model, ((x_corr.cuda() - mu) / std), y_corr.cuda())
            l2_acc = clean_accuracy(l2_model, x_corr.cuda(), y_corr.cuda())
            std_acc = clean_accuracy(std_model, ((x_corr.cuda() - mu) / std), y_corr.cuda())
            print("{} {} | \t Wass 0.01: {:.4f} ({:.4f}) \t Wass 0.005: {:.4f} ({:.4f}) \t Wass 0.002: {:.4f} ({:.4f}) \t Wass 0.0008: {:.4f} ({:.4f}) \t FrankWolfe: {:.4f} ({:.4f}) \t L2: {:.4f} ({:.4f}) \t Standard: {:.4f} ({:.4f})".format(corr.ljust(20), i, l1_1_acc, l1_1_acc_cln - l1_1_acc, l1_2_acc, l1_2_acc_cln - l1_2_acc, l1_3_acc, l1_3_acc_cln - l1_3_acc, l1_4_acc, l1_4_acc_cln - l1_4_acc, fw_acc, fw_acc_cln - fw_acc, l2_acc, l2_acc_cln - l2_acc, std_acc, std_acc_cln - std_acc))




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
