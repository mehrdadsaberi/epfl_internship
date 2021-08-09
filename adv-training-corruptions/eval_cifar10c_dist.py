import os
import argparse
import numpy as np
import torch
import models
import pandas as pd
import data
import setGPU

from robustbench.data import load_cifar10c, load_cifar10
from robustbench.utils import clean_accuracy
from torch.utils.data import Dataset, DataLoader

from resnet import ResNet18
from wass_sinkhorn import Wasserstein_Sinkhorn


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

class CustomDataset(Dataset):
    def __init__(self, cln_images, corr_images, labels):
        self.cln_images = cln_images
        self.corr_images = corr_images
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        cln_image = self.cln_images[idx]
        corr_image = self.corr_images[idx]
        sample = {"Clean Image": cln_image, "Corrupted Image": corr_image, "Label": label}
        return sample
    

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

    model = ResNet18().cuda()
    ckpt = torch.load("models/cifar_vanilla.pth")
    if "net" in ckpt.keys():
        for key in ckpt["net"].keys():
            assert "module" in key
        ckpt["net"] = dict((key[7:], value) for key, value in ckpt["net"].items())
        model.load_state_dict(ckpt["net"])
    model.eval()

    acc_cln = clean_accuracy(model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    print("Clean accuracy: {:.4f}".format(acc_cln))

    wass_func = Wasserstein_Sinkhorn(lam=0.1).forward

    for corr in corruptions:
        for i in range(1,6):
            x_corr, y_corr = load_cifar10c(n_examples=args.n_samples, severity=i, corruptions=(corr,), data_dir=args.data_dir)
            acc_corr = clean_accuracy(model, ((x_corr.cuda() - mu) / std), y_corr.cuda())

            dataset = CustomDataset(x_corr, x_clean, y_clean)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            
            mean_dist = 0

            for data in dataloader:
                cln_x = data["Clean Image"].cuda()
                corr_x = data["Corrupted Image"].cuda()

                dist = wass_func(cln_x, corr_x)
                #dist = ((cln_x - corr_x) * (cln_x - corr_x)).sum(dim=(1,2,3))
                #dist = (cln_x - corr_x).abs().amax(dim=(1,2,3))
                #print(dist)
                #exit()
                mean_dist += dist.sum()

            print("{} {} | \t Mean Wasserstein: {:.4f} \t Model Accuracy: {:.4f}".format(corr.ljust(20), i, mean_dist / args.n_samples, acc_corr))

                
                


    prnt = []
    for acc in mean_acc:
        prnt += ["{:.4f}".format(acc / cnt)]
    print()
    print_info("Average", prnt)
    


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
