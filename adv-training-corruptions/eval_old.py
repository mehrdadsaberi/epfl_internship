import os
import argparse
import numpy as np
import torch
import models
import pandas as pd
import data
import setGPU
from tqdm import tqdm

from robustbench.data import load_cifar10c, load_cifar10
from robustbench.utils import clean_accuracy
from torch.utils.data import Dataset, DataLoader

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
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--output', default='output.csv', type=str)
    parser.add_argument('--only_clean', action='store_true')
    parser.add_argument('--n_samples', default=1000, type=int)

    return parser.parse_args()

corruptions = ['shot_noise', 'motion_blur', 'snow', 'pixelate', 'gaussian_noise', 'defocus_blur',
               'brightness', 'fog', 'zoom_blur', 'frost', 'glass_blur', 'impulse_noise', 'contrast',
               'jpeg_compression', 'elastic_transform']


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
    

def main():
    args = get_args()
    torch.manual_seed(11)
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    cln_X, cln_y = load_cifar10(n_examples=args.n_samples, data_dir=args.data_dir)

    wass_func = Wasserstein_Sinkhorn(lam=0.1).forward

    model = models.PreActResNet18(n_cls=10, model_width=64, cifar_norm=True)
    if args.use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load(args.checkpoint)['last'])
    model.eval()

    clean_acc = clean_accuracy(model, cln_X.cuda() if args.use_gpu else cln_X, cln_y.cuda() if args.use_gpu else cln_y)
    print("Clean accuracy: {:.4f}".format(clean_acc))

    for corr in corruptions:
        for i in range(1, 6):
            X, y = load_cifar10c(n_examples=args.n_samples, data_dir=args.data_dir, severity=i, corruptions=(corr,))

            dataset = CustomDataset(X, cln_X, y)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

            acc = clean_accuracy(model, X.cuda() if args.use_gpu else X, y.cuda() if args.use_gpu else y)
            
            mean_dist = 0

            for data in dataloader:
                cln_x = data["Clean Image"]
                corr_x = data["Corrupted Image"]

                if args.use_gpu:
                    cln_x = cln_x.cuda()
                    corr_x = corr_x.cuda()

                #dist = wass_func(cln_x, corr_x)
                #dist = ((cln_x - corr_x) * (cln_x - corr_x)).sum(dim=(1,2,3))
                dist = (cln_x - corr_x).abs().amax(dim=(1,2,3))
                #print(dist)
                #exit()
                mean_dist += dist.sum()

            print("{} {} | \t Mean Wasserstein: {:.4f} \t Model Accuracy: {:.4f}".format(corr.ljust(20), i, mean_dist / args.n_samples, acc))

                
                






    # clean_acc = clean_accuracy(model, x_clean.cuda(), y_clean.cuda())
    # print("Clean accuracy: ", clean_acc)

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
