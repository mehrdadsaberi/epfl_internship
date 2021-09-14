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

import matplotlib.pyplot as plt

import ot.gromov

from models.iRevNet import iRevNet



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

#corruptions = ["elastic_transform"]

def main():
    args = get_args()
    device = "cuda"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str("6")
    x_clean, y_clean = load_cifar10(n_examples=args.n_samples, data_dir=args.data_dir)

    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

    model = ResNet18().cuda()
    ckpt = torch.load("nets/cifar_vanilla.pth")
    if "net" in ckpt.keys():
        for key in ckpt["net"].keys():
            assert "module" in key
        ckpt["net"] = dict((key[7:], value) for key, value in ckpt["net"].items())
        model.load_state_dict(ckpt["net"])
    model.eval()

    acc_cln = clean_accuracy(model, ((x_clean.cuda() - mu) / std), y_clean.cuda())
    print("Clean accuracy: {:.4f}".format(acc_cln))

    wass_func = Wasserstein_Sinkhorn(lam=0.1).forward

    
    plt.rcParams["figure.figsize"] = (30,10)

    
    checkpoint = torch.load("nets/i-revnet-25-bij.t7")
    lpips_model = checkpoint['model'].module

    def revnet_dist(x1, x2):
        x1_features = lpips_model.features(x1)[0]
        x2_features = lpips_model.features(x2)[0]
        
        x1_features_norm = torch.sqrt(torch.sum(x1_features ** 2, dim=1, keepdim=True)) + 1e-10
        x1_features = x1_features / (x1_features_norm * 
                            np.sqrt(x1_features.size()[2] * x1_features.size()[3]))
        
        x2_features_norm = torch.sqrt(torch.sum(x2_features ** 2, dim=1, keepdim=True)) + 1e-10
        x2_features = x2_features / (x2_features_norm * 
                            np.sqrt(x2_features.size()[2] * x2_features.size()[3]))

        return (x1_features - x2_features).norm(p=2, dim=(1,2,3))


    def pi_analyze(pi, dist, fname):
        try:
            os.mkdir("pi_analyze/")
        except:
            pass
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
        plt.title("Wass dist: {:.6f}, More than 10: {:02.4f}%".format(dist.mean().item(), 100 * thresh_val / dist.mean().item()))
        plt.savefig("pi_analyze/{}.png".format(fname))
        plt.close()

    for corr in corruptions:
        for i in range(1,6):
            x_corr, y_corr = load_cifar10c(n_examples=args.n_samples, severity=i, corruptions=(corr,), data_dir=args.data_dir)
            acc_corr = clean_accuracy(model, ((x_corr.cuda() - mu) / std), y_corr.cuda())

            dataset = CustomDataset(x_corr, x_clean, y_clean)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            
            # for ind in range(len(x_corr)):
            #     img_clean = x_clean[ind].swapaxes(0, 1).swapaxes(1, 2)
            #     img_corr = x_corr[ind].swapaxes(0, 1).swapaxes(1, 2)
            #     img_corr_scaled = img_corr / img_corr.sum(dim=(0, 1)) * img_clean.sum(dim=(0, 1))

            #     fig, (ax1, ax2, ax3) = plt.subplots(1,3)

            #     ax1.imshow(img_clean)
            #     ax1.axis('off')
            #     ax1.set_title("Clean Image")
                
            #     ax2.imshow(img_corr)
            #     ax2.axis('off')
            #     ax2.set_title("Corrupted Image, Wass Dist: {:.6f}".format(float(wass_func(x_clean[ind:ind+1], x_corr[ind:ind+1]))))
                
            #     ax3.imshow(img_corr_scaled)
            #     ax3.axis('off')
            #     ax3.set_title("Scaled Corrupted Image")

            #     plt.savefig("imgs/{:03d}-{}-{}.png".format(ind, corr, i))
            #     #plt.show()
            #     plt.close()



            mean_dist = 0
            # mean_l1 = 0
            # mean_diff = 0

            for data in dataloader:
                cln_x = data["Clean Image"].cuda()
                corr_x = data["Corrupted Image"].cuda()

                dist = revnet_dist(cln_x, corr_x)

                # (N,C,X,Y) = cln_x.size()
                # x1 = (cln_x * 0.5 + 0.5).view(N * C, X * Y).clamp(min=3e-4)
                # x1 /= x1.sum(dim=1, keepdim=True) + 1e-35
                # x2 = (corr_x * 0.5 + 0.5).view(N * C, X * Y).clamp(min=3e-4)
                # x2 /= x2.sum(dim=1, keepdim=True) + 1e-35
                # dist_l1 = (x1 - x2).abs().sum(dim=1).view(N, C).sum(dim=1)
                
                
                mean_dist += dist.sum()
                # mean_l1 += dist_l1.sum()
                # mean_diff += (dist - dist_l1).abs().sum()

            # mean_pi = np.zeros(1)

            # for data in dataloader:
            #     for c in range(3):
            #         cln_x = data["Clean Image"][0,c].view(-1).clamp(min=1e-5).numpy().astype(np.float64)
            #         cln_x /= np.sum(cln_x)
            #         corr_x = data["Corrupted Image"][0,c].view(-1).clamp(min=1e-5).numpy().astype(np.float64)
            #         corr_x /= np.sum(corr_x)

            #         M = torch.load("cmats/l2.pt").numpy()

            #         #dist = ot.gromov.gromov_wasserstein2(M, M, cln_x, corr_x, loss_fun="square_loss")
            #         #dist = ot.emd2(cln_x, corr_x, M)
            #         pi = ot.emd(cln_x, corr_x, M)
            #         dist = np.sum(M * pi)

            #         if mean_pi.shape != pi.shape:
            #             mean_pi = pi
            #         else :
            #             mean_pi += pi
                    
            #         #print(np.sum(cln_x), np.sum(corr_x), dist, np.sum(np.abs(cln_x - corr_x)))
            #         #dist = ((cln_x - corr_x) * (cln_x - corr_x)).sum(dim=(1,2,3))
            #         #dist = (cln_x - corr_x).abs().amax(dim=(1,2,3))
            #         #print(dist)
            #         #exit()
            #         mean_dist += dist
            
            # pi_analyze(mean_pi / (args.n_samples * 3), mean_dist / (args.n_samples * 3), "{}_{}".format(corr, i))

            # print("{} {} | \t Mean Wasserstein: {:.4f} \t Model Accuracy: {:.4f}".format(corr.ljust(20), i, mean_dist / (args.n_samples * 3), acc_corr))
            print("{} {} | \t Mean LPIPS: {:.4f} \t Model Accuracy: {:.4f}".format(corr.ljust(20), i, mean_dist / args.n_samples, acc_corr))


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
