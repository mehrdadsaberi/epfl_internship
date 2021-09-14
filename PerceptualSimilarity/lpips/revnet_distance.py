import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import ot
import time

import lpips

import matplotlib.pyplot as plt

from models.iRevNet import iRevNet


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace

class Revnet_Distance(FakeNet):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(Revnet_Distance, self).__init__(use_gpu, colorspace)
        
        checkpoint = torch.load("nets/i-revnet-25-bij.t7")
        self.model = checkpoint['model'].module


    def forward(self, x1, x2, retPerLayer=None):
        x1_features = self.model.features(x1)[0]
        x2_features = self.model.features(x2)[0]
        
        x1_features_norm = torch.sqrt(torch.sum(x1_features ** 2, dim=1, keepdim=True)) + 1e-10
        x1_features = x1_features / (x1_features_norm * 
                            np.sqrt(x1_features.size()[2] * x1_features.size()[3]))
        
        x2_features_norm = torch.sqrt(torch.sum(x2_features ** 2, dim=1, keepdim=True)) + 1e-10
        x2_features = x2_features / (x2_features_norm * 
                            np.sqrt(x2_features.size()[2] * x2_features.size()[3]))

        return (x1_features - x2_features).norm(p=2, dim=(1,2,3))


