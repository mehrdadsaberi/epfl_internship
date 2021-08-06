
from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import zoom
from tqdm import tqdm
import lpips
import os
from .wass_pot import Wasserstein_POT
from .wass_sinkhorn import Wasserstein_Sinkhorn
import matplotlib.pyplot as plt


class Trainer():
    def name(self):
        return self.model_name

    def initialize(self, model='lpips', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False, 
            is_train=False, lr=.0001, beta1=0.5, version='0.1', gpu_ids=[0]):
        '''
        INPUTS
            model - ['lpips'] for linearly calibrated network
                    ['baseline'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]'%(model,net)

        if(self.model == 'lpips'): # pretrained net + linear layer
            self.net = lpips.LPIPS(pretrained=not is_train, net=net, version=version, lpips=True, spatial=spatial, 
                pnet_rand=pnet_rand, pnet_tune=pnet_tune, 
                use_dropout=True, model_path=model_path, eval_mode=False)
        elif(self.model=='baseline'): # pretrained network
            self.net = lpips.LPIPS(pnet_rand=pnet_rand, net=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = lpips.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['Wass-POT','wass-pot']):
            self.net = Wasserstein_POT(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'Wass-POT'
        elif(self.model in ['Wass-Sink','wass-sink']):
            self.net = Wasserstein_Sinkhorn(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'Wass-Sink'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = lpips.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = lpips.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)
        self.var_p1 = Variable(self.input_p1,requires_grad=True)

    def forward_train(self): # run forward pass
        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0,self.d1,self.input_judge)

        self.var_judge = Variable(1.*self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge*2.-1.)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                            ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = lpips.tensor2im(self.var_ref.data)
        p0_img = lpips.tensor2im(self.var_p0.data)
        p1_img = lpips.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr


    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')


def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []
    
    #plt.rcParams['font.size'] = 10
    # plt.rcParams["figure.figsize"] = (30,10)

    # dir = "wass_images/{}-tmp".format(name.replace('/', '-'))
    # try:
    #     os.makedirs(dir)
    # except FileExistsError:
    #     pass

    i = 0
    for data in tqdm(data_loader.load_data(), desc=name):
        d0 = func(data['ref'],data['p0']).data.cpu().numpy().flatten()
        d1 = func(data['ref'],data['p1']).data.cpu().numpy().flatten()
        gt = data['judge'].cpu().numpy().flatten()
        d0s+=d0.tolist()
        d1s+=d1.tolist()
        gts+=gt.tolist()
        nd0s = np.array(d0s)
        nd1s = np.array(d1s)
        ngts = np.array(gts)

        #scores = (nd0s<nd1s)*(1.-ngts) + (nd1s<nd0s)*ngts + (nd1s==nd0s)*.5
        #print(i, "|", np.mean(scores))
        
        # img_ref = data['ref'][0].cpu().numpy() * 0.5 + 0.5
        # img_ref = np.swapaxes(img_ref, 0, 2) 
        # img_0 = data['p0'][0].cpu().numpy() * 0.5 + 0.5
        # img_0 = np.swapaxes(img_0, 0, 2)
        # img_1 = data['p1'][0].cpu().numpy() * 0.5 + 0.5
        # img_1 = np.swapaxes(img_1, 0, 2)
        # score = np.mean((d0<d1)*(1.-gt) + (d1<d0)*gt + (d1==d0)*.5)

        
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3)

        # ax1.imshow(img_ref)
        # ax1.axis('off')
        # ax1.set_title("Ref | score: {:.3f}".format(score))
        
        # ax2.imshow(img_0)
        # ax2.axis('off')
        # ax2.set_title("0 | wass: {:.3f}, l2: {:.3f}".format(d0s[-1], np.sum((img_ref - img_0) * (img_ref - img_0))))
        
        # ax3.imshow(img_1)
        # ax3.axis('off')
        # ax3.set_title("1 | wass: {:.3f}, l2: {:.3f}".format(d1s[-1], np.sum((img_ref - img_1) * (img_ref - img_1))))

        # plt.savefig(dir + "/{:03d}.png".format(i))
        # #plt.show()
        # plt.close()

        
        #print("{} |  d0: {:.3f}, d1: {:.3f}, score: {}".format(i, d0s[-1], d1s[-1], score))
        #input()
        
        i += 1

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds+=func(data['p0'],data['p1']).data.cpu().numpy().tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = lpips.voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))
