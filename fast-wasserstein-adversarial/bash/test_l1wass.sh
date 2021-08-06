#!/bin/bash

#dataset=mnist
#ckpt=mnist_vanilla
# ckpt=mnist_adv_training
# ckpt=mnist_adv_training_attack-frank_eps-0.3
#batch=100
#num_batch=10
#eps=0.5
#nb_iter=300
#postprocess=0

dataset=cifar
#ckpt=cifar_vanilla
ckpt=cifar_adv_training
#ckpt=cifar_adv_training_attack-frank_eps-0.005
batch=100
num_batch=1
eps=0.005
alpha=0.01
nb_iter=40
postprocess=0

# dataset=imagenet
# ckpt=imagenet_resnet50
# batch=50
# num_batch=1
# eps=0.001
# nb_iter=300
# postprocess=1

# save_img_loc=./adversarial_examples/frank_wolfe.pt

python l1wass.py --dataset $dataset \
                      --checkpoint $ckpt \
                      --batch_size $batch \
                      --num_batch $num_batch \
                      --eps $eps \
                      --alpha $alpha \
                      --nb_iter $nb_iter \
                      --seed 0 \
                      --postprocess $postprocess \
                      --save_img_loc ./adversarial_examples/l1_wass.pt \
                      --save_info_loc None