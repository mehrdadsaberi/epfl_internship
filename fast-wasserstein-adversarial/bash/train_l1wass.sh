#!/bin/bash

#dataset=mnist
#batch=128
#lr=0.01
#epoch=30

#attack=frank
#eps=0.5
#nb_iter=40

dataset=cifar
batch=128
lr=0.2
epoch=30

attack=l1wass
eps=0.0008
alpha=0.01
nb_iter=40

resume=10
save_model_loc=cifar_adv_training_attack-l1wass_eps-0.0008_epoch-10

python train.py --dataset $dataset \
                --batch_size $batch \
                --lr $lr \
                --epoch $epoch \
                --attack $attack \
                --eps $eps \
                --nb_iter ${nb_iter} \
                --alpha $alpha \
                --resume $resume \
                --save_model_loc $save_model_loc