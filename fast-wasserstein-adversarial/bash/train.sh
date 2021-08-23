#!/bin/bash

# dataset=mnist
# batch=128
# lr=0.01
# epoch=30

# attack=frank
# eps=0.5
# nb_iter=40

dataset=cifar
batch=128
lr=0.1
epoch=150

attack=frank
eps=0.5
nb_iter=30

resume=45
save_model_loc=checkpoints/cifar_frank_eps-0.5_epoch-45.pth

python train.py --dataset $dataset \
                --batch_size $batch \
                --lr $lr \
                --epoch $epoch \
                --attack $attack \
                --eps $eps \
                --nb_iter ${nb_iter} \
                --resume $resume \
                --save_model_loc ${save_model_loc}