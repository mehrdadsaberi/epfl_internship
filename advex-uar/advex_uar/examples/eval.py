import click
import importlib
import os

import json
import numpy as np
import torch

from advex_uar.eval import ImagenetEvaluator, ImagenetCEvaluator
from advex_uar.eval import CIFAR10Evaluator, CIFAR10CEvaluator
from advex_uar.common.pyt_common import *
from advex_uar.common import FlagHolder

from resnet import ResNet18

import setGPU

def get_ckpt(FLAGS):
    if FLAGS.ckpt_path is not None:
        print('Loading ckpt from {}'.format(FLAGS.ckpt_path))
        return torch.load(FLAGS.ckpt_path)
    elif FLAGS.use_wandb and FLAGS.wandb_run_id is not None:
        globals()['wandb'] = importlib.import_module('wandb')
        print('Loading ckpt from wandb run id {}'.format(FLAGS.wandb_run_id))
        api = wandb.Api()
        run = api.run("{}/{}/{}".format(
                FLAGS.wandb_username, FLAGS.wandb_ckpt_project, FLAGS.wandb_run_id))
        ckpt_file = run.file("ckpt.pth")
        ckpt_file.download(replace=False)
        os.rename('ckpt.pth', os.path.join(wandb.run.dir, 'ckpt.pth'))
        return torch.load(os.path.join(wandb.run.dir, 'ckpt.pth'))
    else:
        raise ValueError('You must specify a wandb_run_id or a ckpt_path.')
    
def run(**flag_kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**flag_kwargs)
    if FLAGS.wandb_ckpt_project is None:
        FLAGS._dict['wandb_ckpt_project'] = FLAGS.wandb_project
    if FLAGS.step_size is None:
        FLAGS.step_size = get_step_size(FLAGS.epsilon, FLAGS.n_iters, FLAGS.use_max_step)
        FLAGS._dict['step_size'] = FLAGS.step_size
    FLAGS.summary()

    logger = init_logger(FLAGS.use_wandb, 'eval', FLAGS._dict)

    if FLAGS.dataset in ['cifar-10', 'cifar-10-c']:
        nb_classes = 10
    else:
        nb_classes = 1000 // FLAGS.class_downsample_factor

    model_dataset = FLAGS.dataset
    if model_dataset == 'imagenet-c':
        model_dataset = 'imagenet'

    model = ResNet18().cuda()
    ckpt = torch.load(FLAGS.ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # model = get_model(model_dataset, FLAGS.resnet_size, nb_classes)
    # ckpt = get_ckpt(FLAGS)
    # model.load_state_dict(ckpt['model'])



    if FLAGS.dataset == 'imagenet':
        Evaluator = ImagenetEvaluator
    elif FLAGS.dataset == 'imagenet-c':
        Evaluator = ImagenetCEvaluator
    elif FLAGS.dataset == 'cifar-10':
        Evaluator = CIFAR10Evaluator
    elif FLAGS.dataset == 'cifar-10-c':
        Evaluator = CIFAR10CEvaluator

    out_f = open(FLAGS.ckpt_path + ".out", 'w')
    
    with open("analysis/calibrations/cifar-10/calibs.out", 'r') as f:
        atas = json.load(f)

    print("MODEL:", FLAGS.ckpt_path)

    sum_acc = {}
    sum_ata = {}

    for x in atas:
        attack_name = x[0][0]
        epsilon = x[0][1]
        n_iters = 50 # x[0][2]
        step_size = x[0][3]
        ata = x[1]

        attack = get_attack(FLAGS.dataset, attack_name, epsilon,
                        n_iters, step_size, False)
        
        evaluator = Evaluator(model=model, attack=attack, dataset=FLAGS.dataset,
                          dataset_path=FLAGS.dataset_path, nb_classes=nb_classes,
                          corruption_type=FLAGS.corruption_type, corruption_name=FLAGS.corruption_name,
                          corruption_level=FLAGS.corruption_level,
                          batch_size=FLAGS.batch_size, stride=FLAGS.class_downsample_factor,
                          fp_all_reduce=FLAGS.use_fp16, logger=logger, tag=FLAGS.tag)
        std_loss, std_acc, adv_loss, adv_acc = evaluator.evaluate()

        if attack_name not in sum_acc:
            sum_acc[attack_name] = 0
            sum_ata[attack_name] = 0
        sum_acc[attack_name] += adv_acc
        sum_ata[attack_name] += ata

        out_s = "{} {:.4f} | \t std: {:.4f} \t adv: {:.4f} \t ata: {:.4f}".format(attack_name, epsilon, std_acc, adv_acc, ata)
        out_f.write(out_s + "\n")
        out_f.flush()
        print(out_s)

    for attack_name in sum_acc.keys():
        out_s = "{} | uar: {:.4f}".format(attack_name, sum_acc[attack_name] / sum_ata[attack_name])
        out_f.write(out_s + "\n")
        out_f.flush()
        print(out_s)
    
    out_f.close()

@click.command()
# wandb options
@click.option("--use_wandb/--no_wandb", is_flag=True, default=True)
@click.option("--wandb_project", default=None, help="WandB project to log to")
@click.option("--tag", default='eval', help="Short tag for WandB")

# Dataset options
# Allowed values: ['imagenet', 'imagenet-c', 'cifar-10', 'cifar-10-c']
@click.option("--dataset", default='imagenet')
@click.option("--dataset_path", default=None)

# Model options
@click.option("--resnet_size", default=50)
@click.option("--class_downsample_factor", default=1, type=int)

# checkpoint options; if --ckpt_path is None, assumes that ckpt is pulled from WandB
@click.option("--ckpt_path", default=None, help="Path to the checkpoint for evaluation")
@click.option("--wandb_username", default=None, help="WandB username to pull ckpt from")
@click.option("--wandb_ckpt_project", default=None, help='WandB project to pull ckpt from')
@click.option("--wandb_run_id", default=None,
              help='If --use_wandb is set, WandB run_id to pull ckpt from.  Otherwise'\
              'a run_id which will be associated with --ckpt_path')

# Evaluation options
@click.option("--use_fp16/--no_fp16", is_flag=True, default=False)
@click.option("--batch_size", default=128)

# Options for ImageNet-C and CIFAR-10-C
@click.option("--corruption_type", default=None)
@click.option("--corruption_name", default=None)
@click.option("--corruption_level", default=None)

# Attack options
# Allowed values: ['pgd_linf', 'pgd_l2', 'fw_l1', 'jpeg_linf', 'jpeg_l2', 'jpeg_l1', 'elastic', 'fog', 'gabor', 'snow']
@click.option("--attack", default=None, type=str)
@click.option("--epsilon", default=16.0, type=float)
@click.option("--step_size", default=None, type=float)
@click.option("--use_max_step", is_flag=True, default=False)
@click.option("--n_iters", default=50, type=int)

def main(**flags):
    run(**flags)

if __name__ == '__main__':
    main()
