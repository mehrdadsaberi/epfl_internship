
from typing import Dict, List
import torch
import csv
import argparse
import setGPU
import time

from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
from perceptual_advex.attacks import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial training evaluation')

    add_dataset_model_arguments(parser, include_checkpoint=True)
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of GPUs to train on')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--per_example', action='store_true', default=False,
                        help='output per-example accuracy')
    parser.add_argument('--output', type=str, help='output CSV')



    
    # parser.add_argument('--nBlocks', nargs='+', type=int)
    # parser.add_argument('--nStrides', nargs='+', type=int)
    # parser.add_argument('--nChannels', nargs='+', type=int)

    args = parser.parse_args()

    dataset, model = get_dataset_model(args)
    train_loader, val_loader = dataset.make_loaders(1, args.batch_size, only_val=False)

    
    mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float, device="cuda").unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float, device="cuda").unsqueeze(-1).unsqueeze(-1)

    normalize = lambda x: (x - mu) / std
    unnormalize = lambda x: x * std + mu

    def create_normal(inp_model):
        class Normal_Model():
            def __init__(self, inp_model):
                self.model = inp_model
            def __call__(self, x):
                return self.model(normalize(x))
            def zero_grad(self):
                self.model.zero_grad()
            def eval(self):
                self.model.eval()
            def forward(self, x):
                return self.model(normalize(x))
        return Normal_Model(inp_model)


    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]

    # Parallelize
    if torch.cuda.is_available():
        device_ids = list(range(args.parallel))
        model = nn.DataParallel(model, device_ids)
        attacks = [nn.DataParallel(attack, device_ids) for attack in attacks]

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    
    for batch_index, (inputs, labels) in enumerate(train_loader): #val_loader):
        print(f'BATCH {batch_index:05d}')

        if (
            args.num_batches is not None and
            batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            st_time = time.time()
            adv_inputs = attack(inputs, labels)
            fnsh_time = time.time()
            with torch.no_grad():
                if args.arch == "resnet18":
                    adv_logits = model(normalize(adv_inputs))
                else :
                    adv_logits = model(adv_inputs)
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}', f', time = {fnsh_time - st_time}',
                  sep='\t')
            batches_correct[attack_name].append(batch_correct)
        exit()

    print('OVERALL')
    accuracies = []
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              sep='\t')
        accuracies.append(accuracy)

    with open(args.output, 'w') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow(attack_names)
        if args.per_example:
            for example_correct in zip(*[
                attacks_correct[attack_name] for attack_name in attack_names
            ]):
                out_csv.writerow(
                    [int(attack_correct.item()) for attack_correct
                     in example_correct])
        out_csv.writerow(accuracies)
