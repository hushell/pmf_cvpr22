import os
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from tabulate import tabulate

from engine import evaluate
import utils.deit_util as utils
from utils.args import get_args_parser
from models import get_model
from datasets import get_bscd_loader


def main(args):
    args.distributed = False # CDFSL dataloader doesn't support DDP
    args.eval = True

    print(args)
    device = torch.device(args.device)

    cudnn.benchmark = True

    ##############################################
    # Model
    print(f"Creating model: {args.deploy} {args.arch}")

    model = get_model(args)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f'Load ckpt from {args.resume}')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    ##############################################
    # Test
    criterion = torch.nn.CrossEntropyLoss()
    #datasets = ["EuroSAT", "ISIC", "CropDisease", "ChestX"]
    datasets = args.cdfsl_domains
    var_accs = {}

    for domain in datasets:
        print(f'\n# Testing {domain} starts...\n')

        data_loader_val = get_bscd_loader(domain, args.test_n_way, args.n_shot, args.image_size)

        # validate lr
        best_lr = args.ada_lr
        if args.deploy == 'finetune':
            print("Start selecting the best lr...")
            best_acc = 0
            for lr in [0, 0.0001, 0.0005, 0.001]:
                model.lr = lr
                test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5)
                acc = test_stats['acc1']
                print(f"*lr = {lr}: acc1 = {acc}")
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
            model.lr = best_lr
            print(f"### Selected lr = {best_lr}")

        # final classification
        data_loader_val.generator.manual_seed(args.seed + 10000)
        test_stats = evaluate(data_loader_val, model, criterion, device)
        var_accs[domain] = (test_stats['acc1'], test_stats['acc_std'], best_lr)

        print(f"{domain}: acc1 on {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir and utils.is_main_process():
            test_stats['domain'] = domain
            test_stats['lr'] = best_lr
            with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")

    # print results as a table
    if utils.is_main_process():
        rows = []
        for dataset_name in datasets:
            row = [dataset_name]
            acc, std, lr = var_accs[dataset_name]
            conf = (1.96 * std) / np.sqrt(len(data_loader_val.dataset))
            row.append(f"{acc:0.2f} +- {conf:0.2f}")
            row.append(f"{lr}")
            rows.append(row)
        np.save(os.path.join(output_dir, f'test_results_{args.deploy}_{args.train_tag}.npy'), {'rows': rows})

        table = tabulate(rows, headers=['Domain', args.arch, 'lr'], floatfmt=".2f")
        print(table)
        print("\n")

        if args.output_dir:
            with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
                f.write(table)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.train_tag = 'pt' if args.resume == '' else 'ep'
    args.train_tag += f'_step{args.ada_steps}_lr{args.ada_lr}_prob{args.aug_prob}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import sys
    with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
        f.write(" ".join(sys.argv) + "\n")

    main(args)
