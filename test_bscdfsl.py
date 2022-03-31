import os
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from engine import evaluate
import utils.deit_util as utils
from utils.args import get_args_parser
from models import get_model
from datasets import get_bscd_loader


def main(args):
    args.distributed = False
    args.eval = True
    #utils.init_distributed_mode(args)

    print(args)
    device = torch.device(args.device)

    ## fix the seed for reproducibility
    #seed = args.seed + utils.get_rank()
    #args.seed = seed
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)

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

    print(f'Testing {args.bscd_dataset} starts')

    data_loader_val = get_bscd_loader(args.bscd_dataset, args.test_n_way, args.n_shot, args.image_size)

    test_stats = evaluate(data_loader_val, model, criterion, device)

    if args.output_dir:
        test_stats['domain'] = args.bscd_dataset
        test_stats['ada_lr'] = args.ada_lr
        test_stats['ada_steps'] = args.ada_steps
        with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
            f.write(json.dumps(test_stats) + "\n")

    acc, std = test_stats['acc1'], test_stats['acc_std']
    conf = (1.96 * std) / np.sqrt(len(data_loader_val.dataset))
    print(f"{args.bscd_dataset}: acc1 on {len(data_loader_val.dataset)} test images: {acc:0.2f} +- {conf:0.2f}")


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
