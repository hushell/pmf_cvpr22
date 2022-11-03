import os
import datetime
import numpy as np
import time
import torch
import h5py as h5

from tqdm import tqdm
from pathlib import Path

import utils.deit_util as utils
from utils.args import get_args_parser

from datasets.meta_dataset import get_metadataset
from datasets.meta_dataset.utils import Split


def main(args):
    ###########################################################################
    # EDIT here
    #args.data_path = '/path/to/meta-dataset/tf_records'
    args.dataset = 'meta_dataset'
    args.image_size = 128
    args.max_ways_upper_bound = 50
    args.max_support_size_contrib_per_class = 100
    args.max_num_query = 10
    args.num_workers = 10

    if args.eval: # fixed meta-test (NOT used)
        args.nValEpisode = 600
        args.val_sources = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower', 'traffic_sign', 'mscoco']
    else:
        args.nValEpisode = 120
        args.val_sources = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
    ###########################################################################

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    ##############################################
    # Data loaders
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        #random.seed(worker_seed)

    for j, source in enumerate(args.val_sources):
        if args.eval:
            dataset_val = get_metadataset(args, [source], Split["TEST"])
            h5_file = f'{source}/test_ep{args.nValEpisode}_img{args.image_size}.h5'
        else:
            dataset_val = get_metadataset(args, [source], Split["VALID"])
            h5_file = f'{source}/val_ep{args.nValEpisode}_img{args.image_size}.h5'

        h5_file = os.path.join(args.data_path, h5_file)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000 + j)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator
        )

        ##############################################
        # h5 writing
        with h5.File(h5_file, 'w') as f:
            for i, batch in enumerate(data_loader_val):
                sx, sy, x, y = batch
                grp = f.create_group(f'{i}')
                grp.create_dataset('sx', data=sx[0])
                grp.create_dataset('sy', data=sy[0])
                grp.create_dataset('x', data=x[0])
                grp.create_dataset('y', data=y[0])

                if i % 10 == 0:
                    print(f'Ep{i}: support={sx[0].shape}, query={x[0].shape}')

        ##############################################
        # h5 reading
        with h5.File(h5_file, 'r') as f:
            for k, v in f.items():
                if int(k) % 10 == 0:
                    print('Ep'+k, v['sx'].shape, v['x'].shape)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
