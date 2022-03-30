import os
import h5py
from PIL import Image

import torch


class MetaValDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, num_episodes=4000):
        super().__init__()

        self.num_episodes = num_episodes
        self.h5_path = h5_path
        self.h5_file = None

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        record = self.h5_file[str(idx)]
        support_images = record['sx'][()]
        support_labels = record['sy'][()]
        query_images = record['x'][()]
        query_labels = record['y'][()]

        return support_images, support_labels, query_images, query_labels


if __name__ == '__main__':
    dset = MetaValDataset('../../tf_records/ilsvrc_2012_v2/val_episodes4000.h5')

    data_loader_val = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    sx, sy, x, y = next(iter(data_loader_val))
    print(sx.shape)
    print(x.shape)
