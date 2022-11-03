import os
import datetime
import numpy as np
import time
import torch
import h5py as h5
import sys

from tqdm import tqdm
from pathlib import Path

import utils.deit_util as utils

from datasets.meta_dataset import reader
#from datasets.meta_dataset.utils import parse_record
import cv2
import torchvision.transforms as transforms
from PIL import Image

from datasets.meta_dataset.utils import Split
from datasets.meta_dataset import dataset_spec as dataset_spec_lib


def convert_class_datasets(dataset_spec_list):
    for i in range(len(dataset_spec_list)):

        # each class of every source/dataset is stored in a tfrecord -> TFRecordDataset
        class_datasets = []

        episode_reader = reader.Reader(dataset_spec=dataset_spec_list[i],
                                       split=Split.TRAIN,
                                       shuffle=True)
        class_datasets += episode_reader.construct_class_datasets()

        episode_reader = reader.Reader(dataset_spec=dataset_spec_list[i],
                                       split=Split.VALID,
                                       shuffle=True)
        class_datasets += episode_reader.construct_class_datasets()

        episode_reader = reader.Reader(dataset_spec=dataset_spec_list[i],
                                       split=Split.TEST,
                                       shuffle=True)
        class_datasets += episode_reader.construct_class_datasets()

        print(f'==> Converting {dataset_spec_list[i].name} of {len(class_datasets)} classes.')

        if dataset_spec_list[i].name == 'quickdraw':
            resizer = transforms.Resize((28, 28))
        else:
            resizer = transforms.Resize((224, 224))

        # h5 writing
        #h5_file = dataset_spec_list[i].path + '.h5'
        for cls_dset in class_datasets:
            h5_file = cls_dset.data_path.replace('tfrecords', 'h5')
            print(f'* Writing {h5_file}:')
            with h5.File(h5_file, 'w') as f:
                for j, feat_dic in enumerate(tqdm(cls_dset)):
                    grp = f.create_group(f'{j}')
                    image = feat_dic["image"]
                    image = cv2.imdecode(image, -1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # from BGR to RGB

                    # due to storage limit, we store resized images if original is larger than 224
                    # for QuickDraw, we use 28x28 as this already yields an h5 file with 218GB
                    if (image.shape[0] >= 224 and image.shape[1] >= 224) or dataset_spec_list[i].name == 'quickdraw':
                        image = Image.fromarray(image)
                        image = resizer(image)
                        image = np.array(image)

                    grp.create_dataset('image', data=image)
                    grp.create_dataset('label', data=np.array(feat_dic['label']))
                    grp.create_dataset('id', data=np.array(feat_dic['id']))
                    #if j % 10 == 0:
                    #    print(f'Ep{j}: img.shape={image.shape}')


def main(data_path='/path/to/meta-dataset/tf_records'):
    ###########################################################################
    # EDIT here if you don't want to convert all domains
    domains = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower', 'traffic_sign', 'mscoco']
    ###########################################################################

    # Conversion
    all_dataset_specs = []
    for dname in domains:
        dataset_records_path = os.path.join(data_path, dname)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    convert_class_datasets(all_dataset_specs)


if __name__ == '__main__':
    main(sys.argv[1])
