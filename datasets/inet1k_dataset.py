import os
import h5py
from PIL import Image
import json

import torch
from .meta_dataset import config as config_lib
from .meta_dataset import sampling
from .meta_dataset.utils import Split
from .meta_dataset.transform import get_transforms
from .meta_dataset import dataset_spec as dataset_spec_lib


class INet1kFewshotDataset(torch.utils.data.Dataset):
    def __init__(self, args, split=Split['TRAIN']):
        super().__init__()

        data_config = config_lib.DataConfig(args)
        episod_config = config_lib.EpisodeDescriptionConfig(args)

        episod_config.use_bilevel_ontology = False
        episod_config.use_dag_ontology = True

        if split == Split.TRAIN:
            tag = 'train'
        elif split == Split.VALID:
            tag = 'val' # NOTE: val-set has only 50 images per class -- too small
            episod_config.num_episodes = args.nValEpisode
        elif split == Split.TEST:
            tag = 'test'
            episod_config.num_episodes = 600
        else:
            raise ValueError(f'Split {split} is not valid.')

        # Read dataset_spec.json for MetaDataset V2
        dataset_records_path = os.path.join(data_config.path, 'ilsvrc_2012_v2')
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        self.num_classes = dataset_spec.get_classes_per_split()[split]
        print(f"=> There are {self.num_classes} classes in the {tag}-set.")

        #self.h5_path = os.path.join(dataset_records_path, f'imagenet-{tag}-256.h5')
        self.h5_path = os.path.join(dataset_records_path, 'imagenet-train-256.h5')
        self.h5_file = None

        classdict_path = os.path.join(dataset_records_path, f'imagenet-train-classdict.json')
        if os.path.exists(classdict_path):
            print(f'Load class_dict from {classdict_path}.')
            with open(classdict_path, 'r') as f:
                self.class_dict = json.load(f)
        else:
            print(f'Save class_dict to {classdict_path}.')
            # Make a class-wise dictionary recording images for each class
            self.class_dict = {str(i):[] for i in range(1000)}
            with h5py.File(self.h5_path, 'r') as file:
                for k, v in file.items():
                    y = v['target'][()]
                    self.class_dict[str(y)].append(k)
            with open(classdict_path, 'w') as f:
                json.dump(self.class_dict, f)

        self.sampler = sampling.EpisodeDescriptionSampler(
            dataset_spec=dataset_spec,
            split=split,
            episode_descr_config=episod_config,
            use_dag_hierarchy=True,
            use_bilevel_hierarchy=False,
            ignore_hierarchy_probability=args.ignore_hierarchy_probability)

        self.transforms = get_transforms(data_config, split)

        self.len = episod_config.num_episodes

    def __len__(self):
        return self.len

    def get_next(self, class_id):
        class_id = str(class_id)
        which = torch.randint(high=len(self.class_dict[class_id]), size=(1,)).item()
        idx = self.class_dict[class_id][which] # idx is a str e.g. '0'

        if idx in self.used_ids:
            return None

        self.used_ids.append(idx)

        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        record = self.h5_file[idx]

        if self.transforms:
            x = Image.fromarray(record['data'][()])
            x = self.transforms(x)
        else:
            x = torch.from_numpy(record['data'][()])

        return x

    def __getitem__(self, idx):
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        episode_description = self.sampler.sample_episode_description()
        episode_description = tuple( # relative ids --> abs ids
            (class_id + self.sampler.class_set[0], num_support, num_query)
            for class_id, num_support, num_query in episode_description)
        episode_classes = list({class_ for class_, _, _ in episode_description})

        for class_id, nb_support, nb_query in episode_description:
            self.used_ids = []
            sup_added = 0
            query_added = 0

            # support
            time_budget = 0
            while sup_added < nb_support:
                #print('support fetch:', sup_added, class_id)
                x = self.get_next(class_id)
                if x is not None:
                    support_images.append(x)
                    sup_added += 1
                else:
                    time_budget += 1
                    assert time_budget < 100, f'Fetching support images from class {class_id} failed.'

            # query
            time_budget = 0
            while query_added < nb_query:
                x = self.get_next(class_id)
                if x is not None:
                    query_images.append(x)
                    query_added += 1
                else:
                    time_budget += 1
                    assert time_budget < 100, f'Fetching query images from class {class_id} failed.'

            # print(f"Class {class_id} contains duplicate: {contains_duplicates(used_ids)}")
            support_labels.extend([episode_classes.index(class_id)] * nb_support)
            query_labels.extend([episode_classes.index(class_id)] * nb_query)

        support_images = torch.stack(support_images, dim=0)
        query_images = torch.stack(query_images, dim=0)

        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels

