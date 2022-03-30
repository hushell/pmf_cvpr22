import torchvision
from . import reader
from . import sampling
import torch
from .transform import get_transforms
import numpy as np
from .utils import Split, cycle_, parse_record
from typing import List, Union, Optional
from .dataset_spec import HierarchicalDatasetSpecification as HDS
from .dataset_spec import BiLevelDatasetSpecification as BDS
from .dataset_spec import DatasetSpecification as DS
from .config import EpisodeDescriptionConfig, DataConfig
from .tfrecord.torch.dataset import TFRecordDataset
from .sampling import EpisodeDescriptionSampler


def make_episode_pipeline(dataset_spec_list: List[Union[HDS, BDS, DS]],
                          split: Split,
                          episode_descr_config: EpisodeDescriptionConfig,
                          data_config: DataConfig,
                          ignore_hierarchy_probability: int = 0.0,
                          **kwargs):
    """Returns a pipeline emitting data from potentially multiples source as Episodes.

    Args:
      dataset_spec_list: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
      episode_descr_config: An instance of EpisodeDescriptionConfig containing
        parameters relating to sampling shots and ways for episodes.
      ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
        the sampler ignore the hierarchy for this proportion of episodes and
        instead sample categories uniformly.

    Returns:
    """

    episodic_dataset_list = []
    for i in range(len(dataset_spec_list)):
        episode_reader = reader.Reader(dataset_spec=dataset_spec_list[i],
                                       split=split,
                                       shuffle=data_config.shuffle)

        # each class of every source/dataset is stored in a tfrecord -> TFRecordDataset
        class_datasets = episode_reader.construct_class_datasets()

        sampler = sampling.EpisodeDescriptionSampler(
            dataset_spec=episode_reader.dataset_spec,
            split=split,
            episode_descr_config=episode_descr_config,
            use_dag_hierarchy=episode_descr_config.use_dag_ontology_list[i],
            use_bilevel_hierarchy=episode_descr_config.use_bilevel_ontology_list[i],
            ignore_hierarchy_probability=ignore_hierarchy_probability)

        transforms = get_transforms(data_config, split)

        _, max_support_size, max_query_size = sampler.compute_chunk_sizes()

        episodic_dataset_list.append(EpisodicDataset(class_datasets=class_datasets,
                                                     sampler=sampler,
                                                     transforms=transforms,
                                                     max_support_size=max_support_size,
                                                     max_query_size=max_query_size,
                                                     num_episodes=episode_descr_config.num_episodes
                                                    )
                                    )

    return ZipDataset(episodic_dataset_list)


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self,
                 class_datasets: List[TFRecordDataset],
                 sampler: EpisodeDescriptionSampler,
                 transforms: torchvision.transforms,
                 max_support_size: int,
                 max_query_size: int,
                 num_episodes: int = 2000):
        super().__init__()

        self.class_datasets = class_datasets
        self.sampler = sampler
        self.transforms = transforms
        self.max_query_size = max_query_size
        self.max_support_size = max_support_size
        self.num_episodes = num_episodes

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        episode_description = self.sampler.sample_episode_description()
        episode_classes = list({class_ for class_, _, _ in episode_description})
        #return episode_classes # DEBUG

        for class_id, nb_support, nb_query in episode_description:
            used_ids = []
            sup_added = 0
            query_added = 0

            # support
            while sup_added < nb_support:
                sample_dic = self.get_next(class_id)

                if sample_dic['id'] not in used_ids:
                    sample_dic = parse_record(sample_dic)
                    used_ids.append(sample_dic['id'])

                    support_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                    sup_added += 1

            # query
            while query_added < nb_query:
                sample_dic = self.get_next(class_id)

                if sample_dic['id'] not in used_ids:
                    sample_dic = parse_record(sample_dic)

                    used_ids.append(sample_dic['id'])
                    query_images.append(self.transforms(sample_dic['image']).unsqueeze(0))

                    query_added += 1

            # print(f"Class {class_id} contains duplicate: {contains_duplicates(used_ids)}")
            support_labels.extend([episode_classes.index(class_id)] * nb_support)
            query_labels.extend([episode_classes.index(class_id)] * nb_query)

        support_images = torch.cat(support_images, 0)
        query_images = torch.cat(query_images, 0)

        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels

    def get_next(self, class_id):
        try:
            sample_dic = next(self.class_datasets[class_id])
        except (StopIteration, TypeError) as e:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])
        return sample_dic


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list: List[EpisodicDataset]):
        self.dataset_list = dataset_list

    def __len__(self):
        return sum([ds.__len__() for ds in self.dataset_list])

    def __getitem__(self, idx):
        rand_source = np.random.randint(len(self.dataset_list))
        return self.dataset_list[rand_source].__getitem__(idx)
