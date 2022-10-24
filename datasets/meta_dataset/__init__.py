from .utils import Split
from . import config as config_lib
from . import dataset_spec as dataset_spec_lib
from . import pipeline
from .utils import worker_init_fn_


def get_metadataset(args, datasets=["ilsvrc_2012"], split=Split["TRAIN"]):
    # Recovering configurations
    data_config = config_lib.DataConfig(args)
    episod_config = config_lib.EpisodeDescriptionConfig(args)

    if split == Split["VALID"] or split == Split["TEST"]:
        episod_config.num_episodes = args.nValEpisode

    # Get the data specifications
    use_dag_ontology_list = [False]*len(datasets)
    use_bilevel_ontology_list = [False]*len(datasets)
    if episod_config.num_ways:
        if len(datasets) > 1:
            raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
    else:
        # Enable ontology aware sampling for Omniglot and ImageNet.
        if 'omniglot' in datasets:
            use_bilevel_ontology_list[datasets.index('omniglot')] = True
        if 'ilsvrc_2012' in datasets:
            use_dag_ontology_list[datasets.index('ilsvrc_2012')] = True

    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list

    all_dataset_specs = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                             split=split,
                                             data_config=data_config,
                                             episode_descr_config=episod_config)

    #  If you want to get the total number of classes (i.e from combined datasets)
    num_classes = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])
    print(f"=> There are {num_classes} classes in the {split} split of the combined datasets")

    return dataset
