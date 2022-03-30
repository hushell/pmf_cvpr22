import argparse


class DataConfig(object):
    """Common configuration options for creating data processing pipelines."""
    def __init__(
            self,
            args: argparse.Namespace
    ):
        """Initialize a DataConfig.
        """

        # General info
        self.path = args.data_path
        #self.batch_size = args.batch_size
        #self.num_workers = args.num_workers
        self.shuffle = args.shuffle

        # Transforms and augmentations
        self.image_size = args.image_size
        self.test_transforms = args.test_transforms
        self.train_transforms = args.train_transforms


class EpisodeDescriptionConfig(object):
    """Configuration options for episode characteristics."""
    def __init__(self,
                 args: argparse.Namespace):
        """Initialize a EpisodeDescriptionConfig.

        This is used in sampling.py in Trainer and in EpisodeDescriptionSampler to
        determine the parameters of episode creation relating to the ways and shots.

        Args:
            num_ways: Integer, fixes the number of classes ("ways") to be used in each
                episode. None leads to variable way.
            num_support: An integer, a tuple of two integers, or None. In the first
                case, the number of examples per class in the support set. In the
                second case, the range from which to sample the number of examples per
                class in the support set. Both of these cases would yield class-balanced
                episodes, i.e. all classes have the same number of support examples.
                Finally, if None, the number of support examples will vary both within
                each episode (introducing class imbalance) and across episodes.
            num_query: Integer, fixes the number of examples for each class in the
                query set.
            min_ways: Integer, the minimum value when sampling ways.
            max_ways_upper_bound: Integer, the maximum value when sampling ways. Note
                that the number of available classes acts as another upper bound.
            max_num_query: Integer, the maximum number of query examples per class.
            max_support_set_size: Integer, the maximum size for the support set.
            max_support_size_contrib_per_class: Integer, the maximum contribution for
                any given class to the support set size.
            min_log_weight: Float, the minimum log-weight to give to any particular
                class when determining the number of support examples per class.
            max_log_weight: Float, the maximum log-weight to give to any particular
                class.
            ignore_dag_ontology: Whether to ignore ImageNet's DAG ontology when
                sampling classes from it. This has no effect if ImageNet is not part of
                the benchmark.
            ignore_bilevel_ontology: Whether to ignore Omniglot's DAG ontology when
                sampling classes from it. This has no effect if Omniglot is not part of
                the benchmark.
            ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
                the sampler ignore the hierarchy for this proportion of episodes and
                instead sample categories uniformly.
            simclr_episode_fraction: Float, fraction of episodes that will be
                converted to SimCLR Episodes as described in the CrossTransformers
                paper.
            min_examples_in_class: An integer, the minimum number of examples that a
                class has to contain to be considered. All classes with fewer examples
                will be ignored. 0 means no classes are ignored, so having classes with
                no examples may trigger errors later. For variable shots, a value of 2
                makes it sure that there are at least one support and one query samples.
                For fixed shots, you could set it to `num_support + num_query`.

        Raises:
            RuntimeError: if incompatible arguments are passed.
        """
        arg_groups = {
                'num_ways': (args.num_ways, ('min_ways', 'max_ways_upper_bound'), (args.min_ways, args.max_ways_upper_bound)),
                'num_query': (args.num_query, ('max_num_query',), (args.max_num_query,)),
                'num_support':
                        (args.num_support,  # noqa: E131
                        ('max_support_set_size', 'max_support_size_contrib_per_class',  # noqa: E128
                         'min_log_weight', 'max_log_weight'),
                        (args.max_support_set_size, args.max_support_size_contrib_per_class,  # noqa: E128
                         args.min_log_weight, args.max_log_weight)),
        }

        for first_arg_name, values in arg_groups.items():
            first_arg, required_arg_names, required_args = values
            if ((first_arg is None) and any(arg is None for arg in required_args)):
                # Get name of the nones
                none_arg_names = [
                        name for var, name in zip(required_args, required_arg_names)
                        if var is None
                ]
                raise RuntimeError(
                        'The following arguments: %s can not be None, since %s is None. '
                        'Arguments can be set up with gin, for instance by providing '
                        '`--gin_file=learn/gin/setups/data_config.gin` or calling '
                        '`gin.parse_config_file(...)` in the code. Please ensure the '
                        'following gin arguments of EpisodeDescriptionConfig are set: '
                        '%s' % (none_arg_names, first_arg_name, none_arg_names))

        self.num_ways = args.num_ways
        self.num_support = args.num_support
        self.num_query = args.num_query
        self.min_ways = args.min_ways
        self.max_ways_upper_bound = args.max_ways_upper_bound
        self.max_num_query = args.max_num_query
        self.max_support_set_size = args.max_support_set_size
        self.max_support_size_contrib_per_class = args.max_support_size_contrib_per_class
        self.min_log_weight = args.min_log_weight
        self.max_log_weight = args.max_log_weight
        self.ignore_dag_ontology = args.ignore_dag_ontology
        self.ignore_bilevel_ontology = args.ignore_bilevel_ontology
        self.ignore_hierarchy_probability = args.ignore_hierarchy_probability
        self.min_examples_in_class = args.min_examples_in_class
        self.num_episodes = args.nEpisode

    @property
    def max_ways(self):
        """Returns the way (maximum way if variable) of the episode."""
        return self.num_ways or self.max_ways_upper_bound
