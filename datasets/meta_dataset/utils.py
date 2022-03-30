import enum
import torch
import numpy as np
import cv2
from PIL import Image
import torch.distributed as dist


def worker_init_fn_(worker_id, seed):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    rank = dist.get_rank()
    random_gen = np.random.RandomState(seed + worker_id + rank)
    dataset.random_gen = random_gen
    for d in dataset.dataset_list:
        d.random_gen = random_gen


def cycle_(iterable):
    # Creating custom cycle since itertools.cycle attempts to save all outputs in order to
    # re-cycle through them, creating amazing memory leak
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class Split(enum.Enum):
    """The possible data splits."""
    TRAIN = 0
    VALID = 1
    TEST = 2


def parse_record(feat_dic):
    # typename_mapping = {
    #     "byte": "bytes_list",
    #     "float": "float_list",
    #     "int": "int64_list"
    # }
    # get BGR image from bytes
    image = cv2.imdecode(feat_dic["image"], -1)
    # from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    feat_dic["image"] = image
    return feat_dic


