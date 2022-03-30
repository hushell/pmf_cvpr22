import numpy as np
import torchvision.transforms as transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def build_transform(is_train, input_size=224):
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def dataset_setting(nSupport, img_size=32):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    trainTransform = build_transform(True, img_size)
    valTransform = build_transform(False, img_size)

    inputW, inputH, nbCls = img_size, img_size, 64

    trainDir = './data/cifar-fs/train/'
    valDir = './data/cifar-fs/val/'
    testDir = './data/cifar-fs/test/'
    episodeJson = './data/cifar-fs/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
            else './data/cifar-fs/val1000Episode_5_way_5_shot.json'

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
