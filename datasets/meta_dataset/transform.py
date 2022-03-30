import torch
import torchvision.transforms as transforms
from PIL import ImageEnhance

from .utils import Split
from .config import DataConfig

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)


class ImageJitter(object):
    def __init__(self, transformdict):
        transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                                 Contrast=ImageEnhance.Contrast,
                                 Sharpness=ImageEnhance.Sharpness,
                                 Color=ImageEnhance.Color)
        self.params = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.params))

        for i, (transformer, alpha) in enumerate(self.params):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def get_transforms(data_config: DataConfig,
                   split: Split):
    if split == Split["TRAIN"]:
        return train_transform(data_config)
    else:
        return test_transform(data_config)


def test_transform(data_config: DataConfig):
    resize_size = int(data_config.image_size * 256 / 224)
    assert resize_size == data_config.image_size * 256 // 224
    # resize_size = data_config.image_size

    transf_dict = {'resize': transforms.Resize(resize_size),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = data_config.test_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])


def train_transform(data_config: DataConfig):
    transf_dict = {'resize': transforms.Resize(data_config.image_size),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'random_resized_crop': transforms.RandomResizedCrop(data_config.image_size),
                   'jitter': ImageJitter(jitter_param),
                   'random_flip': transforms.RandomHorizontalFlip(),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = data_config.train_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])
