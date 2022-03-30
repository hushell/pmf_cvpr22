import numpy as np
import torchvision.transforms as transforms

def dataset_setting(nSupport, img_size=80):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([#transforms.RandomCrop(img_size, padding=8),
                                         transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                         transforms.RandomHorizontalFlip(),
                                         #lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    valTransform = transforms.Compose([#transforms.CenterCrop(80),
                                       transforms.Resize((img_size, img_size)),
                                       #lambda x: np.asarray(x),
                                       transforms.ToTensor(),
                                       normalize])

    inputW, inputH, nbCls = img_size, img_size, 64

    trainDir = './data/Mini-ImageNet/train/'
    valDir = './data/Mini-ImageNet/val/'
    testDir = './data/Mini-ImageNet/test/'
    episodeJson = './data/Mini-ImageNet/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
            else './data/Mini-ImageNet/val1000Episode_5_way_5_shot.json'

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
