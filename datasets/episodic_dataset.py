import os
import torch
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import json

from torchvision import transforms
from torchvision.datasets import ImageFolder


def PilLoaderRGB(imgPath) :
    return Image.open(imgPath).convert('RGB')


class EpisodeDataset(data.Dataset):
    """
    Dataloader to sample a task/episode.
    In case of 5-way 1-shot: nSupport = 1, nCls = 5.

    :param string imgDir: image directory, each category is in a sub file;
    :param int nCls: number of classes in each episode;
    :param int nSupport: number of support examples;
    :param int nQuery: number of query examples;
    :param transform: image transformation/data augmentation;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    """
    def __init__(self, imgDir, nCls, nSupport, nQuery, transform, inputW, inputH, nEpisode=2000):
        super().__init__()

        self.imgDir = imgDir
        self.clsList = os.listdir(imgDir)
        self.nCls = nCls
        self.nSupport = nSupport
        self.nQuery = nQuery
        self.transform = transform
        self.nEpisode = nEpisode

        floatType = torch.FloatTensor
        intType = torch.LongTensor

        self.tensorSupport = floatType(nCls * nSupport, 3, inputW, inputH)
        self.labelSupport = intType(nCls * nSupport)
        self.tensorQuery = floatType(nCls * nQuery, 3, inputW, inputH)
        self.labelQuery = intType(nCls * nQuery)
        self.imgTensor = floatType(3, inputW, inputH)

        # labels {0, ..., nCls-1}
        for i in range(self.nCls) :
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i

    def __len__(self):
        return self.nEpisode

    def __getitem__(self, idx):
        """
        Return an episode

        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery}
        """
        # select nCls from clsList
        clsEpisode = np.random.choice(self.clsList, self.nCls, replace=False)
        for i, cls in enumerate(clsEpisode) :
            clsPath = os.path.join(self.imgDir, cls)
            imgList = os.listdir(clsPath)

            # in total nQuery+nSupport images from each class
            imgCls = np.random.choice(imgList, self.nQuery + self.nSupport, replace=False)

            for j in range(self.nSupport) :
                img = imgCls[j]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery) :
                img = imgCls[j + self.nSupport]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        ## Random permutation. Though this is not necessary in our approach
        permSupport = torch.randperm(self.nCls * self.nSupport)
        permQuery = torch.randperm(self.nCls * self.nQuery)

        return (self.tensorSupport[permSupport],
               self.labelSupport[permSupport],
               self.tensorQuery[permQuery],
               self.labelQuery[permQuery])


class EpisodeJSONDataset(data.Dataset):
    """
    To make validation results comparable, we fix 1000 episodes for validation.

    :param string episodeJson: ./data/Dataset/val1000Episode_K_way_N_shot.json
    :param string imgDir: image directory, each category is in a sub file;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    :param valTransform: image transformation/data augmentation;
    """
    def __init__(self, episodeJson, imgDir, inputW, inputH, valTransform):
        with open(episodeJson, 'r') as f :
            self.episodeInfo = json.load(f)

        self.imgDir = imgDir
        self.nEpisode = len(self.episodeInfo)
        self.nCls = len(self.episodeInfo[0]['Support'])
        self.nSupport = len(self.episodeInfo[0]['Support'][0])
        self.nQuery = len(self.episodeInfo[0]['Query'][0])
        self.transform = valTransform

        floatType = torch.FloatTensor
        intType = torch.LongTensor

        self.tensorSupport = floatType(self.nCls * self.nSupport, 3, inputW, inputH)
        self.labelSupport = intType(self.nCls * self.nSupport)
        self.tensorQuery = floatType(self.nCls * self.nQuery, 3, inputW, inputH)
        self.labelQuery = intType(self.nCls * self.nQuery)

        self.imgTensor = floatType(3, inputW, inputH)
        for i in range(self.nCls) :
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i

    def __getitem__(self, index):
        """
        Return an episode

        :param int index: index of data example
        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery}
        """
        for i in range(self.nCls) :
            for j in range(self.nSupport) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Support'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Query'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        return (self.tensorSupport,
               self.labelSupport,
               self.tensorQuery,
               self.labelQuery)

    def __len__(self):
        """
        Number of episodes
        """
        return self.nEpisode


def ValLoader(episodeJson, imgDir, inputW, inputH, valTransform):
    dataloader = data.DataLoader(ValImageFolder(episodeJson, imgDir, inputW, inputH, valTransform),
                                 shuffle=False)
    return dataloader


def TrainLoader(batchSize, imgDir, trainTransform) :
    dataloader = data.DataLoader(ImageFolder(imgDir, trainTransform),
                                 batch_size=batchSize, shuffle=True, drop_last=True)
    return dataloader


if __name__ == '__main__' :
    import torchvision.transforms as transforms
    mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([
                                         transforms.RandomCrop(32, padding=8),
                                         transforms.RandomHorizontalFlip(),
                                         lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    TrainEpisodeSampler = EpisodeDataset(imgDir = '../data/cifar-fs/train/',
                                         nCls = 5,
                                         nSupport = 5,
                                         nQuery = 15,
                                         transform = trainTransform,
                                         inputW = 32,
                                         inputH = 32)
    data = TrainEpisodeSampler[0]
    print (data[1])

