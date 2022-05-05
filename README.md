# P>M>F pipeline for few-shot learning (CVPR2022)
---

**Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference**

*[Shell Xu Hu](https://hushell.github.io/), [Da Li](https://dali-dl.github.io/), [Jan Stühmer](https://scholar.google.com/citations?user=pGukv5YAAAAJ&hl=en), [Minyoung Kim](https://sites.google.com/site/mikim21/) and [Timothy Hospedales](https://homepages.inf.ed.ac.uk/thospeda/)*


[[Project page](https://hushell.github.io/pmf/)]
[[Arxiv](https://arxiv.org/abs/2204.07305)]


If you find our project helpful, please consider cite:
```
@inproceedings{hu2022pmf,
               author = {Hu, Shell Xu
                         and Li, Da
                         and St\"uhmer, Jan
                         and Kim, Minyoung
                         and Hospedales, Timothy M.},
               title = {Pushing the Limits of Simple Pipelines for Few-Shot Learning:
                        External Data and Fine-Tuning Make a Difference},
               booktitle = {CVPR},
               year = {2022}
}
```


## Updates

***18/04/2022****
We released a [Gradio demo on Huggingface Space](https://huggingface.co/spaces/hushell/pmf_with_gis) for few-shot learning where the support set is found by Google Image Search, making it a cheap version of CLIP-like zero-shot learning.

***08/12/2021***
[An early version of P>M>F](https://github.com/henrygouk/neurips-metadl-2021) won 2nd place in [NeurIPS-MetaDL-2021](https://metalearning.chalearn.org/metadlneurips2021) with [Henry Gouk](https://www.henrygouk.com/).


## Table of Content
* [Prerequisites](#prerequisites)
* [Datasets](#datasets)
    * [CIFAR-FS and Mini-ImageNet](#cifar-fs-and-mini-imagenet)
    * [Meta-Dataset](#meta-dataset)
    * [CDFSL](#cdfsl)
* [Pre-training](#pre-training)
* [Meta-training](#meta-training)
    * [ProtoNet on CIFAR-FS and Mini-ImageNet](#protonet-on-cifar-fs-and-mini-imagenet)
    * [ProtoNet on Meta-Dataset](#protonet-on-meta-dataset)
    * [ProtoNet on Meta-Dataset with ImageNet only](#protonet-on-meta-dataset-with-imagenet-only)
* [Meta-testing](#meta-testing)
    * [Vanilla](#vanilla)
    * [Test-time fine-tuning on Meta-Dataset](#test-time-fine-tuning-on-meta-dataset)
    * [Cross-domain few-shot learning](#cross-domain-few-shot-learning)


## Prerequisites
```
pip install -r requirements.txt
```

## Datasets
We provide dataset classes and DDP dataloaders for CIFAR-FS, Mini-ImageNet and Meta-Dataset. 
For more details, check [datasets/__init__.py:get_sets()](datasets/__init__.py#L15) and [datasets/__init__.py:get_loaders()](datasets/__init__.py#L70).

### CIFAR-FS and Mini-ImageNet
```
cd scripts
sh download_cifarfs.sh
sh download_miniimagenet.sh
```
To use these two datasets, set `args.dataset = cifar_fs` or `args.dataset = mini_imagenet`.

### Meta-Dataset
We implement a pytorch version of [Meta-Dataset](https://github.com/google-research/meta-dataset).
Our version is based on [mboudiaf's pytorch-meta-dataset](https://github.com/mboudiaf/pytorch-meta-dataset).
We replace the `tfrecords` to `h5` files to largely reduce IO latency. 

The dataset has 10 domains, 4000+ classes. Episodes are formed in various-way-various-shot fashion (can have 900+ images).
The images are stored class-wise in h5 files (converted from the origianl tfrecords).
To use this dataset, set `args.dataset = full_meta_dataset` and `args.data_path = /path/to/meta_dataset`.
We will soon provide a link for downloading h5 files.


## Pre-training
We support various pretrained ViT and ResNet50 backbones:
```
args.arch = 'dino_base_patch16' # DINO self-supervised ViT-base on ImageNet-1k
          = 'dino_small_patch16' # DINO self-supervised ViT-small on ImageNet-1k
          = 'beit_base_patch16_224_pt22k' # BERT self-supervised ViT-base on ImageNet-21k
          = 'clip_base_patch16_224' # ViT-base via CLIP on YFCC-100M
          = 'clip_resnet50' # ResNet50 via CLIP on YFCC-100M
          = 'dino_resnet50' # DINO self-supervised ResNet50 on ImageNet-1k
```


## Meta-Training

### On CIFAR-FS and Mini-ImageNet
It is recommended to run on a single GPU first by specifying `args.device = cuda:i`, where i is the GPU id. 
We use `args.nSupport` to set the number of shots. For example, 5-way-5-shot training is the following:
```
python main.py --output outputs/your_experiment_name --dataset cifar_fs --epoch 100 --lr 5e-5 --arch dino_small_patch16 --device cuda:0 --nSupport 5 --fp16
```
The minimum GPU memory is 11GB.

### On Meta-Dataset (8 base domains)
Since each class is stored in a h5 file, training will open many files. The following command is required before launching the code:
```
ulimit -n 100000 # may need to check `ulimit -Hn` first to know the hard limit
```
Various-way-various-shot training (#ways = 5-50, max #query = 10, max #supp = 500, max #supp per class = 100):

```
python main.py --output outputs/your_experiment_name --dataset meta_dataset --data-path /path/to/h5/files/ --num_workers 4 --base_sources aircraft cu_birds dtd ilsvrc_2012 omniglot fungi vgg_flower quickdraw --epochs 100 --lr 5e-4 --arch dino_small_patch16 --dist-eval --device cuda:0 --fp16
```
The minimum GPU memory is 24GB.

### Meta-training on Meta-Dataset (ImageNet only)
Just replace `--base_sources ...` by `--base_sources ilsvrc_2012`.

### Multi-GPU DDP on a single machine
First, setting up the following environmental variables: 
```
export RANK=0 # machine id
export WORLD_SIZE=1 # total number of machines
```
For example, if you got 8 GPUs, 
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --output outputs/your_experiment_name --dataset meta_dataset --data-path /path/to/h5/files/ --num_workers 4 --base_sources aircraft cu_birds dtd ilsvrc_2012 omniglot fungi vgg_flower quickdraw --epochs 100 --lr 5e-4 --arch dino_small_patch16 --dist-eval --fp16
```


## Meta-Testing

### Default 

### Test-time fine-tuning on Meta-Dataset

### Cross-domain few-shot learning
