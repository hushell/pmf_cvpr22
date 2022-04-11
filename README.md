# Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference
---

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
We support ProtoNet training with various pretrained backbones:
```
args.arch = 'vit_base_patch16_224_in21k'
          = 'dino_base_patch16'
          = 'dino_small_patch16'
          = 'beit_base_patch16_224_pt22k'
          = 'clip_base_patch16_224'
          = 'clip_resnet50'
          = 'dino_resnet50'
          = 'dino_xcit_medium_24_p16'
          = 'dino_xcit_medium_24_p8'
```


## Meta-Training

### Meta-training on CIFAR-FS and Mini-ImageNet
It is recommended to run on a single GPU first by specifying `args.device = cuda:i`, where i is the GPU id. 
We use `args.nSupport` to set the number of shots. For example, 5-way-5-shot training is the following:
```
python main.py --output outputs/your_experiment_name --dataset cifar_fs --epoch 100 --lr 5e-5 --arch dino_small_patch16 --device cuda:0 --nSupport 5 --fp16
```
The minimum GPU memory is 11GB.

### Meta-training on Meta-Dataset (8 base domains)
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
