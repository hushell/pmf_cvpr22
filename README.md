# Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference
---

## Prerequisites
```
pip install -r requirements.txt
```

## Datasets
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

The dataset has 10 domains, 4000+ classes. Episodes are formed in various-way-various-shot fashion.
The images are stored class-wise in h5 files (converted from the origianl tfrecords).
To use this dataset, set `args.dataset = full_meta_dataset` and `args.data_path = /path/to/meta_dataset`.
We will soon provide a link for downloading h5 files.


## Meta-Training
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
It is recommended to run on a single GPU first by specifying `args.device = cuda:i`, where i is the GPU id. 

### Meta-training on CIFAR-FS and Mini-ImageNet
We use `args.nSupport` to set the number of shots. For example, 5-way-5-shot training is the following:
```
python train.py --output outputs/your_experiment_name --dataset cifar_fs --epoch 100 --lr 5e-5 --arch dino_small_patch16 --device cuda:0 --nSupport 5 --fp16
```

### Meta-training on Meta-Dataset ()
```
python train.py --output outputs/your_experiment_name --dataset full_meta_dataset --data-path /path/to/h5/files/ --num_workers 4 --base_sources aircraft cu_birds dtd
 ilsvrc_2012 omniglot fungi vgg_flower quickdraw --epochs 100 --lr 5e-4 --arch dino_small_patch16 --dist-eval --device cuda:0 --fp16
```
