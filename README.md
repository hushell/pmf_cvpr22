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
We have a pytorch episodic dataset of [Meta-Dataset](https://github.com/google-research/meta-dataset).
The dataset has 10 domains, 4000+ classes. Episodes are formed in various-way-various-shot fashion.
The images are stored class-wise in h5 files (converted from the origianl tfrecords).
To use this dataset, set `args.dataset = full_meta_dataset` and 
`args.data_path = /group-volume/ASR-Unlabeled-Data/Datasets/meta_dataset`.


## Training
We support ProtoNet training with various pretrained backbones:
```
if args.arch == 'vit_base_patch16_224_in21k':
    ...
elif args.arch == 'dino_base_patch16':
    ...
elif args.arch == 'dino_small_patch16':
    ...
elif args.arch == 'beit_base_patch16_224_pt22k':
    ...
elif args.arch == 'clip_base_patch16_224':
    ...
elif args.arch == 'clip_resnet50':
    ...
elif args.arch == 'dino_resnet50':
    ...
elif args.arch == 'dino_xcit_medium_24_p16':
    ...
elif args.arch == 'dino_xcit_medium_24_p8':
    ...
```
It is recommended to run on a single GPU first by specifying `args.device = cuda:i`, where i = 0, 1, ... is the GPU id. 

### CIFAR-FS and Mini-ImageNet
We use `args.nSupport` to set the number of shots. For example, 5-way-5-shot training is the following:
```
python train.py --output outputs/your_experiment_name --dataset mini_imagenet --epoch 100 --lr 1e-4 --arch dino_base_patch16 --device cuda:1 --nSupport 5 
```
