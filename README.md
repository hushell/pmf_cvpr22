# P>M>F pipeline for few-shot learning (CVPR2022)
---

**Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference**

*[Shell Xu Hu](https://hushell.github.io/), [Da Li](https://dali-dl.github.io/), [Jan Stühmer](https://scholar.google.com/citations?user=pGukv5YAAAAJ&hl=en), [Minyoung Kim](https://sites.google.com/site/mikim21/) and [Timothy Hospedales](https://homepages.inf.ed.ac.uk/thospeda/)*

[[Project page](https://hushell.github.io/pmf/)]
[[Arxiv](https://arxiv.org/abs/2204.07305)]
[[Gradio demo](https://huggingface.co/spaces/hushell/pmf_with_gis)]

If you find our project helpful, please consider cite our paper:
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

***18/04/2022***
We released a [Gradio demo on Huggingface Space](https://huggingface.co/spaces/hushell/pmf_with_gis) for few-shot learning where the support set is created by text-to-image retrieval, making it a cheap version of CLIP-like zero-shot learning.

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
    * [On CIFAR-FS and Mini-ImageNet](#on-cifar-fs-and-mini-imagenet)
    * [On Meta-Dataset with 8 source domains](#on-meta-dataset-with-8-source-domains)
    * [On Meta-Dataset with ImageNet only](#on-meta-dataset-with-imagenet-only)
* [Meta-testing](#meta-testing)
    * [For datasets without domain shift](#for-datasets-without-domain-shift)
    * [Fine-tuning on meta-test tasks](#fine-tuning-on-meta-test-tasks)
    * [Cross-domain few-shot learning](#cross-domain-few-shot-learning)


## Prerequisites
```
pip install -r requirements.txt
```
The code was tested with Python 3.8.1 and Pytorch >= 1.7.0.


## Datasets
We provide dataset classes and DDP dataloaders for CIFAR-FS, Mini-ImageNet and Meta-Dataset,
and also adapted the [CDFSL datasets](https://arxiv.org/abs/1912.07200v3) to our pipeline.
For more details, check [datasets/__init__.py:get_sets()](datasets/__init__.py#L15), [datasets/__init__.py:get_loaders()](datasets/__init__.py#L70) and [datasets/__init__.py:get_bscd_loader()](datasets/__init__.py#L158).

### CIFAR-FS and Mini-ImageNet
```
cd scripts
sh download_cifarfs.sh
sh download_miniimagenet.sh
```
To use these two datasets, set `args.dataset = cifar_fs` or `args.dataset = mini_imagenet`.

### Meta-Dataset
We implement a pytorch version of [Meta-Dataset](https://github.com/google-research/meta-dataset).
Our implementation is based on [mboudiaf's pytorch-meta-dataset](https://github.com/mboudiaf/pytorch-meta-dataset).
The major change is we replace the `tfrecords` to `h5` files to largely reduce IO latency. 

The dataset has 10 domains, 4000+ classes. Episodes are formed in various-way-various-shot fashion, where an episode can have 900+ images.
The images are stored class-wise in h5 files (converted from the origianl tfrecords).
To train and test on this dataset, set `args.dataset = meta_dataset` and `args.data_path = /path/to/meta_dataset`.

We will soon provide a link for downloading the `h5` files. You may generate them by yourself following these steps:
1. Download 10 datasets (one for each domain) listed in [Downloading and converting datasets](https://github.com/google-research/meta-dataset#downloading-and-converting-datasets), which will create `tfrecords` files for each class and `dataset_spec.json` for each domain.
2. Convert `tfrecords` to `h5` using [scripts/convert_tfrecord_to_h5.py](scripts/convert_tfrecord_to_h5.py).
3. Generate 600 validation tasks on a set of reserved classes using [scripts/generate_val_episodes_to_h5.py](scripts/generate_val_episodes_to_h5.py) into a single `h5` file. This is to remove randomness in validation.
You will need to specify the path to your `tfrecords` files in the above python scripts.

### CDFSL
The purpose of this benchmark is to evaluate model trained on Mini-ImageNet (source domain) by cross-domain meta-test tasks. 
So we only need to download the [target datasets (domains)](https://github.com/yunhuiguo/CVPR-2021-L2ID-Classification-Challenges#target-domains), and extract the files into `./data/`.
You'll need to have these 4 sub-folders: 
```
./data/ChestX
./data/CropDiseases
./data/EuroSAT/2750
./data/ISIC
```

## Pre-training
We support multiple pretrained foundation models. E.g., 
`DINO ViT-base`, `DINO ViT-small`, `DINO ResNet50`, `BeiT ViT-base`, `CLIP ViT-base`, `CLIP ResNet50` and so on.
For options of `args.arch`, please check [models/__init__.py::get_backbone()](models/__init__.py#L9).


## Meta-Training

### On CIFAR-FS and Mini-ImageNet
It is recommended to run on a single GPU first by specifying `args.device = cuda:i`, where i is the GPU id to be used. 
We use `args.nSupport` to set the number of shots. For example, 5-way-5-shot training command of CIFAR-FS writes as
```
python main.py --output outputs/your_experiment_name --dataset cifar_fs --epoch 100 --lr 5e-5 --arch dino_small_patch16 --device cuda:0 --nSupport 5 --fp16
```
Because at least one episode has to be hosted on the GPU, the program is quite memory hungry. Mixed precision (`--fp16`) is recommended.

### On Meta-Dataset with 8 source domains
Since each class is stored in a h5 file, training will open many files. The following command is required before launching the code:
```
ulimit -n 100000 # may need to check `ulimit -Hn` first to know the hard limit
```
For various-way-various-shot training (#ways = 5-50, max #query = 10, max #supp = 500, max #supp per class = 100), the following command yields a P(DINO ViT-small) -> M(ProtoNet) updated backbone:
```
python main.py --output outputs/your_experiment_name --dataset meta_dataset --data-path /path/to/meta-dataset/ --num_workers 4 --base_sources aircraft cu_birds dtd ilsvrc_2012 omniglot fungi vgg_flower quickdraw --epochs 100 --lr 5e-4 --arch dino_small_patch16 --dist-eval --device cuda:0 --fp16
```
The minimum GPU memory is 24GB. The logging file `outputs/your_experiment_name/log.txt` can be used to monitor model performance (as you can check it remotely).

### On Meta-Dataset with ImageNet only
Just replace `--base_sources ...` by `--base_sources ilsvrc_2012`.

### Distributed data parallel on a single machine
First, setting up the following environmental variables: 
```
export RANK=0 # machine id
export WORLD_SIZE=1 # total number of machines
```
For example, if you got 8 GPUs, run this command will accumulate gradients from 8 episodes:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --output outputs/your_experiment_name --dataset meta_dataset --data-path /path/to/h5/files/ --num_workers 4 --base_sources aircraft cu_birds dtd ilsvrc_2012 omniglot fungi vgg_flower quickdraw --epochs 100 --lr 5e-4 --arch dino_small_patch16 --dist-eval --fp16
```

## Meta-Testing

### For datasets without domain shift
Copy the same command for training, which can be found in `outputs/your_experiment_name/log.txt` (search `main.py`),
and add `--eval`

### Fine-tuning on meta-test tasks
When domain shift exists between meta-training and meta-testing, we enable different model deployments: vanilla (ProtoNet classification) and fine-tuning (the backbone will be updated on support set).
For the latter, a few hyper-parameters are introduced: `args.ada_steps`, `args.ada_lr`, `aug_prob`, `aug_types`, among which `args.ada_lr` is the more sensitive one and requires validation.

An example of meta-testing command for Meta-Dataset with fine-tuning is
```
python test_meta_dataset.py --data-path /path/to/meta_dataset/ --dataset meta_dataset --device cuda:0 --arch dino_small_patch16 --deploy finetune --output outputs/your_experiment_name --resume outputs/your_experiment_name/best.pth 
```
To meta-test without fine-tuning, just replace `--deploy finetune` with `--deploy vanilla`.

A DDP version of the above command is also available: just replacing `--device cuda:0` with `--dist-eval`. By default, all 10 domains will be evaluated, but you may meta-test only a subset by specifying which domains should be executed with `--test_sources`. Check [utils/args.py](utils/args.py) for domain names.

### Cross-domain few-shot learning
Meta-testing CDFSL is almost the same as that of Meta-Dataset. However, we create another script to fit CDFSL's original data loaders. 
An example of meta-testing command for CDFSL with fine-tuning is
```
python test_meta_dataset.py --test_n_way 5 --n_shot 5 --device cuda:0 --arch dino_small_patch16 --deploy finetune --output outputs/your_experiment_name --resume outputs/your_experiment_name/best.pth 
```
