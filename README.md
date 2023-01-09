# P>M>F pipeline for few-shot learning (CVPR2022)
---

**Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference**

*[Shell Xu Hu](https://hushell.github.io/), [Da Li](https://dali-dl.github.io/), [Jan Stühmer](https://scholar.google.com/citations?user=pGukv5YAAAAJ&hl=en), [Minyoung Kim](https://sites.google.com/site/mikim21/) and [Timothy Hospedales](https://homepages.inf.ed.ac.uk/thospeda/)*

[[Project page](https://hushell.github.io/pmf/)]
[[blog](https://research.samsung.com/blog/CVPR-2022-Series-5-P-M-F-The-Pre-Training-Meta-Training-and-Fine-Tuning-Pipeline-for-Few-Shot-Learning)]
[[Arxiv](https://arxiv.org/abs/2204.07305)]
[[Gradio demo](https://huggingface.co/spaces/hushell/pmf_with_gis)]


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pushing-the-limits-of-simple-pipelines-for/few-shot-image-classification-on-meta-dataset)](https://paperswithcode.com/sota/few-shot-image-classification-on-meta-dataset?p=pushing-the-limits-of-simple-pipelines-for)


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
***08/11/2022***
We released the [checkpoints](https://huggingface.co/hushell/pmf_metadataset_dino) meta-trained on Meta-Dataset with pre-trained DINO.

***18/04/2022***
We released a [Gradio demo on Huggingface Space](https://huggingface.co/spaces/hushell/pmf_with_gis) for few-shot learning where the support set is created by text-to-image retrieval, making it a cheap version of CLIP-like zero-shot learning.

***08/12/2021***
[An early version of P>M>F](https://github.com/henrygouk/neurips-metadl-2021) won 2nd place in [NeurIPS-MetaDL-2021](https://metalearning.chalearn.org/metadlneurips2021) with [Henry Gouk](https://www.henrygouk.com/).


## Table of Content
* [Pre-trained model checkpoints](#pre-trained-model-checkpoints)
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


## Pre-trained model checkpoints
We release three [checkpoints](https://huggingface.co/hushell/pmf_metadataset_dino) meta-trained on Meta-Dataset with pre-trained DINO for reproducibility.


## Prerequisites
```
pip install -r requirements.txt
```
The code was tested with Python 3.8.1 and Pytorch >= 1.7.0.


## Datasets
We provide dataset classes and DDP dataloaders for CIFAR-FS, Mini-ImageNet and Meta-Dataset,
and also adapted the [CDFSL datasets](https://arxiv.org/abs/1912.07200v3) to our pipeline.

The overall structure of the `datasets` folder is the following:
```
datasets/
├── cdfsl/                     # CDFSL datasets
├── episodic_dataset.py        # CIFAR-FS & Mini-ImageNet
├── __init__.py                # summary & interface
├── meta_dataset/              # code adapted from Google's meta-dataset and pytorch-meta-dataset
├── meta_h5_dataset.py         # meta-dataset class to sample episodes and fetch data from h5 files
├── meta_val_dataset.py        # meta-dataset class for validation with fixed val episodes
```

We unify the dataset definition, distributed sampler and randomness control in [datasets/\__init__.py](datasets/__init__.py), but in general the only functions you may need to pay attention are
1. [get_loaders()](datasets/__init__.py#L70). Usage: [main.py:51](https://github.com/hushell/pmf_cvpr22/blob/86fa88c9a446886d530d865a360b09a1064d928f/main.py#L51)
2. [get_bscd_loader()](datasets/__init__.py#L158), which is for CDFSL benchmark. Usage: [test_bscdfsl.py:52](https://github.com/hushell/pmf_cvpr22/blob/86fa88c9a446886d530d865a360b09a1064d928f/test_bscdfsl.py#L52).


### CIFAR-FS and Mini-ImageNet
```
cd scripts
sh download_cifarfs.sh
sh download_miniimagenet.sh
```
To use these two datasets, set `--dataset cifar_fs` or `--dataset mini_imagenet`.

### Meta-Dataset
We implement a pytorch version of [Meta-Dataset](https://github.com/google-research/meta-dataset).
Our implementation is based on [mboudiaf's pytorch-meta-dataset](https://github.com/mboudiaf/pytorch-meta-dataset).
The major change is we replace the `tfrecords` to `h5` files to largely reduce IO latency. 
This also enables efficient DDP data-loading otherwise tfrecords leads to streaming dataset which is less easy to DDP.

The dataset has 10 domains, 4000+ classes. Episodes are formed in various-way-various-shot fashion, where an episode can have 900+ images.
The images are stored class-wise in h5 files (converted from the origianl tfrecords, one for each class).
To train and test on this dataset, set `--dataset meta_dataset` and `--data_path /path/to/meta_dataset`.

To download the h5 files, 
```
git clone https://huggingface.co/datasets/hushell/meta_dataset_h5
```

You can also generate h5 files by yourself following these steps:
1. Download 10 domains (e.g., cu_birds) listed in [google-research/meta-dataset](https://github.com/google-research/meta-dataset#downloading-and-converting-datasets), which will create `tfrecords` files for each class and `dataset_spec.json` for each domain.
Once done, you should get a folder `meta-dataset/tf_records` with 11 sub-folders of `*.tfrecords` files (including ilsvrc_2012 and ilsvrc_2012_v2).

2. Generate the index files of tfrecords with existing tool:
```
export RECORDS='path/to/tfrecords'
for source in omniglot aircraft cu_birds dtd quickdraw vgg_flower traffic_sign mscoco ilsvrc_2012; do \
		source_path=${RECORDS}/${source} ;\
		find ${source_path} -name '*.tfrecords' -type f -exec sh -c 'python3 datasets/meta_dataset/tfrecord/tools/tfrecord2idx.py $2 ${2%.tfrecords}.index' sh ${source_path} {} \; ;\
	done ;\
```
This command will create for each tfrecords file an index file with the same name. E.g., `0.tfrecords -> 0.index`.

3. Convert `tfrecords` to `h5` by calling 
```
python scripts/convert_tfrecord_to_h5.py /path/to/meta-dataset/tf_records
```
This command will create for each tfrecords file an h5 file with the same name. E.g., `0.tfrecords -> 0.h5`.

4. Generate 120 validation tasks per domain on a set of reserved classes by calling 
```
python scripts/generate_val_episodes_to_h5.py --data-path /path/to/meta-dataset/tf_records
```
This goes into a single `h5` file for each domain. E.g., `cu_birds/val_ep120_img128.h5`. This is to remove randomness in validation.

### CDFSL
The purpose of this benchmark is to evaluate how model trained on Mini-ImageNet (source domain) performs on cross-domain meta-test tasks. 
So we only need to download the [target domains](https://github.com/yunhuiguo/CVPR-2021-L2ID-Classification-Challenges#target-domains), and extract the files into `./data/`.
You'll need to have these 4 sub-folders: 
```
./data/ChestX
./data/CropDiseases
./data/EuroSAT/2750
./data/ISIC
```
Check [get_bscd_loader()](datasets/__init__.py#L158) for the data loader details.

## Pre-training
We support multiple pretrained foundation models. E.g., 
`DINO ViT-base`, `DINO ViT-small`, `DINO ResNet50`, `BeiT ViT-base`, `CLIP ViT-base`, `CLIP ResNet50` and so on.
For options of `args.arch`, please check [get_backbone()](models/__init__.py#L9).


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
For various-way-various-shot training (#ways = 5-50, max #query = 10, max #supp = 500, max #supp per class = 100), the following command yields a P(DINO ViT-small) -> M(ProtoNet) pipeline to update backbone:
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
For example, if you got 8 GPUs, run this command will accumulate gradients from 8 parallel episodes:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --output outputs/your_experiment_name --dataset meta_dataset --data-path /path/to/h5/files/ --num_workers 4 --base_sources aircraft cu_birds dtd ilsvrc_2012 omniglot fungi vgg_flower quickdraw --epochs 100 --lr 5e-4 --arch dino_small_patch16 --dist-eval --fp16
```

## Meta-Testing

### For datasets without domain shift
Copy the same command for training, which can be found in `outputs/your_experiment_name/log.txt` (should be the first line or search keyword main.py),
and add `--eval`.

### Fine-tuning on meta-test tasks
When domain shift exists between meta-training and meta-testing, we enable different model deployment modes: `vanilla` (ProtoNet classification) and `finetune` (the backbone will be updated on support set). Check [get_model()](models/__init__.py:175) for the actual implementation.

For `finetune`, a few hyper-parameters are introduced: `args.ada_steps`, `args.ada_lr`, `args.aug_prob`, `args.aug_types`, among which `args.ada_lr` is the more sensitive one and requires validation (e.g., `--ada_lr 0.001`). We also recommend to play with `args.ada_steps` (e.g., `--ada_steps 50`). The good news is that empirically we find good performance can be achieved by domain-wise hyper-parameter search. As recommended in our paper, 3-5 episodes with labeled query set is sufficient to tell what the best hyper-parameters are, which makes model deployment practical for a novel domain with a few annotated examples. 

A meta-testing command example for Meta-Dataset with fine-tuning is 
``` 
python -m torch.distributed.launch --nproc_per_node=8 --use_env test_meta_dataset.py --data-path /path/to/meta_dataset/ --dataset meta_dataset --arch dino_small_patch16 --deploy finetune --output outputs/your_experiment_name --resume outputs/your_experiment_name/best.pth --dist-eval --ada_steps 100 --ada_lr 0.0001 --aug_prob 0.9 --aug_types color transition
``` 

To meta-test without fine-tuning, just replace `--deploy finetune` with `--deploy vanilla`.

If you don't want to enable DDP for testing, just replacing `--dist-eval` by `--device cuda:0` and remove `torch.distributed.launch` part. 
By default, all 10 domains will be evaluated, but you may meta-test only a subset by specifying which domains should be executed with `--test_sources`. Check [utils/args.py:48](utils/args.py) for domain names.

Below are the results on Meta-Dataset test-set for DINO checkpoints:

Method                     |ILSVRC (test)              |Omniglot                   |Aircraft                   |Birds                      |Textures                   |QuickDraw                  |Fungi                      |VGG Flower                 |Traffic signs              |MSCOCO
---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------
[md_full_128x128_dinosmall_fp16_lr5e-5](https://huggingface.co/hushell/pmf_metadataset_dino/blob/main/md_full_128x128_dinosmall_fp16_lr5e-5/best.pth)             |73.52±0.80&nbsp;(lr=0.0001)       |92.17±0.57&nbsp;(lr=0.0001)     |89.49±0.52&nbsp;(lr=0.001)        |91.04±0.37&nbsp;(lr=0.0001)       |85.73±0.62&nbsp;(lr=0.001)      |79.43±0.67&nbsp;(lr=0.0001)     |74.99±0.94&nbsp;(lr=0)       |95.30±0.44&nbsp;(lr=0.001)        |89.85±0.76&nbsp;(lr=0.01)        |59.69±1.02&nbsp;(lr=0.001)
[md_inet_128x128_dinosmall_fp16_lr2e-4](https://huggingface.co/hushell/pmf_metadataset_dino/blob/main/md_imagenet_128x128_dinosmall_fp16_lr2e-4/best.pth)          |75.51±0.72&nbsp;(lr=0.001)       |82.81±1.10&nbsp;(lr=0.01)       |78.38±1.09&nbsp;(lr=0.01)       |85.18±0.77&nbsp;(lr=0.001)     |86.95±0.60&nbsp;(lr=0.001)       |74.47±0.83&nbsp;(lr=0.01)     |55.16±1.09&nbsp;(lr=0)       |94.66±0.48&nbsp;(lr=0)       |90.04±0.81&nbsp;(lr=0.01)       |62.60±0.96&nbsp;(lr=0.001)


### Cross-domain few-shot learning
Meta-testing CDFSL is almost the same as described in previous section for Meta-Dataset. However, we create another script [test_bscdfsl.py](test_bscdfsl.py) to fit CDFSL's original data loaders. 

An meta-testing command example for CDFSL with fine-tuning is
```
python test_bscdfsl.py --test_n_way 5 --n_shot 5 --device cuda:0 --arch dino_small_patch16 --deploy finetune --output outputs/your_experiment_name --resume outputs/your_experiment_name/best.pth --ada_steps 100 --ada_lr 0.0001 --aug_prob 0.9 --aug_types color transition
```
Changing `--n_shot` to 20 or 50 to evaluate other settings.
