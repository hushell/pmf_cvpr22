import os
import numpy as np
import torch
#from timm.models import create_model
from .protonet import ProtoNet
from .deploy import ProtoNet_Finetune, ProtoNet_Auto_Finetune, ProtoNet_AdaTok, ProtoNet_AdaTok_EntMin


def get_backbone(args):
    if args.arch == 'vit_base_patch16_224_in21k':
        from .vit_google import VisionTransformer, CONFIGS

        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, 224)

        url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
        pretrained_weights = 'pretrained_ckpts/vit_base_patch16_224_in21k.npz'

        if not os.path.exists(pretrained_weights):
            try:
                import wget
                os.makedirs('pretrained_ckpts', exist_ok=True)
                wget.download(url, pretrained_weights)
            except:
                print(f'Cannot download pretrained weights from {url}. Check if `pip install wget` works.')

        model.load_from(np.load(pretrained_weights))
        print('Pretrained weights found at {}'.format(pretrained_weights))

    elif args.arch == 'dino_base_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'deit_base_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'deit_small_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'dino_small_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        if not args.no_pretrain:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

            model.load_state_dict(state_dict, strict=True)
            print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'beit_base_patch16_224_pt22k':
        from .beit import default_pretrained_model
        model = default_pretrained_model(args)
        print('Pretrained BEiT loaded')

    elif args.arch == 'clip_base_patch16_224':
        from . import clip
        model, _ = clip.load('ViT-B/16', 'cpu')

    elif args.arch == 'clip_resnet50':
        from . import clip
        model, _ = clip.load('RN50', 'cpu')

    elif args.arch == 'dino_resnet50':
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        model.fc = torch.nn.Identity()

        if not args.no_pretrain:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)

    elif args.arch == 'resnet50':
        from torchvision.models.resnet import resnet50

        pretrained = not args.no_pretrain
        model = resnet50(pretrained=pretrained)
        model.fc = torch.nn.Identity()

    elif args.arch == 'resnet18':
        from torchvision.models.resnet import resnet18

        pretrained = not args.no_pretrain
        model = resnet18(pretrained=pretrained)
        model.fc = torch.nn.Identity()

    elif args.arch == 'dino_xcit_medium_24_p16':
        model = torch.hub.load('facebookresearch/xcit:main', 'xcit_medium_24_p16')
        model.head = torch.nn.Identity()
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)

    elif args.arch == 'dino_xcit_medium_24_p8':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')

    elif args.arch == 'simclrv2_resnet50':
        import sys
        sys.path.insert(
            0,
            'cog',
        )
        import model_utils

        model_utils.MODELS_ROOT_DIR = 'cog/models'
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts/simclrv2_resnet50.pth')
        resnet, _ = model_utils.load_pretrained_backbone(args.arch, ckpt_file)

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x, apply_fc=False)

        model = Wrapper(resnet)

    elif args.arch in ['mocov2_resnet50', 'swav_resnet50', 'barlow_resnet50']:
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts_converted/{}.pth'.format(args.arch))
        ckpt = torch.load(ckpt_file)

        msg = model.load_state_dict(ckpt, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        # remove the fully-connected layer
        model.fc = torch.nn.Identity()

    else:
        raise ValueError(f'{args.arch} is not conisdered in the current code.')

    return model


def get_model(args):
    backbone = get_backbone(args)

    if args.deploy == 'vanilla':
        model = ProtoNet(backbone)
    elif args.deploy == 'finetune':
        model = ProtoNet_Finetune(backbone, args.ada_steps, args.ada_lr, args.aug_prob, args.aug_types)
    elif args.deploy == 'finetune_autolr':
        model = ProtoNet_Auto_Finetune(backbone, args.ada_steps, args.aug_prob, args.aug_types)
    elif args.deploy == 'ada_tokens':
        model = ProtoNet_AdaTok(backbone, args.num_adapters,
                                args.ada_steps, args.ada_lr)
    elif args.deploy == 'ada_tokens_entmin':
        model = ProtoNet_AdaTok_EntMin(backbone, args.num_adapters,
                                       args.ada_steps, args.ada_lr)
    else:
        raise ValueError(f'deploy method {args.deploy} is not supported.')
    return model
