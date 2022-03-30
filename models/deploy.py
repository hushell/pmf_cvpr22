import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from copy import deepcopy
from tqdm import tqdm
from timm.utils import accuracy
from .protonet import ProtoNet
from .utils import trunc_normal_, DiffAugment


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@torch.jit.script
def entropy_loss(x):
    return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()


def unique_indices(x):
    """
    Ref: https://github.com/rusty1s/pytorch_unique
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, perm


class ProtoNet_Auto_Finetune(ProtoNet):
    def __init__(self, backbone, num_iters=50, aug_prob=0.9,
                 aug_types=['color', 'translation'], lr_lst=[0.01, 0.001, 0.0001]):
        super().__init__(backbone)
        self.num_iters = num_iters
        self.lr_lst = lr_lst
        self.aug_types = aug_types
        self.aug_prob = aug_prob

        state_dict = backbone.state_dict()
        self.backbone_state = deepcopy(state_dict)

    def forward(self, supp_x, supp_y, qry_x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        qry_x.shape = [B, nQry, C, H, W]
        """
        B, nSupp, C, H, W = supp_x.shape
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
        device = qry_x.device

        criterion = nn.CrossEntropyLoss()
        supp_x = supp_x.view(-1, C, H, W)
        qry_x = qry_x.view(-1, C, H, W)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp
        supp_y = supp_y.view(-1)

        def single_step(z, mode=True, x=None, y=None, y_1hot=None):
            '''
            z = Aug(supp_x) or qry_x
            global vars: supp_x, supp_y, supp_y_1hot
            '''
            with torch.set_grad_enabled(mode):
                # recalculate prototypes from supp_x with updated backbone
                proto_f = self.backbone.forward(x).unsqueeze(0)

                if y_1hot is None:
                    prototypes = proto_f
                else:
                    prototypes = torch.bmm(y_1hot.float(), proto_f) # B, nC, d
                    prototypes = prototypes / y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0

                # compute feature for z
                feat = self.backbone.forward(z)
                feat = feat.view(B, z.shape[0], -1) # B, nQry, d

                # classification
                logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
                loss = None

                if mode: # if enable grad, compute loss
                    loss = criterion(logits.view(len(y), -1), y)

            return logits, loss

        # load trained weights
        self.backbone.load_state_dict(self.backbone_state, strict=True)

        #zz = DiffAugment(supp_x, ["color", "offset", "offset_h", "offset_v", "translation", "cutout"], 1., detach=True)
        proto_y, proto_i = unique_indices(supp_y)
        proto_x = supp_x[proto_i]
        zz_i = np.setdiff1d(range(len(supp_x)), proto_i.cpu().numpy())
        zz_x = supp_x[zz_i]
        zz_y = supp_y[zz_i]

        best_lr = 0
        max_acc1 = 0

        if len(zz_y) > 0:
            # eval non-finetuned weights (lr=0)
            logits, _ = single_step(zz_x, False, x=proto_x)
            max_acc1 = accuracy(logits.view(len(zz_y), -1), zz_y, topk=(1,))[0]
            print(f'## *lr = 0: acc1 = {max_acc1}\n')

            for lr in self.lr_lst:
                # create optimizer
                opt = torch.optim.Adam(self.backbone.parameters(),
                                       lr=lr,
                                       betas=(0.9, 0.999),
                                       weight_decay=0.)

                # main loop
                _num_iters = 50
                pbar = tqdm(range(_num_iters)) if is_main_process() else range(_num_iters)
                for i in pbar:
                    opt.zero_grad()
                    z = DiffAugment(proto_x, self.aug_types, self.aug_prob, detach=True)
                    _, loss = single_step(z, True, x=proto_x, y=proto_y)
                    loss.backward()
                    opt.step()
                    if is_main_process():
                        pbar.set_description(f'     << lr = {lr}: loss = {loss.item()}')

                logits, _ = single_step(zz_x, False, x=proto_x)
                acc1 = accuracy(logits.view(len(zz_y), -1), zz_y, topk=(1,))[0]
                print(f'## *lr = {lr}: acc1 = {acc1}\n')

                if acc1 > max_acc1:
                    max_acc1 = acc1
                    best_lr = lr

                # reset backbone state
                self.backbone.load_state_dict(self.backbone_state, strict=True)

        print(f'***Best lr = {best_lr} with acc1 = {max_acc1}.\nStart final loop...\n')

        # create optimizer
        opt = torch.optim.Adam(self.backbone.parameters(),
                               lr=best_lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)

        # main loop
        pbar = tqdm(range(self.num_iters)) if is_main_process() else range(self.num_iters)
        for i in pbar:
            opt.zero_grad()
            z = DiffAugment(supp_x, self.aug_types, self.aug_prob, detach=True)
            _, loss = single_step(z, True, x=supp_x, y=supp_y, y_1hot=supp_y_1hot)
            loss.backward()
            opt.step()
            if is_main_process():
                pbar.set_description(f'    >> lr = {best_lr}: loss = {loss.item()}')

        logits, _ = single_step(qry_x, False, x=supp_x, y_1hot=supp_y_1hot) # supp_x has to pair with y_1hot

        return logits


class ProtoNet_Finetune(ProtoNet):
    def __init__(self, backbone, num_iters=50, lr=5e-2, aug_prob=0.9,
                 aug_types=['color', 'translation']):
        super().__init__(backbone)
        self.num_iters = num_iters
        self.lr = lr
        self.aug_types = aug_types
        self.aug_prob = aug_prob

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        state_dict = self.backbone.state_dict()
        self.backbone_state = deepcopy(state_dict)

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        # reset backbone state
        self.backbone.load_state_dict(self.backbone_state, strict=True)

        if self.lr == 0:
            return super().forward(supp_x, supp_y, x)

        B, nSupp, C, H, W = supp_x.shape
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
        device = x.device

        criterion = nn.CrossEntropyLoss()
        supp_x = supp_x.view(-1, C, H, W)
        x = x.view(-1, C, H, W)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp
        supp_y = supp_y.view(-1)

        # create optimizer
        opt = torch.optim.Adam(self.backbone.parameters(),
                               lr=self.lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)

        def single_step(z, mode=True):
            '''
            z = Aug(supp_x) or x
            '''
            with torch.set_grad_enabled(mode):
                # recalculate prototypes from supp_x with updated backbone
                supp_f = self.backbone.forward(supp_x)
                supp_f = supp_f.view(B, nSupp, -1)
                prototypes = torch.bmm(supp_y_1hot.float(), supp_f) # B, nC, d
                prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0

                # compute feature for z
                feat = self.backbone.forward(z)
                feat = feat.view(B, z.shape[0], -1) # B, nQry, d

                # classification
                logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
                loss = None

                if mode: # if enable grad, compute loss
                    loss = criterion(logits.view(B*nSupp, -1), supp_y)

            return logits, loss

        # main loop
        pbar = tqdm(range(self.num_iters)) if is_main_process() else range(self.num_iters)
        for i in pbar:
            opt.zero_grad()
            z = DiffAugment(supp_x, self.aug_types, self.aug_prob, detach=True)
            _, loss = single_step(z, True)
            loss.backward()
            opt.step()
            if is_main_process():
                pbar.set_description(f'lr{self.lr}, nSupp{nSupp}, nQry{x.shape[0]}: loss = {loss.item()}')

        logits, _ = single_step(x, False)
        return logits


class ProtoNet_AdaTok(ProtoNet):
    def __init__(self, backbone, num_adapters=1, num_iters=50, lr=5e-2, momentum=0.9, weight_decay=0.):
        super().__init__(backbone)
        self.num_adapters = num_adapters
        self.num_iters = num_iters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        B, nSupp, C, H, W = supp_x.shape
        nQry = x.shape[1]
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
        device = x.device

        criterion = nn.CrossEntropyLoss()
        supp_x = supp_x.view(-1, C, H, W)
        x = x.view(-1, C, H, W)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp
        supp_y = supp_y.view(-1)

        # prepare adapter tokens
        ada_tokens = torch.zeros(1, self.num_adapters, self.backbone.embed_dim, device=device)
        trunc_normal_(ada_tokens, std=.02)
        ada_tokens = ada_tokens.detach().requires_grad_()
        #optimizer = torch.optim.SGD([ada_tokens],
        optimizer = torch.optim.Adadelta([ada_tokens],
                                     lr=self.lr,
                                     #momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        def single_step(mode=True):
            with torch.set_grad_enabled(mode):
                supp_f = self.backbone.forward(supp_x, ada_tokens)
                supp_f = supp_f.view(B, nSupp, -1)

                # B, nC, nSupp x B, nSupp, d = B, nC, d
                prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
                prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0

            if mode == False: # no grad
                feat = self.backbone.forward(x, ada_tokens)
                feat = feat.view(B, nQry, -1) # B, nQry, d

                logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
                loss = None
            else:
                with torch.enable_grad():
                    logits = self.cos_classifier(prototypes, supp_f) # B, nQry, nC
                    loss = criterion(logits.view(B*nSupp, -1), supp_y)

            return logits, loss

        pbar = tqdm(range(self.num_iters)) if is_main_process() else range(self.num_iters)
        for i in pbar:
            optimizer.zero_grad()
            _, loss = single_step(True)
            loss.backward()
            optimizer.step()
            if is_main_process():
                pbar.set_description(f'loss = {loss.item()}')

        logits, _ = single_step(False)
        return logits


class ProtoNet_AdaTok_EntMin(ProtoNet):
    def __init__(self, backbone, num_adapters=1, num_iters=50, lr=5e-3, momentum=0.9, weight_decay=0.):
        super().__init__(backbone)
        self.num_adapters = num_adapters
        self.num_iters = num_iters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        B, nSupp, C, H, W = supp_x.shape
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
        device = x.device

        criterion = entropy_loss
        supp_x = supp_x.view(-1, C, H, W)
        x = x.view(-1, C, H, W)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp

        # adapter tokens
        ada_tokens = torch.zeros(1, self.num_adapters, self.backbone.embed_dim, device=device)
        trunc_normal_(ada_tokens, std=.02)
        ada_tokens = ada_tokens.detach().requires_grad_()
        optimizer = torch.optim.SGD([ada_tokens],
                                     lr=self.lr,
                                     momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        def single_step(mode=True):
            with torch.set_grad_enabled(mode):
                supp_f = self.backbone.forward(supp_x, ada_tokens)
                supp_f = supp_f.view(B, nSupp, -1)

                # B, nC, nSupp x B, nSupp, d = B, nC, d
                prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
                prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0

                feat = self.backbone.forward(x, ada_tokens)
                feat = feat.view(B, x.shape[1], -1) # B, nQry, d

                logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
                loss = criterion(logits.view(-1, num_classes))

            return logits, loss

        pbar = tqdm(range(self.num_iters)) if is_main_process() else range(self.num_iters)
        for i in pbar:
            optimizer.zero_grad()
            _, loss = single_step(True)
            loss.backward()
            optimizer.step()
            if is_main_process():
                pbar.set_description(f'loss = {loss.item()}')

        logits, _ = single_step(False)
        return logits
