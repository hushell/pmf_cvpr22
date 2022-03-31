import math
import sys
import warnings
from typing import Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils.deit_util as utils
from utils import AverageMeter, to_device


def train_one_epoch(data_loader: Iterable,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    loss_scaler = None,
                    fp16: bool = False,
                    max_norm: float = 0, # clip_grad
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    writer: Optional[SummaryWriter] = None,
                    set_training_mode=True):

    global_step = epoch * len(data_loader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    model.train(set_training_mode)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        # forward
        with torch.cuda.amp.autocast(fp16):
            output = model(SupportTensor, SupportLabel, x)

        output = output.view(x.shape[0] * x.shape[1], -1)
        y = y.view(-1)
        loss = criterion(output, y)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if fp16:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

        # tensorboard
        if utils.is_main_process() and global_step % print_freq == 0:
            writer.add_scalar("train/loss", scalar_value=loss_value, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=lr, global_step=global_step)

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(data_loaders, model, criterion, device, seed=None, ep=None):
    if isinstance(data_loaders, dict):
        test_stats_lst = {}
        test_stats_glb = {}

        for j, (source, data_loader) in enumerate(data_loaders.items()):
            print(f'* Evaluating {source}:')
            seed_j = seed + j if seed else None
            test_stats = _evaluate(data_loader, model, criterion, device, seed_j)
            test_stats_lst[source] = test_stats
            test_stats_glb[source] = test_stats['acc1']

        # apart from individual's acc1, accumulate metrics over all domains to compute mean
        for k in test_stats_lst[source].keys():
            test_stats_glb[k] = torch.tensor([test_stats[k] for test_stats in test_stats_lst.values()]).mean().item()

        return test_stats_glb
    elif isinstance(data_loaders, torch.utils.data.DataLoader): # when args.eval = True
        return _evaluate(data_loaders, model, criterion, device, seed, ep)
    else:
        warnings.warn(f'The structure of {data_loaders} is not recognizable.')
        return _evaluate(data_loaders, model, criterion, device, seed)


@torch.no_grad()
def _evaluate(data_loader, model, criterion, device, seed=None, ep=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if seed is not None:
        data_loader.generator.manual_seed(seed)

    for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if ep is not None:
            if ii > ep:
                break

        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch

        # compute output
        with torch.cuda.amp.autocast():
            output = model(SupportTensor, SupportLabel, x)

        output = output.view(x.shape[0] * x.shape[1], -1)
        y = y.view(-1)
        loss = criterion(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))

        batch_size = x.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std

    return ret_dict
