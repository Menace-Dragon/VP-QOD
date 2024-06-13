# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, load_checkpoint,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)
from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)
from mmdet.apis.test import single_gpu_test
from torch.fx import Tracer
import torch.fx as fx
from mqb_general_process import *
from mqbench.utils.state import *

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from copy import deepcopy

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')

def calibrate(model, data_loader, cali_num):  # TODO 这里给修改成达到一定次数后就break；然后只用eval_data
    model.eval()
    if isinstance(data_loader, list):
        dataset = data_loader[0].dataset
        data_loader = data_loader[0]
    else:
        dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    
    rank, world_size = get_dist_info()
    
    if rank == 0:
        prog_bar = mmcv.ProgressBar(cali_num)
    # prog_bar = mmcv.ProgressBar(cali_num)
    steped_num = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'] = data['img'].data
            data['img_metas'] = data['img_metas'].data
            data.pop('gt_bboxes')
            data.pop('gt_labels')
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        steped_num += batch_size * world_size
        dist.barrier()
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
        if steped_num >= cali_num:
            logger.info(f'Truly calibrate num {steped_num}')
            break


def qat_detector(model,
                   dataset,
                   cfg,
                   quant_config,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    

    data_loaders = [build_dataloader(ds, **train_loader_cfg, pin_memory=True) for ds in dataset]
    
    # cali loader的设置
    cali_loader_cfg = deepcopy(train_loader_cfg)
    cali_loader_cfg['samples_per_gpu'] = 1
    cali_loader_cfg['workers_per_gpu'] = 1
    cali_data_loader = [build_dataloader(ds, **cali_loader_cfg, pin_memory=True) for ds in dataset][0]
    
    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    elif cfg.load_from:  # 这个应该可以解决所谓的quant baseline问题
        checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')  # TODO 测一下这样的evaluate对不对
    else:
        checkpoint = {}
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset[0].CLASSES
        
        
    # print('\nGet FakeQuant model\n')
    # model.backbone = get_quantize_model(model.backbone, quant_config, cfg.trace_config.backbone_detail)  # QAT时，这个需要eval还是train
    # model.neck = get_quantize_model(model.neck, quant_config, cfg.trace_config.neck_detail)  # QAT时，这个需要eval还是train
    # temp = get_quantize_model(model.bbox_head, quant_config, cfg.trace_config.bbox_head_detail)  # QAT时，这个需要eval还是train
    # model.bbox_head.forward = temp.forward  # 太傻蛋勒
    # model.bbox_head = temp
    model.train()  # prepare 前一定得是train模式
    model = make_qmodel_for_mmd(model, quant_config, cfg.trace_config)
    
    if not hasattr(cfg, 'tune_from'):
        cfg.tune_from = False
    if cfg.tune_from:  # 加载进行tune
        _ = load_checkpoint(model, cfg.tune_from, map_location='cpu')
    
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(  # 细节不给看
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # cali_data_loader = deepcopy(val_dataloader)  # TODO 这里改成train的
    if not cfg.resume_from and not cfg.tune_from:
        print('\nCalibrating\n')
        enable_calibration(model.module.backbone)
        enable_calibration(model.module.neck)
        model_general_architecture = cfg.trace_config.get('model_general_architecture', None)
        if model_general_architecture == 'FasterRCNN':
            enable_calibration(model.module.rpn_head)
            enable_calibration(model.module.roi_head.bbox_head)
        else:
            enable_calibration(model.module.bbox_head)
        
        calibrate(model, cali_data_loader
                #   , 1
                  , quant_config.quantize.cali_batchnum
                  )  # NOTE 训练集的data确实全带container
    # 清除cali loader，释放内存
    del cali_data_loader
    
    enable_quantization(model.module.backbone)
    enable_quantization(model.module.neck)
    model_general_architecture = cfg.trace_config.get('model_general_architecture', None)
    if model_general_architecture == 'FasterRCNN':
        enable_quantization(model.module.rpn_head)
        enable_quantization(model.module.roi_head.bbox_head)
    else:
        enable_quantization(model.module.bbox_head)
        
    
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    runner.run(data_loaders, cfg.workflow)
    
def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(  # 细节不给看
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
