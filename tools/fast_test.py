# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, fast_compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
from mqb_general_process import make_qmodel_for_mmd, prepocess
from mqbench.utils.state import *
import global_placeholder
from mqb_general_process import *
from mmcv.image import tensor2imgs
import numpy as np
from mmdet.core.visualization import imshow_det_bboxes

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('pkl_result_path', help='pkl result file')
    # parser.add_argument('quant_config', default=None, help='quant config file path')
    # parser.add_argument('--aqd-mode', type=int, default=0, help='when bigger than 0 , it means switch on aqd, and equals the neck output level num')
    # parser.add_argument('--quantize', 
        # action='store_true', help='quant flag')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = fast_compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)
    set_random_seed(args.seed)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # if 'pretrained' in cfg.model:
    #     cfg.model.pretrained = None
    # elif 'init_cfg' in cfg.model.backbone:
    #     cfg.model.backbone.init_cfg = None

    # if cfg.model.get('neck'):
    #     if isinstance(cfg.model.neck, list):
    #         for neck_cfg in cfg.model.neck:
    #             if neck_cfg.get('rfp_backbone'):
    #                 if neck_cfg.rfp_backbone.get('pretrained'):
    #                     neck_cfg.rfp_backbone.pretrained = None
    #     elif cfg.model.neck.get('rfp_backbone'):
    #         if cfg.model.neck.rfp_backbone.get('pretrained'):
    #             cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    # cfg.model.train_cfg = None
    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # global setting
    # if args.aqd_mode != 0:
    #     global_placeholder.modify_AQD_mode(args.aqd_mode)
    
    # 如果是QAT模型，那么就需要提前定义好量化模型结构
    # if args.quantize:
    #     quant_config = prepocess(args.quant_config)
    #     # copy_config_file(args.quant_config, cfg.work_dir)
    #     logger.info(quant_config)
    #     global_placeholder.modify_quant_bit(quant_config.extra_prepare_dict.extra_qconfig_dict.w_qscheme.bit)
    #     global_placeholder.modify_quant_algorithm(quant_config.quantize.quant_algorithm)
    #     global_placeholder.modify_buff_flag(quant_config.training.my_buff_flag)
    #     global_placeholder.modify_qloss_flag(quant_config.training.qloss_flag)
    #     global_placeholder.modify_fold_bn_flag(quant_config.training.fold_bn_flag)
        
    #     model.train()
    #     model = make_qmodel_for_mmd(model, quant_config, cfg.trace_config)
        
    # else:
    #     if 'HQOD' in cfg.work_dir:
    #         logger.info("插播!!!!直接启用harmony！！")
    #         logger.info("插播!!!!直接启用harmony！！")
    #         logger.info("插播!!!!直接启用harmony！！\n")
    #         global_placeholder.modify_buff_flag(1) # 为mypro
    #     elif 'HarDet' in cfg.work_dir:
    #         logger.info("插播!!!!直接启用harmony！！")
    #         logger.info("插播!!!!直接启用harmony！！")
    #         logger.info("插播!!!!直接启用harmony！！\n")
    #         global_placeholder.modify_buff_flag(2) # 为hardet
            
    # # 检查 num levels 一致性
    # if global_placeholder.aqd_mode != 0 and model.neck.num_outs != global_placeholder.aqd_mode:
    #     # 说明 num levels给的不对
    #     raise  ValueError(f'num levels给的不对! aqd_mode={global_placeholder.aqd_mode} 而 neck.num_outs={model.neck.num_outs}')
    
    # init rfnext if 'RFSearchHook' is defined in cfg
    # rfnext_init_model(model, cfg=cfg)
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is None and cfg.get('device', None) == 'npu':
    #     fp16_cfg = dict(loss_scale='dynamic')
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)
    # # old versions did not save class info in checkpoints, this walkaround is
    # # for backward compatibility
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES

    
    # if args.quantize:
    #     enable_quantization(model.backbone)
    #     enable_quantization(model.neck)
    #     model_general_architecture = cfg.trace_config.get('model_general_architecture', None)
    #     if model_general_architecture == 'FasterRCNN':
    #         enable_quantization(model.rpn_head)
    #         enable_quantization(model.roi_head.bbox_head)
    #     else:
    #         enable_quantization(model.bbox_head)

    # model.eval()
    # if not distributed:
    #     model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    #     outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                               args.show_score_thr)
    # else:
    #     model = build_ddp(
    #         model,
    #         cfg.device,
    #         device_ids=[int(os.environ['LOCAL_RANK'])],
    #         broadcast_buffers=False)

    #     # In multi_gpu_test, if tmpdir is None, some tesnors
    #     # will init on cuda by default, and no device choice supported.
    #     # Init a tmpdir to avoid error on npu here.
    #     if cfg.device == 'npu' and args.tmpdir is None:
    #         args.tmpdir = './npu_tmpdir'

    #     outputs = multi_gpu_test(
    #         model, data_loader, args.tmpdir, args.gpu_collect
    #         or cfg.evaluation.get('gpu_collect', False))

    # TODO 加载pkl 文件
    outputs = mmcv.load(args.pkl_result_path)
    print(f'Loaded pkl result : {args.pkl_result_path}')
    
    # # create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(os.path.join(args.show_dir, 'imgs')))
    single_gpu_draw(outputs, data_loader, out_dir=args.show_dir)
    rank, _ = get_dist_info()
    if rank == 0:
        # if args.out:
        #     print(f'\nwriting results to {args.out}')
        #     mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False





def single_gpu_draw(results,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    # model.eval()
    # results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # with torch.no_grad():
        #     result = model(return_loss=False, rescale=True, **data)
        
        batch_size = data_loader.batch_size
        
        result = results[i*batch_size:(i+1)*batch_size]
        

        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            
            selected_imgs = None
            selected_imgs = ['000062.jpg', '000069.jpg', '000074.jpg', '000108.jpg', '000852.jpg'
                             ,'001285.jpg','001745.jpg','006193.jpg']

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                if selected_imgs is not None:
                    # 说明是筛选模式
                    matched_flag = has_matched_img(img_meta, selected_imgs)
                    if not matched_flag:
                        # 跳过循环
                        continue
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                show_result(
                    dataset,
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)


        for _ in range(batch_size):
            prog_bar.update()
    return 


def has_matched_img(img_meta, selected_imgs):
    for img_key in selected_imgs:
        if img_key in img_meta['filename']:
            return True
    return False
    
    
def show_result(dataset,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor or tuple): The results to draw over `img`
            bbox_result or (bbox_result, segm_result).
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
            The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
            The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
            Color of masks. The tuple of color should be in BGR order.
            Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=dataset.CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    if not (show or out_file):
        return img



if __name__ == '__main__':
    main()
