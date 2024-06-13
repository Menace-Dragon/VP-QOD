# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from my_equation import *
import global_placeholder

# TODO: add loss evaluator for SSD
@HEADS.register_module()
class SSDHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Default: 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Dictionary to construct and config activation layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 stacked_convs=0,
                 feat_channels=256,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Xavier',
                     layer='Conv2d',
                     distribution='uniform',
                     bias=0)):
        super(AnchorHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.cls_out_channels = num_classes + 1  # add background class
        self.prior_generator = build_prior_generator(anchor_generator)

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors

        self._init_layers()

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Number of base_anchors on each point of each level.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'please use "num_base_priors" instead')
        return self.num_base_priors

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # TODO: Use registry to choose ConvModule type
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule

        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            cls_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            reg_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * 4,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        if hasattr(self,'in_num') and self.in_num == 6:
            feats = [feats[0], feats[1], feats[2], feats[3], feats[4], feats[5]]
        elif hasattr(self,'in_num'):
            raise NotImplementedError
        else:
            self.in_num = 6
            feats = [feats[0], feats[1], feats[2], feats[3], feats[4], feats[5]]
        
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if self.num_base_priors == [4, 6, 6, 6, 4, 4]:
            # 说明是ssd
            level_counter = [5776, 2166, 600, 150, 36, 4]  # 这个只出现在SSD300里，否则有问题
            
        elif self.num_base_priors == [6, 6, 6, 6, 6, 6]:
            # 说明是ssdlite
            level_counter = [6*20*20, 6*10*10, 6*5*5, 6*3*3, 6*2*2, 6*1*1]  # 这个只出现在SSDlite里，否则有问题
        else:
            raise NotImplementedError
        # 实现level上的标记  编码第一个level为0；第二个level为1；第三个level为2
        level_mapping = []
        for it, temp in enumerate(level_counter):
            temp_tensor = torch.zeros(temp,dtype=torch.uint8,device=labels.device) + it
            level_mapping.append(temp_tensor)
        level_mapping = torch.cat(level_mapping)
            
            
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(
            as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, topk_inds = loss_cls_all[neg_inds].topk(num_neg_samples)

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            reduction='none')  # 这里的处理（包括后面的），等价于这里的mean
        
        loss_bbox = loss_bbox.sum(dim=1)
                
        loss_cls_pos = loss_cls_all[pos_inds]
        loss_cls_neg = topk_loss_cls_neg
        pos_level_mapping = level_mapping[pos_inds]
        neg_level_mapping = level_mapping[neg_inds][topk_inds]
        
        if global_placeholder.mybuff_flag:
            
            # # -----加上最小量化误差
            # qloss_flag = global_placeholder.qloss_flag
            # q_loss = torch.tensor(0)
            # # if qloss_flag:
            # q_loss_sum = []
            # for name, module in self.named_modules():
            #     if hasattr(module, 'compute_qloss') and module.compute_qloss:
            #     # if 'fake_quant' in name.split('.')[-1]:
            #         # 说明是act quantizer
            #         q_loss_sum.append(module.quantization_loss)
                
            # q_loss = sum(q_loss_sum) / len(q_loss_sum)
            
            conf_values = F.softmax(cls_score, dim=1)
            num_classes = conf_values.shape[-1]
            pos_gtconf_idx = F.one_hot(labels, num_classes=num_classes)
            pos_gtconf_values, _ = (conf_values * pos_gtconf_idx).max(dim=1)  # TODO 想到一件事，是不是不能用GT来筛选正样本的结果？因为其实正样本也有错误的东西，所以就是得错
            pos_gtconf_values = pos_gtconf_values[pos_inds]
            
            pos_conf_values, pos_conf_values_idx = conf_values.max(dim=1)  # TODO 好像真的是这个问题，正样本本来就是得对应到max的那个，不能经过GT筛选
            pos_conf_values = pos_conf_values[pos_inds]
            
            abs_bbox_pred = self.bbox_coder.decode(anchor, bbox_pred) # 直接解读
            abs_bbox_targets = self.bbox_coder.decode(anchor, bbox_targets) # 直接解读
            pos_ious = bbox_overlaps(abs_bbox_pred[pos_inds], abs_bbox_targets[pos_inds], is_aligned=True)  # 这玩意得是ltrb坐标，好像就已经是了？？？
            if global_placeholder.mybuff_flag == 1:
                
                level_slicer = [6, 0, 0]  # [0]为level数；[1]为weight的个数；[2]为单level下的act个数
                level_cls_factors = []
                level_reg_factors = []
                # level_obj_factors = []
                qloss_flag = global_placeholder.qloss_flag
                
                if False:
                    q_loss_total = []
                    cls_branch = []
                    reg_branch = []
                    # obj_branch = []
                    for name, module in self.named_modules():
                        if hasattr(module, 'compute_qloss') and module.compute_qloss:
                            # 挑出来量化器
                            
                            if 'cls' in name:
                                # 说明是cls分支的量化器
                                # cls_branch.append(module.quantization_loss)
                                if 'post_act' in name:
                                    # 说明是act量化器
                                    cls_branch.append([name, module.scale * 1.])
                                else:
                                    cls_branch.append([name, module.scale * 2.])
                                    
                            elif 'reg' in name:
                                # 说明是reg分支的量化器
                                # reg_branch.append(module.quantization_loss)
                                if 'post_act' in name:
                                    # 说明是act量化器
                                    reg_branch.append([name, module.scale * 1.])
                                else:
                                    reg_branch.append([name, module.scale * 2.])        
                                    
                            # elif 'obj' in name:
                            #     # 说明是reg分支的量化器
                            #     # reg_branch.append(module.quantization_loss)
                            #     if 'post_act' in name:
                            #         # 说明是act量化器
                            #         obj_branch.append([name, module.scale * 1.])
                            #     else:
                            #         obj_branch.append([name, module.scale * 2.])
                                
                            q_loss_total.append([name, module.scale * 1])

                    # NOTE 不需要加item_post_act_quant的
                    # 由于obj branch 的特殊性，所以得加上reg_branch分支的共用东西
                    # obj_branch = obj_branch + reg_branch[3:]
                    
                    # if level_slicer[0] != 3:
                    #     raise NotImplementedError
                    
                    for it in range(level_slicer[0]):
                        cls_summation = 0
                        reg_summation = 0
                        # obj_summation = 0
                        tmp_cls_infos = [cls_branch[it]]
                        for info in tmp_cls_infos:
                            cls_summation += info[1]
                            
                        tmp_reg_infos = [reg_branch[it]]
                        for info in tmp_reg_infos:
                            reg_summation += info[1]
                            
                        # tmp_obj_infos = [obj_branch[it]] + obj_branch[3+it*2:3+(it+1)*2] + obj_branch[9+it*2:9+(it+1)*2]
                        # for info in tmp_obj_infos:
                        #     obj_summation += info[1]
                    
                        level_cls_factors.append(cls_summation)
                        level_reg_factors.append(reg_summation)
                        # level_obj_factors.append(obj_summation)
                
                
                if len(level_cls_factors) + len(level_reg_factors) == 0:
                    # 必须让list内有level个空list
                    level_cls_factors = [torch.tensor(1)]
                    level_reg_factors = [torch.tensor(1)]
        

                # # NOTE 编码第一个level为0 第二个为1 ......   
                # for it, [cls_branch_factor, reg_branch_factor] in enumerate(zip(level_cls_factors, level_reg_factors)):
                #     single_level_masks = (level_mapping == it)
                #     single_pos_level_masks = (pos_level_mapping == it)
                #     single_neg_level_masks = (neg_level_mapping == it)
                    
                #     cls_trade_off = (cls_branch_factor / (cls_branch_factor + reg_branch_factor)).detach() * 2
                #     reg_trade_off = (reg_branch_factor / (cls_branch_factor + reg_branch_factor)).detach() * 2
                #     # obj_trade_off = (obj_branch_factor / (cls_branch_factor + reg_branch_factor + obj_branch_factor)).detach() * 3
                    
                #     loss_cls_pos[single_pos_level_masks] = cls_trade_off * loss_cls_pos[single_pos_level_masks]
                #     loss_cls_neg[single_neg_level_masks] = cls_trade_off * loss_cls_neg[single_neg_level_masks]
                #     loss_bbox[single_level_masks] = reg_trade_off * loss_bbox[single_level_masks]
                
                loss_bbox[pos_inds], loss_cls_pos = HQOD_loss(loss_bbox[pos_inds], loss_cls_pos, conf_values[pos_inds], pos_gtconf_values, pos_conf_values, pos_ious, [torch.tensor(1)], [torch.tensor(1)], torch.tensor(1.))
                
                # loss_bbox[pos_inds], loss_cls_pos = HQOD_loss(loss_bbox[pos_inds], loss_cls_pos, conf_values[pos_inds], pos_gtconf_values, pos_conf_values, pos_ious, torch.tensor(1.))
            elif global_placeholder.mybuff_flag == 2:
                # 对比HarDet
                loss_bbox[pos_inds], loss_cls_pos = HarDet_loss(loss_bbox[pos_inds], loss_cls_pos, conf_values[pos_inds], pos_gtconf_values, pos_conf_values, pos_ious, torch.tensor(1.))
                
                # loss_bbox[pos_inds], loss_cls_pos = HarDet_loss(loss_bbox[pos_inds], loss_cls_pos, conf_values[pos_inds], pos_gtconf_values, pos_conf_values, pos_ious, torch.tensor(1.))
            else:
                raise NotImplementedError
        
        
        
        loss_cls_pos_sum = loss_cls_pos.sum()
        loss_cls_neg_sum = loss_cls_neg.sum()
        
        loss_cls = (loss_cls_pos_sum + loss_cls_neg_sum) / num_total_samples
        loss_bbox_reduced = loss_bbox.sum() / num_total_samples
        return loss_cls[None], loss_bbox_reduced


    # def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
    #                 bbox_targets, bbox_weights, num_total_samples):
    #     """Compute loss of a single image.

    #     Args:
    #         cls_score (Tensor): Box scores for eachimage
    #             Has shape (num_total_anchors, num_classes).
    #         bbox_pred (Tensor): Box energies / deltas for each image
    #             level with shape (num_total_anchors, 4).
    #         anchors (Tensor): Box reference for each scale level with shape
    #             (num_total_anchors, 4).
    #         labels (Tensor): Labels of each anchors with shape
    #             (num_total_anchors,).
    #         label_weights (Tensor): Label weights of each anchor with shape
    #             (num_total_anchors,)
    #         bbox_targets (Tensor): BBox regression targets of each anchor
    #             weight shape (num_total_anchors, 4).
    #         bbox_weights (Tensor): BBox regression loss weights of each anchor
    #             with shape (num_total_anchors, 4).
    #         num_total_samples (int): If sampling, num total samples equal to
    #             the number of total anchors; Otherwise, it is the number of
    #             positive anchors.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """

    #     loss_cls_all = F.cross_entropy(
    #         cls_score, labels, reduction='none') * label_weights
    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
    #         as_tuple=False).reshape(-1)
    #     neg_inds = (labels == self.num_classes).nonzero(
    #         as_tuple=False).view(-1)

    #     num_pos_samples = pos_inds.size(0)
    #     num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
    #     if num_neg_samples > neg_inds.size(0):
    #         num_neg_samples = neg_inds.size(0)
    #     topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
    #     loss_cls_pos = loss_cls_all[pos_inds].sum()
    #     loss_cls_neg = topk_loss_cls_neg.sum()
    #     loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

    #     if self.reg_decoded_bbox:
    #         # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
    #         # is applied directly on the decoded bounding boxes, it
    #         # decodes the already encoded coordinates to absolute format.
    #         bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

    #     loss_bbox = smooth_l1_loss(
    #         bbox_pred,
    #         bbox_targets,
    #         bbox_weights,
    #         beta=self.train_cfg.smoothl1_beta,
    #         avg_factor=num_total_samples)
    #     return loss_cls[None], loss_bbox


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))
        # 这个multi apply的是batch，每张图片进行
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
