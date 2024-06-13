# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from my_equation import *
import global_placeholder

@HEADS.register_module()
class ATSSHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 pred_kernel_size=3,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox=True,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(ATSSHead, self).__init__(
            num_classes,
            in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    # def forward(self, feats):
    #     """Forward features from the upstream network.

    #     Args:
    #         feats (tuple[Tensor]): Features from the upstream network, each is
    #             a 4D-tensor.

    #     Returns:
    #         tuple: Usually a tuple of classification scores and bbox prediction
    #             cls_scores (list[Tensor]): Classification scores for all scale
    #                 levels, each is a 4D-tensor, the channels number is
    #                 num_anchors * num_classes.
    #             bbox_preds (list[Tensor]): Box energies / deltas for all scale
    #                 levels, each is a 4D-tensor, the channels number is
    #                 num_anchors * 4.
    #     """
    #     return multi_apply(self.forward_single, feats, self.scales)
    
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        if hasattr(self,'in_num') and self.in_num == 5:
            feats = [feats[0], feats[1], feats[2], feats[3], feats[4]]
        elif hasattr(self,'in_num'):
            raise NotImplementedError
        else:
            self.in_num = 5
            feats = [feats[0], feats[1], feats[2], feats[3], feats[4]]
        
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.atss_cls(cls_feat)
            # we just follow atss, not apply exp in bbox_pred
            bbox_pred = scale(self.atss_reg(reg_feat)).float()
            centerness = self.atss_centerness(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
            
        return cls_scores, bbox_preds, centernesses
    

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, cls_branch_factor, reg_branch_factor, centerness_branch_factor, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, reduction_override='none')
        loss_cls = loss_cls.sum(dim=1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                reduction_override='none')
            # loss_bbox = loss_bbox.sum(dim=1)
            

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                reduction_override='none')
            # loss_centerness = loss_centerness.sum(dim=1)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        if global_placeholder.mybuff_flag and pos_inds.numel():
            # 只在pos_indx有的情况下执行

            conf_values = torch.sigmoid(cls_score)  # TODO 注意，这里是因为Retina要sigmoid！
            
            num_classes = conf_values.shape[-1]
            if num_classes == 81 or num_classes == 21:
                # 说明默认backbgound为81类 21类
                pos_gtconf_idx = F.one_hot(labels, num_classes=num_classes)
            else:
                # 说明没有显式给出类
                
                num_classes = num_classes + 1
                pos_gtconf_idx = F.one_hot(labels, num_classes=num_classes)
                pos_gtconf_idx = pos_gtconf_idx[:, :-1]# 剔除最后一个背景类
                
            pos_gtconf_values, _ = (conf_values * pos_gtconf_idx).max(dim=1)  # TODO 想到一件事，是不是不能用GT来筛选正样本的结果？因为其实正样本也有错误的东西，所以就是得错
            pos_gtconf_values = pos_gtconf_values[pos_inds]
            
            pos_conf_values, pos_conf_values_idx = conf_values.max(dim=1)  # TODO 好像真的是这个问题，正样本本来就是得对应到max的那个，不能经过GT筛选
            pos_conf_values = pos_conf_values[pos_inds]

            try:
                pos_ious = bbox_overlaps(pos_decode_bbox_pred, pos_bbox_targets, is_aligned=True)  # 这玩意得是ltrb坐标，好像就已经是了？？？
            except IndexError as e:
                print(f"inds.numel(): {pos_inds.numel()} \ninds.max(): {pos_inds.max()} \ninds.shape: {pos_inds.shape}")
                print(e)
                exit()
            if global_placeholder.mybuff_flag == 1:
                # loss_bbox, loss_cls = loss_bbox * (1 + reg_branch_factor), loss_cls * (1 + cls_branch_factor)  # 确实这玩意应该是全局？？？
                # loss_bbox, loss_cls = loss_bbox * (1 + cls_branch_factor), loss_cls * (1 + reg_branch_factor)  # 确实这玩意应该是全局？？？
                # conf_values = F.softmax(cls_score, dim=1)
                        
                # cls_trade_off = (cls_branch_factor / (cls_branch_factor + reg_branch_factor + centerness_branch_factor)).detach() * 3
                # reg_trade_off = (reg_branch_factor / (cls_branch_factor + reg_branch_factor + centerness_branch_factor)).detach() * 3
                # centerness_trade_off = (centerness_branch_factor / (cls_branch_factor + reg_branch_factor + centerness_branch_factor)).detach() * 3
                
                # # loss_cls = cls_trade_off * loss_cls
                # # loss_bbox = reg_trade_off * loss_bbox
                # loss_cls = cls_trade_off * loss_cls
                # loss_bbox = reg_trade_off * loss_bbox
                # loss_centerness = centerness_trade_off * loss_centerness
                
                loss_bbox, loss_cls[pos_inds] = HQOD_loss(loss_bbox, loss_cls[pos_inds], conf_values[pos_inds], pos_gtconf_values, pos_conf_values, pos_ious, cls_branch_factor, reg_branch_factor, torch.tensor(1.))
            elif global_placeholder.mybuff_flag == 2:
                loss_bbox, loss_cls[pos_inds] = HarDet_loss(loss_bbox, loss_cls[pos_inds], conf_values[pos_inds], pos_gtconf_values, pos_conf_values, pos_ious, torch.tensor(1.))
            
        


        loss_cls = loss_cls.sum() / num_total_samples
        loss_bbox = loss_bbox.sum() / 1.0
        loss_centerness = loss_centerness.sum() / num_total_samples



        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    # def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
    #                 label_weights, bbox_targets, level_cls_factor, level_reg_factor, level_centerness_factor, num_total_samples):
    #     """Compute loss of a single scale level.

    #     Args:
    #         cls_score (Tensor): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W).
    #         bbox_pred (Tensor): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W).
    #         anchors (Tensor): Box reference for each scale level with shape
    #             (N, num_total_anchors, 4).
    #         labels (Tensor): Labels of each anchors with shape
    #             (N, num_total_anchors).
    #         label_weights (Tensor): Label weights of each anchor with shape
    #             (N, num_total_anchors)
    #         bbox_targets (Tensor): BBox regression targets of each anchor
    #             weight shape (N, num_total_anchors, 4).
    #         num_total_samples (int): Number os positive samples that is
    #             reduced over all GPUs.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """

    #     anchors = anchors.reshape(-1, 4)
    #     cls_score = cls_score.permute(0, 2, 3, 1).reshape(
    #         -1, self.cls_out_channels).contiguous()
    #     bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #     centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     labels = labels.reshape(-1)
    #     label_weights = label_weights.reshape(-1)

    #     # classification loss
    #     loss_cls = self.loss_cls(
    #         cls_score, labels, label_weights, avg_factor=num_total_samples)

    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((labels >= 0)
    #                 & (labels < bg_class_ind)).nonzero().squeeze(1)

    #     if len(pos_inds) > 0:
    #         pos_bbox_targets = bbox_targets[pos_inds]
    #         pos_bbox_pred = bbox_pred[pos_inds]
    #         pos_anchors = anchors[pos_inds]
    #         pos_centerness = centerness[pos_inds]

    #         centerness_targets = self.centerness_target(
    #             pos_anchors, pos_bbox_targets)
    #         pos_decode_bbox_pred = self.bbox_coder.decode(
    #             pos_anchors, pos_bbox_pred)

    #         # regression loss
    #         loss_bbox = self.loss_bbox(
    #             pos_decode_bbox_pred,
    #             pos_bbox_targets,
    #             weight=centerness_targets,
    #             avg_factor=1.0)

    #         # centerness loss
    #         loss_centerness = self.loss_centerness(
    #             pos_centerness,
    #             centerness_targets,
    #             avg_factor=num_total_samples)

    #     else:
    #         loss_bbox = bbox_pred.sum() * 0
    #         loss_centerness = centerness.sum() * 0
    #         centerness_targets = bbox_targets.new_tensor(0.)

    #     return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
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
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)


        level_slicer = [len(labels_list), 5, 4]  # [0]为level数；[1]为weight的个数；[2]为单level下的act个数
        level_cls_factors = []
        level_reg_factors = []
        level_centerness_factors = []
        
        qloss_flag = global_placeholder.qloss_flag

        if False:
            q_loss_total = []
            cls_branch = []
            reg_branch = []
            centerness_branch = []
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
                            
                    elif 'centerness' in name:
                        # 说明是reg分支的量化器
                        # reg_branch.append(module.quantization_loss)
                        if 'post_act' in name:
                            # 说明是act量化器
                            centerness_branch.append([name, module.scale * 1.])
                        else:
                            centerness_branch.append([name, module.scale * 2.])
                        
                    q_loss_total.append([name, module.scale * 1])

            # NOTE 不需要加item_post_act_quant的
            # 由于centerness branch 的特殊性，所以得加上reg_branch分支的共用东西
            centerness_branch = centerness_branch + reg_branch[1:]
            
            if level_slicer[0] != 5:
                raise NotImplementedError
            
            for it in range(level_slicer[0]):
                cls_summation = 0
                reg_summation = 0
                centerness_summation = 0
                # tmp_cls_infos = cls_branch[0:level_slicer[1]] + cls_branch[level_slicer[1]+level_slicer[2]*it:level_slicer[1]+level_slicer[2]*(it+1)]
                tmp_cls_infos = cls_branch[0:level_slicer[1]]
                for info in tmp_cls_infos:
                    cls_summation += info[1]
                    
                tmp_reg_infos = reg_branch[0:level_slicer[1]]
                for info in tmp_reg_infos:
                    reg_summation += info[1]
                    
                tmp_centerness_infos = centerness_branch[0:level_slicer[1]]
                for info in tmp_centerness_infos:
                    centerness_summation += info[1]
            
                level_cls_factors.append(cls_summation)
                level_reg_factors.append(reg_summation)
                level_centerness_factors.append(centerness_summation)
        
        if len(level_cls_factors) + len(level_reg_factors) == 0:
            # 必须让list内有level个空list
            level_cls_factors = [torch.tensor(1)] * len(labels_list)
            level_reg_factors = [torch.tensor(1)] * len(labels_list)
            level_centerness_factors = [torch.tensor(1)] * len(labels_list)


        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                level_cls_factors,
                level_reg_factors,
                level_centerness_factors,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness)

    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
