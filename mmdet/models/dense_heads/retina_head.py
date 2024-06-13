# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer, ConvModule
import global_placeholder
from ..builder import HEADS
from .anchor_head import AnchorHead

class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result

@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        if global_placeholder.aqd_mode != 0:
            self.norm_cfg = dict(type='BN', requires_grad=True)
            
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
            # if i == 0:
            #     self.norm_cfg = dict(type='BN', requires_grad=True)
            # else:
            #     self.norm_cfg = None
            # # self.norm_cfg = None
            
            if global_placeholder.aqd_mode != 0:
                # 启动AQD模式
                self.num_levels = global_placeholder.aqd_mode
                
                lega = self.cls_convs[-1].bn
                self.cls_convs[-1].bn = ModuleListDial(
                        [build_norm_layer(self.norm_cfg, self.feat_channels)[-1] for _ in range(self.num_levels)
                    ])
                del lega  # 删除，避免占内存（强迫症
                
                lega = self.reg_convs[-1].bn
                self.reg_convs[-1].bn = ModuleListDial(
                        [build_norm_layer(self.norm_cfg, self.feat_channels)[-1] for _ in range(self.num_levels)
                    ])
                del lega
                
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)


    def forward(self, feats):  # forward_single直接改成forward，应该不会有啥影响
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        if hasattr(self,'in_num') and self.in_num == 5:
            feats = [feats[0], feats[1], feats[2], feats[3], feats[4]]
        elif hasattr(self,'in_num'):
            raise NotImplementedError
        else:
            self.in_num = 5
            feats = [feats[0], feats[1], feats[2], feats[3], feats[4]]
        
        for x in feats:
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            
        return cls_scores, bbox_preds


    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
