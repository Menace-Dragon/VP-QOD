from typing import Optional, Type

import torch
import torch.nn as nn
from torch.quantization.fx.fusion_patterns import ConvBNReLUFusion, ModuleReLUFusion
from torch.quantization.fx.quantization_types import QuantizerCls
from torch.fx.graph import Node

import mqbench.nn as qnn
import mqbench.nn.intrinsic as qnni
import mqbench.nn.intrinsic.qat as qnniqat
from mqbench.utils.fusion import fuse_deconv_bn_eval
from mqbench.nn.modules import FrozenBatchNorm2d


fuse_custom_config_dict = {
    "additional_fuser_method_mapping": {
        (torch.nn.Linear, torch.nn.BatchNorm1d): fuse_linear_bn,
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d): fuse_deconv_bn,
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_deconv_bn_relu,
        (torch.nn.ConvTranspose2d, torch.nn.ReLU): qnni.ConvTransposeReLU2d,
        (nn.Conv2d, FrozenBatchNorm2d, nn.ReLU): fuse_conv_freezebn_relu,
        (nn.Conv2d, FrozenBatchNorm2d): fuse_conv_freezebn,
        (nn.ConvTranspose2d, FrozenBatchNorm2d, nn.ReLU): fuse_deconv_freezebn_relu,
        (nn.ConvTranspose2d, FrozenBatchNorm2d): fuse_deconv_freezebn,
    },
    "additional_fusion_pattern": {  # 似乎这些都是torch官方的定义
        (torch.nn.BatchNorm1d, torch.nn.Linear):
        ConvBNReLUFusion,
        (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
        (torch.nn.ReLU, (FrozenBatchNorm2d, torch.nn.Conv2d)):
        ConvFreezebnReLUFusion,
        (FrozenBatchNorm2d, torch.nn.Conv2d):
        ConvFreezebnReLUFusion,
        (torch.nn.ReLU, (FrozenBatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvFreezebnReLUFusion,
        (FrozenBatchNorm2d, torch.nn.ConvTranspose2d):
        ConvFreezebnReLUFusion,
    },
    "additional_qat_module_mappings": {
        nn.ConvTranspose2d: qnn.qat.ConvTranspose2d,
        qnni.LinearBn1d: qnniqat.LinearBn1d,
        qnni.ConvTransposeBn2d: qnniqat.ConvTransposeBn2d,
        qnni.ConvTransposeReLU2d: qnniqat.ConvTransposeReLU2d,
        qnni.ConvTransposeBnReLU2d: qnniqat.ConvTransposeBnReLU2d,
        qnni.ConvFreezebn2d: qnniqat.ConvFreezebn2d,
        qnni.ConvFreezebnReLU2d: qnniqat.ConvFreezebnReLU2d,
        qnni.ConvTransposeFreezebn2d: qnniqat.ConvTransposeFreezebn2d,
        qnni.ConvTransposeFreezebnReLU2d: qnniqat.ConvTransposeFreezebnReLU2d,
        nn.Embedding: qnn.qat.Embedding,
    },
}
