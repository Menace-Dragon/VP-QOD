import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils import is_symmetric_quant, is_tracing_state
from mqbench.utils.hook import PerChannelLoadHook
import global_placeholder

class LearnableFakeQuantize(QuantizeBase):
    r""" This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.
    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.
    """

    def __init__(self, observer, scale=1., zero_point=0., use_grad_scaling=True, **observer_kwargs):
        super(LearnableFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.use_grad_scaling = use_grad_scaling
        self.scale = Parameter(torch.tensor([scale]))
        self.register_buffer('zero_point', torch.tensor([zero_point]))  # NOTE 已改  这里就是不对劲，就应该是buffer，而且grad还会占用显存
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        # Check whether the module will load a state dict;
        # Initialize the shape of per-channel 'scale' and 'zero-point' before copying values
        self.load_state_dict_hook = PerChannelLoadHook(self)
        # NOTE test
        # self.register_buffer('quantization_loss', torch.tensor([0]))
        self.compute_qloss = False


    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List[%s]' % str(self.scale.shape),
                   self.zero_point if self.ch_axis == -1 else 'List')

    def forward(self, X):
        # Learnable fake quantize have to zero_point.float() to make it learnable.
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
            
            # if self.compute_qloss:
            #     # 计算std，初始化margin
            #     self.regular_margin.data.copy_(2 * X.std())
        else:
            # if self.compute_qloss:
            #     # 计算std，初始化margin
            #     self.regular_margin.data.abs_()  # 要求绝对化
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        # TODO 写能求最小化量化误差的代码
        X_old = X
        # if self.compute_qloss:
        #     X = grad_scale(X, 1+self.scale.detach())


        if self.fake_quant_enabled[0] == 1:
            if is_symmetric_quant(self.qscheme):
                self.zero_point.data.zero_()
            else:
                self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()

            if self.is_per_channel:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                if is_tracing_state():
                    X = FakeQuantizeLearnablePerchannelAffine.apply(
                        X, self.scale, self.zero_point, self.ch_axis,
                        self.quant_min, self.quant_max, grad_factor)
                else:
                    X = _fake_quantize_learnable_per_channel_affine_training(
                        X, self.scale, self.zero_point, self.ch_axis,
                        self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = torch._fake_quantize_learnable_per_tensor_affine(  # 原装
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)
                # X = _fake_quantize_learnable_per_tensor_affine_training(X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
            # NOTE 算
            # self.input = X_old.detach()
            
            # if self.compute_qloss and hasattr(self, 'identity'):
            # # #     # self.quantization_loss = (torch.norm(X_old - X, p="fro", dim=1) ** 2).mean()  # 这玩意也不行了
                
            # # #     # gap = (X_old - X_old.min())/(X_old.max() - X_old.min()) - (X - X.min())/(X.max() - X.min())
            # # #     # gap = ((X_old - X) / self.scale + self.zero_point) / (self.quant_max - self.quant_min + 1)
            # # #     scale = self.scale.detach()
            # # #     zero_point = self.zero_point.detach()
            # # #     # scale = grad_scale(scale, grad_factor)
            # # #     # zero_point = grad_scale(zero_point, grad_factor)
            # # #     gap = ((X_old - X) / scale)
            # # #     # self.quantization_loss = (torch.norm(X_old - X, p="fro", dim=1) ** 2).mean()  # 这玩意也不行了
            # # #     self.quantization_loss = (gap.abs()).mean()
            # # #     # self.quantization_loss = (gap ** 2).mean()   # 拉爆了
            # #     diff = torch.max(X_old.abs() - self.regular_margin)
            # #     diff = torch.where(diff < 0., torch.zeros_like(diff), diff)
            # #     self.quantization_loss = self.regular_margin + diff + 1/global_placeholder.quant_bit * self.scale.detach() * self.identity * X_old.std()
            # #     # self.quantization_loss = self.regular_margin + diff
                
            #     # self.quantization_loss =1/global_placeholder.quant_bit * self.scale.detach() * self.identity * X.std()
            #     scale = grad_scale(self.scale, grad_factor)
                
            #     self.quantization_loss =(1/global_placeholder.quant_bit * scale) ** 2


        return X


def _fake_quantize_learnable_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x = x / scale + zero_point
    x = (x.round() - x).detach() + x
    x = torch.clamp(x, quant_min, quant_max)
    return (x - zero_point) * scale

def _fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x = x / scale + zero_point
    x = (x.round() - x).detach() + x
    x = torch.clamp(x, quant_min, quant_max)
    return (x - zero_point) * scale


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)


class FakeQuantizeLearnablePerchannelAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
        return _fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis,
                                                                    quant_min, quant_max, grad_factor)

    @staticmethod
    def symbolic(g, x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
        return g.op("::FakeQuantizeLearnablePerchannelAffine", x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max)
