import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils import is_symmetric_quant, is_tracing_state
from mqbench.utils.hook import PerChannelLoadHook
import global_placeholder

class PureHooker(QuantizeBase):
    r""" 
    """

    def __init__(self, observer, scale=1., zero_point=0., use_grad_scaling=True, **observer_kwargs):
        super(PureHooker, self).__init__(observer, **observer_kwargs)
        # self.regular_margin = Parameter(torch.tensor([1.]))
        # self.use_grad_scaling = use_grad_scaling
        # self.scale = Parameter(torch.tensor([scale]))
        # self.register_buffer('zero_point', torch.tensor([zero_point]))  # NOTE 已改  这里就是不对劲，就应该是buffer，而且grad还会占用显存
        # self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        # # Check whether the module will load a state dict;
        # # Initialize the shape of per-channel 'scale' and 'zero-point' before copying values
        # self.load_state_dict_hook = PerChannelLoadHook(self)
        # # NOTE test
        # # self.register_buffer('quantization_loss', torch.tensor([0]))
        self.compute_qloss = False
        self.ema = 0


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
        if self.observer_enabled[0] == 1:
            
            if self.compute_qloss:
                # 计算std，初始化margin
                if self.ema == 0:
                    self.ema = 2 * X.std()
                else:
                    self.ema = 0.9 * 2 * X.std() + 0.1 * self.ema
                self.regular_margin.data.copy_(self.ema)
            
        if self.fake_quant_enabled[0] == 1:
        # Learnable fake quantize have to zero_point.float() to make it learnable.
            if self.compute_qloss:
                diff = torch.max(X.abs() - self.regular_margin)
                diff = torch.where(diff < 0., torch.zeros_like(diff), diff)
                self.quantization_loss = self.regular_margin + diff + X.std() ** 2  # 这样子一开始直接崩
                # self.quantization_loss = self.regular_margin + diff
        
        return X



def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)

