import torch 


class QuantizeScheme(object):  # 结构体类，记录qscheme
    """Describe quantization scheme.
    """
    def __init__(self, symmetry=True, per_channel=False, pot_scale=False, bit=8, **kwargs):
        self.symmetry = symmetry
        self.per_channel = per_channel
        self.pot_scale = pot_scale
        self.bit = bit
        if self.per_channel:
            self.torch_qscheme = torch.per_channel_symmetric if self.symmetry else torch.per_channel_affine  # 原来非对称叫affine
        else:
            self.torch_qscheme = torch.per_tensor_symmetric if self.symmetry else torch.per_tensor_affine
        if 'symmetric_range' in kwargs:
            self.symmetric_range = kwargs['symmetric_range']
            del kwargs['symmetric_range']
        else:
            self.symmetric_range = False
            
        if 'sign' in kwargs:
            self.sign = kwargs['sign']
        else:
            self.sign = True
            kwargs['sign'] = True
            
        self.kwargs = kwargs

    def to_observer_params(self):  # 生成更细致的量化参数
        naive_para = {
            'quant_min': (-2 ** (self.bit - 1) + 1 if self.symmetric_range else -2 ** (self.bit - 1)) if (self.symmetry and self.sign) else 0,
            'quant_max': 2 ** (self.bit - 1) - 1 if (self.symmetry and self.sign) else 2 ** self.bit - 1,
            'dtype': torch.qint8 if (self.symmetry  and self.sign) else torch.quint8,
            'pot_scale': self.pot_scale,
            'qscheme': self.torch_qscheme,
            'reduce_range': False,
            'ch_axis': 0 if self.per_channel else -1
            , 'bit': self.bit
        }
        naive_para.update(self.kwargs)
        return naive_para

    def __str__(self):
        return "Symmetric: {} / Bitwidth: {} / Per channel: {} / Pot scale: {} / Extra kwargs: {}".format(
            self.symmetry, self.bit, self.per_channel, self.pot_scale, self.kwargs)
