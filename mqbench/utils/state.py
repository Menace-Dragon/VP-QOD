import torch

from mqbench.utils.logger import logger


def enable_calibration(model):  # 启动所有的observer，但停用quantizer
    logger.info('Enable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()

def enable_calibration_woquantization(model, quantizer_type='fake_quant'):  # 启动对应前缀名字的量化器的observer，关闭量化器。同时停用非名字匹配的quantizer和ob
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name:   # TODO 突发！原来weight quantize也是一个独立的个体!但不是layer，但是是怎么访问到的？？
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()

def enable_calibration_quantization(model, quantizer_type='fake_quant'):  # 启动对应前缀名字的量化器及ob。同时停用非名字匹配的quantizer和ob
    logger.info('Enable observer and Enable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name: 
                logger.info('Disable observer and Disable quantize for {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Enable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.enable_fake_quant()

def enable_quantization(model):  # 启用所有的quantizer，但停用ob
    logger.info('Disable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()


def disable_all(model):# 停用所有的quantizer和ob
    logger.info('Disable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Disable observer and Disable quantize: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()
