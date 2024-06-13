from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.convert_deploy import convert_deploy
import os
import yaml
from easydict import EasyDict
from mmdet.models.dense_heads import GARPNHead, RPNHead

from shutil import copyfile
import time
import errno

backend_dict = {
    'Academic': BackendType.Academic,
    'Tensorrt': BackendType.Tensorrt,
    'SNPE': BackendType.SNPE,
    'PPLW8A16': BackendType.PPLW8A16,
    'NNIE': BackendType.NNIE,
    'Vitis': BackendType.Vitis,
    'ONNX_QNN': BackendType.ONNX_QNN,
    'PPLCUDA': BackendType.PPLCUDA,
}

# def load_calibrate_data(train_loader, cali_batchsize):
#     cali_data = []
#     targets = []
#     for i, batch in enumerate(train_loader):
#         cali_data.append(batch[0])
#         targets.append(batch[1])
#         if i + 1 == cali_batchsize:
#             break
#     return cali_data, targets

def get_quantize_model(model, config, structure_detail):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    return prepare_by_platform(
        model, backend_type, structure_detail, extra_prepare_dict)


# def deploy(model, config):
#     backend_type = BackendType.Academic if not hasattr(
#         config.quantize, 'backend') else backend_dict[config.quantize.backend]
#     output_path = './' if not hasattr(
#         config.quantize, 'deploy') else config.quantize.deploy.output_path
#     model_name = config.quantize.deploy.model_name
#     deploy_to_qlinear = False if not hasattr(
#         config.quantize.deploy, 'deploy_to_qlinear') else config.quantize.deploy.deploy_to_qlinear

#     convert_deploy(model, backend_type, {
#                    'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)


def make_qmodel_for_mmd(model, quant_config, cfg):
    print('\nGet FakeQuant model\n')
    model.backbone = get_quantize_model(model.backbone, quant_config, cfg.backbone_detail)  # QAT时，这个需要eval还是train
    model.neck = get_quantize_model(model.neck, quant_config, cfg.neck_detail)  # QAT时，这个需要eval还是train
    model_general_architecture = cfg.get('model_general_architecture', None)
    if model_general_architecture == 'FasterRCNN':
        temp = get_quantize_model(model.rpn_head, quant_config, cfg.rpn_head_detail)  # QAT时，这个需要eval还是train
        # temp.__class__ = model.rpn_head.__class__  # NOTE 无奈之举
        model.rpn_head.forward = temp.forward  # 太傻蛋勒
        model.rpn_head = temp
        temp = get_quantize_model(model.roi_head.bbox_head, quant_config, cfg.roi_head_bbox_head_detail)  # QAT时，这个需要eval还是train
        # temp.__class__ = model.roi_head.bbox_head.__class__
        model.roi_head.bbox_head.forward = temp.forward  # 太傻蛋勒
        model.roi_head.bbox_head = temp
    else:
        temp = get_quantize_model(model.bbox_head, quant_config, cfg.bbox_head_detail)  # QAT时，这个需要eval还是train
        model.bbox_head.forward = temp.forward  # 太傻蛋勒
        model.bbox_head = temp
    
    # TODO 把东西del掉？
    return model

def prepocess(config_path):
    # TODO 写一个cp yaml操作，
    config = parse_config(config_path)
    # seed first
    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    # if config.misc.output_dir:
    #     mkdir(config.misc.output_dir)
    #     config.misc.output_data_dir = os.path.join(config.misc.output_dir, 'data')
    #     mkdir(config.misc.output_data_dir)
    # seed_all(config.process.seed)
    return config


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
        
def copy_config_file(config_file_path, dir):
    # file_name = os.path.basename(config_file_path)
    now_stamp = int(round(time.time()*1000))
    now_time = time.strftime('%Y_%m_%d_%H_%M',time.localtime(now_stamp/1000))
    new_file_name = f'config_{now_time}.yaml'
    new_file_path = os.path.join(dir, new_file_name)
    if not os.path.exists(dir):  # 判断路径是否存在
        os.makedirs(dir)  # 创建文件夹
    copyfile(config_file_path, new_file_path)
    
    