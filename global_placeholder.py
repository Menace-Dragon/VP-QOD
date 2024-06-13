
#--------- 常值
# res18_pretrained_path = '/workspace/share/pretrained/resnet18_imagenet.pth'
# res34_pretrained_path = '/workspace/share/pretrained/resnet34_imagenet.pth'
# res50_pretrained_path = '/workspace/share/pretrained/resnet50_imagenet.pth'
# mbnv2_pretrained_path = '/workspace/share/pretrained/mobilenetv2_imagenet.pth'
# vgg16_pretrained_path = '/workspace/share/pretrained/vgg16_imagenet.pth'

# ssd300_pretrained_path = '/workspace/share/pretrained/nvidia_ssdpyt_amp_200703.pt'
#------------ 等着被初始化
quant_bit = None
quant_algorithm = None
model_type = None
num_classes = None
pretrained_flag = None
mybuff_flag = 0
qloss_flag = False
fold_bn_flag = False
aqd_mode = 0

def modify_quant_bit(bit_setting):
    global quant_bit
    print(f'\nModify the global bit hyparam as {bit_setting}')
    quant_bit = bit_setting
    
def modify_quant_algorithm(algorithm_setting):
    global quant_algorithm
    print(f'\nModify the global algorithm as {algorithm_setting}')
    quant_algorithm = algorithm_setting
    
def modify_model_type(model_type_setting):
    global model_type
    print(f'\nModify the global model type as {model_type_setting}')
    model_type = model_type_setting
    
def modify_buff_flag(my_buff_setting):
    global mybuff_flag
    if my_buff_setting == 0:
        text = 'naive'
    elif my_buff_setting == 1:
        text = 'hqod'
    elif my_buff_setting == 2:
        text = 'hardet'
    else:
        raise NotImplementedError
    print(f'\nModify the global buff flag as {my_buff_setting}：{text}')
    mybuff_flag = my_buff_setting

def modify_qloss_flag(qloss_setting):
    global qloss_flag
    print(f'\nModify the global qloss flag as {qloss_setting}')
    qloss_flag = qloss_setting

def modify_num_classes(num_classes_setting):
    global num_classes
    print(f'\nModify the global class num as {num_classes_setting}')
    num_classes = num_classes_setting
    
def modify_pretrained_flag(pretrained_flag_setting):
    global pretrained_flag
    print(f'\nModify the global pretrained flag as {pretrained_flag_setting}')
    pretrained_flag = pretrained_flag_setting  
      
def modify_fold_bn_flag(fold_bn_flag_setting):
    global fold_bn_flag
    print(f'\nModify the global fold bn flag as {fold_bn_flag_setting}')
    fold_bn_flag = fold_bn_flag_setting
    
def modify_AQD_mode(AQD_mode_setting):
    global aqd_mode
    print(f'\nModify the global AQD mode flag as {AQD_mode_setting}')
    aqd_mode = AQD_mode_setting  