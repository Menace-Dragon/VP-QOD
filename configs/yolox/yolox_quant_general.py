trace_config = dict(
    backbone_detail = dict(
        input_concrete_args = dict(),
        preserve_attr = ['arch_settings', 'dump_patches', 'frozen_stages', 'init_cfg', 'is_init', 'layers', 'norm_eval', 'out_indices', 'use_depthwise'],
        not_duplicated_prefixes = [],
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  # 用于取消某些算子的量化  len(set(x[0].flatten().tolist()))
        specified_general_quantizers = [],
        last_8bit_module = ['stem.conv.conv']
    )),
    neck_detail = dict(
        input_concrete_args = dict(in_num = 3),
        preserve_attr = ['in_channels', 'init_cfg', 'is_init', 'out_channels'],
        not_duplicated_prefixes = [],
        further_detail = dict(  # 'getitem_post_act_fake_quantizer'
        exclude_prefixes = [],  
        removed_quantizer_names = ['getitem_post_act_fake_quantizer', 'getitem_1_post_act_fake_quantizer'],  
        specified_general_quantizers = [], # 输入全是对称的
        last_8bit_module = []
    )),
    bbox_head_detail = dict(
        input_concrete_args = dict(in_num = 3),
        preserve_attr = ['act_cfg', 'assigner', 'cls_out_channels', 'conv_bias', 'conv_cfg', 'dcn_on_last_conv', 'dump_patches', 'feat_channels', 'fp16_enabled', 
        'in_channels', 'init_cfg', 'is_init', 'loss_bbox', 'loss_cls', 'loss_l1', 'loss_obj', 'norm_cfg'
        , 'num_classes', 'prior_generator', 'sampler', 'sampling', 'stacked_convs', 'strides', 'test_cfg', 'train_cfg', 'use_depthwise', 'use_l1', 'use_sigmoid_cls'
        
        , '_bbox_decode', '_bbox_post_process', '_bboxes_nms', '_get_backward_hooks', '_get_bboxes_single', '_get_l1_target', '_get_target_single'
        
        , 'simple_test', 'async_simple_test_rpn', 'aug_test_bboxes', 'aug_test_rpn', 'forward_single', 'forward_train', 'get_bboxes', 'get_targets', 'loss', 'loss_single', 'merge_aug_bboxes', 
        'simple_test_bboxes', 'simple_test_rpn'],
        not_duplicated_prefixes = [],
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  
        qloss_flag = True, 
        specified_general_quantizers = [], # 输入全是对称的
        last_8bit_module = ['multi_level_conv_cls.0', 'multi_level_conv_reg.0', 'multi_level_conv_obj.0', 'multi_level_conv_cls.1', 'multi_level_conv_reg.1', 'multi_level_conv_obj.1', 'multi_level_conv_cls.2', 'multi_level_conv_reg.2', 'multi_level_conv_obj.2']
    )))