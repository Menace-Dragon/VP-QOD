trace_config = dict(
    backbone_detail = dict(
        input_concrete_args = dict(),
        preserve_attr = ['act_cfg', 'arch_settings', 'conv_cfg', 'bn_eval', 'is_init', 'bn_frozen', 'extra_setting', 'frozen_stages', 'init_cfg', 'inplanes'
        , 'out_feature_indices', 'out_indices', 'range_sub_modules', 'stage_blocks'],
        not_duplicated_prefixes = [],
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  
        specified_general_quantizers = [],
        last_8bit_module = []
    )),
    neck_detail = dict(
        input_concrete_args = dict(in_num = 2),
        preserve_attr = ['init_cfg', 'is_init', 'l2_norm'],
        not_duplicated_prefixes = [],
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  
        specified_general_quantizers = ['getitem_1_post_act_fake_quantizer'],
        last_8bit_module = []
    )),
    bbox_head_detail = dict(
        input_concrete_args = dict(in_num = 6),
        preserve_attr = ['simple_test', 'num_classes', 'assigner', 'bbox_coder', 'cls_focal_loss', 'cls_out_channels', 'conv_cfg', 'feat_channels', 
        'fp16_enabled', 'in_channels', 'init_cfg', 'is_init', 'norm_cfg', 'num_base_priors', 'prior_generator', 'reg_decoded_bbox'
        , 'sampler', 'sampling', 'stacked_convs', 'test_cfg', 'train_cfg', 'use_depthwise', 'use_sigmoid_cls', 'async_simple_test_rpn', 'aug_test', 
        'aug_test_bboxes', 'aug_test_rpn', 'forward_single', 'forward_train', 'get_bboxes', 'get_targets', 'loss', 'loss_single', 'merge_aug_bboxes', 
        'simple_test_bboxes', 'simple_test_rpn', '_get_bboxes_single', '_bbox_post_process', 'get_anchors', '_get_targets_single'],
        not_duplicated_prefixes = [],
        further_detail = dict(
        # exclude_prefixes = ['cls_convs.0.0', 'cls_convs.1.0', 'cls_convs.2.0', 'cls_convs.3.0', 'cls_convs.4.0', 'cls_convs.5.0'],   # 只禁用cls分支 
        # exclude_prefixes = ['reg_convs.0.0', 'reg_convs.1.0', 'reg_convs.2.0', 'reg_convs.3.0', 'reg_convs.4.0', 'reg_convs.5.0'], # 只禁用reg分支 
        
        qloss_flag = True, 
        specified_general_quantizers = ['getitem_post_act_fake_quantizer', 'getitem_1_post_act_fake_quantizer', 'getitem_2_post_act_fake_quantizer'
                         , 'getitem_3_post_act_fake_quantizer', 'getitem_4_post_act_fake_quantizer', 'getitem_5_post_act_fake_quantizer'],
        last_8bit_module = ['cls_convs.0.0', 'cls_convs.1.0', 'cls_convs.2.0', 'cls_convs.3.0', 'cls_convs.4.0'
                            , 'cls_convs.5.0', 'reg_convs.0.0', 'reg_convs.1.0',  'reg_convs.2.0', 'reg_convs.3.0', 'reg_convs.4.0',  'reg_convs.5.0']
    )))