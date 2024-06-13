trace_config = dict(
    backbone_detail = dict(
        input_concrete_args = dict(),
        preserve_attr = ['arch_settings', 'avg_down', 'base_channels', 'conv_cfg', 'dcn', 'deep_stem', 'depth', 'dump_patches', 'feat_dim'
        , 'frozen_stages', 'init_cfg', 'inplanes', 'is_init', 'norm_cfg', 'norm_eval', 'out_indices'
        , 'plugins', 'res_layers', 'stage_block', 'stage_with_dcn', 'stem_channels', 'zero_init_residual', 'strides', 'with_cp', 'make_res_layer', 'make_stage_plugins'],
        not_duplicated_prefixes = [],
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  
        specified_general_quantizers = [],
        last_8bit_module = []
    )),
    neck_detail = dict(
        input_concrete_args = dict(in_num = 4),
        preserve_attr = ['in_channels', 'init_cfg', 'is_init', 'l2_norm', 'no_norm_on_lateral', 'num_ins', 'num_outs', 'out_channels', 'relu_before_extra_convs', 'start_level', 'upsample_cfg'],
        not_duplicated_prefixes = [],
        further_detail = dict(  # 'getitem_post_act_fake_quantizer'
        exclude_prefixes = [],  
        removed_quantizer_names = ['getitem_1_post_act_fake_quantizer','getitem_2_post_act_fake_quantizer'],  
        specified_general_quantizers = ['getitem_1_post_act_fake_quantizer', 'getitem_2_post_act_fake_quantizer', 'getitem_3_post_act_fake_quantizer'],
        last_8bit_module = []
    )),
    bbox_head_detail = dict(
        input_concrete_args = dict(in_num = 5),
        preserve_attr = ['anchor_generator', 'assigner', 'bbox_coder', 'cls_out_channels', 'conv_cfg', 'dump_patches', 'feat_channels', 'fp16_enabled', 
        'in_channels', 'init_cfg', 'is_init', 'norm_cfg', 'loss_bbox', 'loss_cls', 'loss_centerness', 'num_anchors'
        , 'num_base_priors', 'num_classes', 'prior_generator', 'reg_decoded_bbox', 'sampler', 'sampling', 'stacked_convs', 'test_cfg', 'train_cfg', 'use_sigmoid_cls'
        
        , 'simple_test', 'async_simple_test_rpn', 'aug_test', 'aug_test_bboxes', 'aug_test_rpn', 'forward_single', 'forward_train', 'get_anchors', 'get_bboxes', 'get_targets', 'loss', 'loss_single', 'merge_aug_bboxes', 
        'simple_test_bboxes', 'simple_test_rpn', '_get_bboxes_single', '_get_target_single', 'get_num_level_anchors_inside', 'centerness_target', '_bbox_post_process', 'get_anchors', '_get_targets_single'],
        not_duplicated_prefixes = ['cls_convs', 'reg_convs', 'atss_cls', 'atss_reg', 'atss_centerness'],
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = ['getitem_2_post_act_fake_quantizer','getitem_3_post_act_fake_quantizer'],  
        qloss_flag = True, 
        specified_general_quantizers = [],
        last_8bit_module = ['atss_cls', 'atss_reg', 'atss_centerness']
    )))