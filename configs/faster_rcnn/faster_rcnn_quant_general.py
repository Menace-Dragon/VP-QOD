trace_config = dict(
    model_general_architecture = 'FasterRCNN', 
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
        preserve_attr = ['add_extra_convs', 'backbone_end_level', 'dump_patches', 'end_level', 'fp16_enabled'
                         , 'in_channels', 'init_cfg', 'is_init', 'l2_norm', 'no_norm_on_lateral', 'num_ins', 'num_outs', 'out_channels'
                         , 'relu_before_extra_convs', 'start_level', 'upsample_cfg'],
        not_duplicated_prefixes = [],
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  
        specified_general_quantizers = ['getitem_post_act_fake_quantizer', 'getitem_1_post_act_fake_quantizer', 'getitem_2_post_act_fake_quantizer', 'getitem_3_post_act_fake_quantizer'],
        last_8bit_module = []
    )),
    rpn_head_detail = dict(
        input_concrete_args = dict(in_num = 5),
        preserve_attr = ['anchor_generator', 'assigner', 'bbox_coder', 'cls_out_channels', 'conv_cfg', 'dump_patches', 'feat_channels', 'fp16_enabled', 
        'in_channels', 'init_cfg', 'is_init', 'norm_cfg', 'loss_bbox', 'loss_cls', 'num_anchors'
        , 'num_base_priors', 'num_classes', 'num_convs', 'prior_generator', 'reg_decoded_bbox', 'sampler', 'sampling', 'stacked_convs', 'test_cfg', 'train_cfg', 'use_sigmoid_cls'
        
        , 'simple_test', 'async_simple_test_rpn', 'aug_test', 'aug_test_bboxes', 'aug_test_rpn', 'forward_single', 'forward_train', 'get_anchors', 'get_bboxes', 'get_targets', 'loss', 'loss_single', 'merge_aug_bboxes', 
        'simple_test_bboxes', 'simple_test_rpn', '_get_bboxes_single', '_bbox_post_process', 'get_anchors', '_get_targets_single'],
        not_duplicated_prefixes = ['rpn_conv', 'rpn_cls', 'rpn_reg'],  # 避免共享头被复制、独立化
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  
        specified_general_quantizers = [],
        last_8bit_module = []
    )), 
    roi_head_bbox_head_detail = dict(
        input_concrete_args = dict(dim_setting = 2),
        preserve_attr = ['bbox_coder', 'cls_last_dim', 'cls_predictor_cfg', 'conv_cfg', 'conv_out_channels', 'custom_accuracy', 'custom_activation', 'custom_cls_channels'
                         , 'debug_imgs', 'dump_patches', 'fc_out_channels', 'fp16_enabled', 'in_channels', 'init_cfg', 'is_init', 'loss_bbox', 'loss_cls', 'norm_cfg', 'num_classes', 'num_cls_convs', 'num_cls_fcs', 'num_reg_convs', 'num_reg_fcs'
                         , 'num_shared_convs', 'num_shared_fcs', 'reg_class_agnostic', 'reg_decoded_bbox', 'reg_last_dim', 'reg_predictor_cfg', 'roi_feat_size', 'share_out_channels', 'with_avg_pool', 'with_cls', 'with_reg'
                         , 'get_bboxes', 'get_targets', 'loss', 'refine_bboxes', 'regress_by_class', '_get_bboxes_single', '_bbox_post_process', 'get_anchors', '_get_targets_single'],
        not_duplicated_prefixes = ['cls_convs', 'reg_convs', 'retina_cls', 'retina_reg'],  # 避免共享头被复制、独立化
        further_detail = dict(
        exclude_prefixes = [],  
        removed_quantizer_names = [],  
        qloss_flag = True, 
        specified_general_quantizers = [],
        last_8bit_module = [] 哦草，这里没写
    ))
    )