_base_ = [
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_qat_w2a2.py', '../_base_/default_runtime.py', 'atss_quant_general.py'
]
model = dict(
    type='ATSS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/workspace/whole_world/bdata1/long.huang/temp/pretrained/backbones/resnet50-19c8e357.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),  # 奇怪，为甚恶魔atss这里是0.6？？
        max_per_img=100))
# dataset settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

evaluation = dict(save_best='auto', interval=10, dynamic_intervals=[(11, 1)],metric='bbox')
checkpoint_config = dict(interval=10)
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
load_from = 'work_dirs/atss_r50_fpn_1x_voc/best_bbox_mAP_epoch_6.pth'
