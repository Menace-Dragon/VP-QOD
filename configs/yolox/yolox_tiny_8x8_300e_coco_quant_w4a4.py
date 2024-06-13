_base_ = './yolox_s_8x8_300e_coco_quant_w4a4.py'

# model settings
model = dict(
    random_size_range=(10, 20),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96))

img_scale = (640, 640)  # height, width

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

train_dataset = dict(pipeline=train_pipeline)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=train_dataset,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.002,  # 确认之后，应该决定0.004开始
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0., 
                    # custom_keys={ 

                    #         # 'norm': dict(lr_mult=0.),
                    #         # 'bn': dict(lr_mult=0.)
                    #         'quant': dict(lr_mult=0.04)  # 还真是这个的问题，破案了
                    #         }
    ))
optimizer_config = dict(grad_clip=None)

max_epochs = 16
num_last_epochs = 6
resume_from = None
interval = 1

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,  # 1 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.1)  # 要让l1启动时为4e-5 但是hqod一来就得更低

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=12)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=200)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
load_from = '/workspace/whole_world/bdata1/long.huang/temp/pretrained/long_used/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
# tune_from = 'work_dirs/yolox_tiny_coco_w4a4_LSQ_LEGA/best_bbox_mAP_epoch_16.pth'
# tune_from = 'work_dirs/yolox_tiny_coco_w4a4_LSQ_HQOD_no_correlationLoss_LEGA/best_bbox_mAP_epoch_15.pth'
# tune_from = 'work_dirs/yolox_tiny_coco_w4a4_LSQ_HQOD/best_bbox_mAP_epoch_15.pth'
