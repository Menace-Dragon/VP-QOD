_base_ = [
    '../_base_/models/retinanet_r50_fpn_no_freeze.py',
    '../_base_/schedules/schedule_qat_w4a4.py'
    , 'retinanet_fpn_quant_general.py', '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'VOCDataset'
data_root = '/workspace/whole_world/bdata1/long.huang/temp/VOC/VOCdevkit/'
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + 'coco_format/voc07_test.json',  # 注意这里走的是COCO格式
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + 'coco_format/voc07_test.json',  # 注意这里走的是COCO格式
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'coco_format/voc07_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes))



# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='/workspace/whole_world/bdata1/long.huang/temp/pretrained/backbones/resnet18-5c106cde.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    bbox_head=dict(num_classes=20))
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(save_best='auto', interval=10, dynamic_intervals=[(11, 1)],metric='bbox')
checkpoint_config = dict(interval=10)



# # dataset settings
# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=4)



# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
load_from = 'work_dirs/retinanet_r18_fpn_1x_voc/best_bbox_mAP_epoch_6.pth'
