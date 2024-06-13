_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py', '../_base_/schedules/schedule_1x_voc_general.py',
    '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='../long_used_pretrained/resnet18-5c106cde.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    roi_head=dict(
        bbox_head=dict(
            num_classes=20)))
evaluation = dict(save_best='auto', interval=1, metric='bbox')
checkpoint_config = dict(interval=10)


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8)

