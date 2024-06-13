_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='/workspace/whole_world/rdata/share/pretrained/checkpoints/resnet18-5c106cde.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(save_best='auto', interval=1, metric='bbox')
checkpoint_config = dict(interval=10)
data = dict(
    samples_per_gpu=2,  # 就是16
    workers_per_gpu=2,
)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=8, norm_type=2))
