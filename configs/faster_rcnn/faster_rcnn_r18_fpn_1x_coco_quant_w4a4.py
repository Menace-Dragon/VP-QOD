_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_no_freeze.py',
    '../_base_/datasets/coco_detection.py', '../_base_/schedules/schedule_qat_w4a4.py', 
    'faster_rcnn_quant_general.py', '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='/workspace/whole_world/rdata/share/pretrained/checkpoints/resnet18-5c106cde.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]))
evaluation = dict(save_best='auto', interval=1, metric='bbox')
checkpoint_config = dict(interval=10)
data = dict(
    samples_per_gpu=4,  # 就是8
    workers_per_gpu=4,  # 就是8
)
# 确实没有预训练模型，得自己训
