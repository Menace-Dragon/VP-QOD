_base_ = [
    '../_base_/models/retinanet_r50_fpn_no_freeze.py',
    '../_base_/datasets/voc0712.py', '../_base_/schedules/schedule_qat_w4a4.py'
    , 'retinanet_fpn_quant_general.py', '../_base_/default_runtime.py'
]


# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='/workspace/whole_world/rdata/share/pretrained/checkpoints/resnet18-5c106cde.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    bbox_head=dict(num_classes=20))
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(save_best='auto', interval=1, metric='bbox')
checkpoint_config = dict(interval=10)



# dataset settings
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8)



# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
load_from = '/workspace/share/long_dir/retinanet_r18_fpn_1x_voc/best_bbox_mAP_epoch_6.pth'
