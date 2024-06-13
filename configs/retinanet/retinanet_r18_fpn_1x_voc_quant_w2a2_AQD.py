_base_ = [
    '../_base_/models/retinanet_r50_fpn_no_freeze.py',
    '../_base_/datasets/voc0712.py', '../_base_/schedules/schedule_qat_w2a2.py'
    , 'retinanet_fpn_quant_general.py', '../_base_/default_runtime.py'
]
# schedule_qat_w2a2  schedule_fine_tune_general

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



# dataset settings
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)



# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
load_from = 'work_dirs/retinanet_r18_fpn_1x_voc/best_bbox_mAP_epoch_6.pth'
# tune_from = 'work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/best_bbox_mAP_epoch_18.pth'