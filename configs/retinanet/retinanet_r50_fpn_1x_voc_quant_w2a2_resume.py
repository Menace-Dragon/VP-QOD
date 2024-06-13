_base_ = [
    '../_base_/models/retinanet_r50_fpn_no_freeze.py',
    '../_base_/datasets/voc0712.py', '../_base_/schedules/schedule_qat_w2a2_big.py'
    , 'retinanet_fpn_quant_general.py', '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(num_classes=20))

evaluation = dict(save_best='auto', interval=10, dynamic_intervals=[(11, 1)],metric='bbox')
checkpoint_config = dict(interval=10)
# dataset settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.004,
    gamma=0.5,
    step=[8, 13, 14, 15, 16, 17, 19, 20, 21])  # 0.01 0.005 0.0025 0.00125 0.000625 
runner = dict(type='EpochBasedRunner', max_epochs=21)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# load_from = 'work_dirs/retinanet_r50_fpn_1x_voc/best_bbox_mAP_epoch_6.pth'
resume_from = 'work_dirs/retinanet_r50_fpn_voc_w2a2_LSQ_HQOD/best_bbox_mAP_epoch_18.pth'