_base_ = [
    '../_base_/models/retinanet_r50_fpn_no_freeze.py',
    '../_base_/datasets/coco_detection.py', '../_base_/schedules/schedule_qat_w4a4.py'
    , 'retinanet_fpn_quant_general.py', '../_base_/default_runtime.py'
]
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(save_best='auto', interval=10, dynamic_intervals=[(11, 1)],metric='bbox')
checkpoint_config = dict(interval=10)
data = dict(
    samples_per_gpu=16,  # 就是16
    workers_per_gpu=8,
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
load_from = '../long_used_pretrained/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
# resume_from = 'work_dirs/retinanet_r50_fpn_1x_coco_quant_w4a4_HQOD/best_bbox_mAP_epoch_21_can.pth'
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))