_base_ = [
    '../_base_/models/retinanet_r50_fpn_no_freeze.py',
    '../_base_/datasets/voc0712.py', '../_base_/schedules/schedule_qat_w4a4.py'
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

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# load_from = '../long_used_pretrained/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'