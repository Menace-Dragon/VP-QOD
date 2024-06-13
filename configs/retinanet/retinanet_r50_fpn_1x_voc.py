_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/voc0712.py', '../_base_/schedules/schedule_1x_voc_general.py', 
    '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(num_classes=20))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
evaluation = dict(save_best='auto', interval=1, metric='bbox')
checkpoint_config = dict(interval=10)
# dataset settings
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)


