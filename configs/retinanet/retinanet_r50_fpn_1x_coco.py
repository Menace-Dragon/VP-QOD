_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(save_best='auto', interval=1, metric='bbox')

data = dict(
    samples_per_gpu=4,  # 就是16
    workers_per_gpu=4,
)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
