_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_no_freeze.py',
    '../_base_/datasets/coco_detection.py', '../_base_/schedules/schedule_qat_w4a4.py',
    'faster_rcnn_quant_general.py', '../_base_/default_runtime.py'
]

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(save_best='auto', interval=1, metric='bbox')
checkpoint_config = dict(interval=10)
data = dict(
    samples_per_gpu=16,  # 就是8
    workers_per_gpu=8,  # 就是8
)

load_from = '../long_used_pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'



