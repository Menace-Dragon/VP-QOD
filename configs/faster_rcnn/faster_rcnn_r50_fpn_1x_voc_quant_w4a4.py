_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_no_freeze.py',
    '../_base_/datasets/voc0712.py', '../_base_/schedules/schedule_qat_w4a4.py', 
    'faster_rcnn_quant_general.py', '../_base_/default_runtime.py'
]
evaluation = dict(save_best='auto', interval=1, metric='bbox')
checkpoint_config = dict(interval=10)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20)))

# dataset settings
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)
# load_from = '../long_used_pretrained/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
