# python tools/analysis_tools/confusion_matrix.py \
#     configs/ssd/ssdlite_mobilenetv2_scratch_600e_voc.py \
#     work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/results.pkl \
#     work_dirs/ssdlite_mobilenetv2_scratch_600e_voc \

# python tools/analysis_tools/confusion_matrix.py \
#     work_dirs/retinanet_r18_fpn_1x_voc/retinanet_r18_fpn_1x_voc.py \
#     work_dirs/retinanet_r18_fpn_1x_voc/results.pkl \
#     work_dirs/retinanet_r18_fpn_1x_voc \

# python tools/analysis_tools/confusion_matrix.py \
#     work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ/retinanet_r18_fpn_1x_voc_quant_w4a4.py \
#     work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ/best_bbox_mAP_epoch_16.pkl \
#     work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ \


# python tools/analysis_tools/confusion_matrix.py \
#     work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/retinanet_r18_fpn_1x_voc_quant_w2a2.py \
#     work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/best_bbox_mAP_epoch_15.pkl \
#     work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ \

# python tools/analysis_tools/confusion_matrix.py \
#     work_dirs/retinanet_r50_fpn_1x_voc/retinanet_r50_fpn_1x_voc.py \
#     work_dirs/retinanet_r50_fpn_1x_voc/results.pkl \
#     work_dirs/retinanet_r50_fpn_1x_voc \

# python tools/analysis_tools/confusion_matrix.py \
#     work_dirs/retinanet_r50_fpn_voc_w4a4_LSQ/retinanet_r50_fpn_1x_voc_quant_w4a4.py \
#     work_dirs/retinanet_r50_fpn_voc_w4a4_LSQ/best_bbox_mAP_epoch_15.pkl \
#     work_dirs/retinanet_r50_fpn_voc_w4a4_LSQ \

python tools/analysis_tools/confusion_matrix.py \
    work_dirs/retinanet_r50_fpn_voc_w2a2_LSQ_le/retinanet_r50_fpn_1x_voc_quant_w2a2.py \
    work_dirs/retinanet_r50_fpn_voc_w2a2_LSQ_le/best_bbox_mAP_epoch_17.pkl \
    work_dirs/retinanet_r50_fpn_voc_w2a2_LSQ_le \


