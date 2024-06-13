# python tools/analysis_tools/plot_pr_curves.py \
#     configs/ssd/ssdlite_mobilenetv2_scratch_600e_voc.py \
#     work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/results.pkl \
#     --out work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/pr_curve \


# python tools/analysis_tools/plot_pr_curves.py \
#     configs/ssd/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4.py \
#     work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4/results.pkl \
#     --out work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4/pr_curve \


# python tools/analysis_tools/plot_pr_curves.py \
#     configs/ssd/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4.py \
#     work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp_qloss/results.pkl \
#     --out work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp_qloss/pr_curve \

python tools/analysis_tools/plot_pr_curves.py \
    configs/ssd/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4.py \
    work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp/results.pkl \
    --out work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp/pr_curve \

