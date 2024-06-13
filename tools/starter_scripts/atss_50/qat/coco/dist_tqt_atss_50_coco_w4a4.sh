CUDA_VISIBLE_DEVICES=6 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12320 \
    --nproc_per_node=1 \
    tools/train.py \
    configs/atss/atss_r50_fpn_1x_coco_quant_w4a4.py \
    mqbconfig/tqt/quant_config_w4a4.yaml \
    --work-dir /work_dirs/atss_r50_fpn_1x_coco_w4a4_TQT \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
