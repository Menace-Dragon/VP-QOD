CUDA_VISIBLE_DEVICES=7 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12322 \
    --nproc_per_node=1 \
    tools/train.py \
    configs/atss/atss_r50_fpn_1x_coco_quant_w4a4.py \
    mqbconfig/tqt/quant_config_mypro_w4a4.yaml \
    --work-dir /work_dirs/atss_r50_fpn_1x_coco_w4a4_TQT_HQOD \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
