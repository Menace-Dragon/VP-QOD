CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12309 \
    --nproc_per_node=1 \
    tools/train.py \
    configs/yolox/yolox_s_8x8_300e_coco_quant_w2a2.py \
    mqbconfig/lsq/quant_config_w2a2_weight_loose.yaml \
    --work-dir /work_dirs/yolox_s_coco_w2a2_LSQ \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
