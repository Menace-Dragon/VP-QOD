CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12310 \
    --nproc_per_node=1 \
    tools/train.py \
    configs/yolox/yolox_s_8x8_300e_coco_quant_w4a4.py \
    mqbconfig/lsq/quant_config_mypro_w4a4.yaml \
    --work-dir /work_dirs/yolox_s_coco_w4a4_LSQ_HQOD \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
