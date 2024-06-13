CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12325 \
    --nproc_per_node=2 \
    tools/train.py \
    configs/atss/atss_r50_fpn_1x_coco_quant_w2a2.py \
    mqbconfig/lsq/quant_config_w2a2_weight_loose.yaml \
    --work-dir /work_dirs/atss_r50_fpn_1x_coco_w2a2_LSQ \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
