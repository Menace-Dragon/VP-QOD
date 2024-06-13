CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12307 \
    --nproc_per_node=4 \
    tools/train.py \
    configs/retinanet/retinanet_r50_fpn_1x_coco_quant_w2a2.py \
    mqbconfig/lsq/quant_config_mypro_w2a2_weight_loose.yaml \
    --work-dir /work_dirs/retinanet_r50_fpn_coco_w2a2_LSQ_HQOD \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
