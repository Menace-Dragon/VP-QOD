CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12313 \
    --nproc_per_node=2 \
    tools/train.py \
    configs/retinanet/retinanet_r18_fpn_1x_coco_quant_w2a2_aqd.py \
    mqbconfig/lsq/quant_config_mypro_w2a2.yaml \
    --work-dir /work_dirs/retinanet_r18_fpn_coco_w2a2_LSQ_HQOD_AQD \
    --aqd-mode 5 \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
