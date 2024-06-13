CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12314 \
    --nproc_per_node=2 \
    tools/train.py \
    configs/retinanet/retinanet_r18_fpn_1x_coco_quant_w4a4_aqd.py \
    mqbconfig/lsq/quant_config_mypro_w4a4.yaml \
    --work-dir /work_dirs/retinanet_r18_fpn_coco_w4a4_LSQ_HQOD_AQD \
    --aqd-mode 5 \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
