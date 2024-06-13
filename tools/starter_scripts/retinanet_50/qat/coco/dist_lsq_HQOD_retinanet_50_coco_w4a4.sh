CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12306 \
    --nproc_per_node=1 \
    tools/train.py \
    configs/retinanet/retinanet_r50_fpn_1x_coco_quant_w4a4.py \
    mqbconfig/lsq/quant_config_mypro_w4a4.yaml \
    --work-dir /work_dirs/retinanet_r50_fpn_coco_w4a4_LSQ_HQOD \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
