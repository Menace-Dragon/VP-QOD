CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12301 \
    --nproc_per_node=1 \
    tools/train.py \
    configs/retinanet/retinanet_r18_fpn_1x_coco_quant_w2a2.py \
    mqbconfig/lsq/quant_config_w2a2.yaml \
    --work-dir /work_dirs/retinanet_r18_fpn_coco_w2a2_LSQ \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
