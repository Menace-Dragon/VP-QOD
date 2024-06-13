CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12321 \
    --nproc_per_node=8 \
    tools/train.py \
    configs/atss/atss_r50_fpn_1x_coco_quant_w2a2.py \
    mqbconfig/tqt/quant_config_w2a2.yaml \
    --work-dir /work_dirs/atss_r50_fpn_1x_coco_w2a2_TQT \
    --quantize \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
