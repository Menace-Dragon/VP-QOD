CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
    --use_env \
    --master_port=12312 \
    --nproc_per_node=8 \
    tools/train.py \
    configs/retinanet/retinanet_r18_fpn_1x_coco.py \
    shit \
    --work-dir /work_dirs/retinanet_r18_fpn_1x_coco_AQD \
    --aqd-mode 5 \
    --seed 1005 \
    --deterministic \
    --launcher pytorch 
