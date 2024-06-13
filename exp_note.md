



# 注意事项
* 在rstar上，启动程序的方法是：`CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env object_detecion_qat/QAT_multi_GPU.py`
    * 其中 CUDA_VISIBLE_DEVICES=0,1 表示可见的GPU序号为0,1。
    * --nproc_per_node=2 表示在一个机器上跑2张卡。
    * --use_env 表示从环境中加载distibuted配置。
    * 主要的程序就是QAT_multi_GPU.py。
    * 该指令在teql文件夹内执行。
* 启动程序前务必确保config.py文件内的文件路径是否正确；如data_path、output_dir等。目前默认使用的config为/workspace/code/Quant/teql/mqbconfig/fp32/coco_fastrcnn_res50_fpn_fp32_config.yaml
* object_detecion_qat/QAT_multi_GPU.py内有管道arg，附上了对应的描述。默认即可。后续再从.sh脚本出发。
* /workspace/code/Quant/teql/global_placeholder.py 内的“常值”变量需要主动改掉，否则一些路径会报错。目前看来是必须得加载backbone的预训练模型的。


# 杂项
* model_type:  

  4. 容器需要跑的实验：                    turn  device
    1. COCO
      1. RetinaNet18                                 20G+
        1. LSQ w4a4                        0r   d0  tools/starter_scripts/retinanet_18/qat/coco/dist_lsq_retinanet_18_coco_w4a4.sh
        2. LSQ w2a2                        0r   d1  tools/starter_scripts/retinanet_18/qat/coco/dist_lsq_retinanet_18_coco_w2a2.sh
        3. LSQ HQOD w4a4                   0r   d2  tools/starter_scripts/retinanet_18/qat/coco/dist_lsq_HQOD_retinanet_18_coco_w4a4.sh
        4. LSQ HQOD w2a2                   0r   d3  tools/starter_scripts/retinanet_18/qat/coco/dist_lsq_HQOD_retinanet_18_coco_w2a2.sh
      2. RetinaNet50                                 40G+
        1. LSQ w4a4                        0r   d4  tools/starter_scripts/retinanet_50/qat/coco/dist_lsq_retinanet_50_coco_w4a4.sh
        2. LSQ w2a2                        0r   d5  tools/starter_scripts/retinanet_50/qat/coco/dist_lsq_retinanet_50_coco_w2a2.sh
        3. LSQ HQOD w4a4                   0r   d6  tools/starter_scripts/retinanet_50/qat/coco/dist_lsq_HQOD_retinanet_50_coco_w4a4.sh
        4. LSQ HQOD w2a2                   2r   d0  tools/starter_scripts/retinanet_50/qat/coco/dist_lsq_HQOD_retinanet_50_coco_w2a2.sh
      3. YOLOX_s_LSQ                                 40G+
        1. LSQ w4a4                        1r   d0  tools/starter_scripts/yolox_s/qat/coco/dist_lsq_yolox_s_coco_w4a4.sh
        2. LSQ w2a2                        1r   d1  tools/starter_scripts/yolox_s/qat/coco/dist_lsq_yolox_s_coco_w2a2.sh
        3. LSQ HQOD w4a4                   1r   d2  tools/starter_scripts/yolox_s/qat/coco/dist_lsq_HQOD_yolox_s_coco_w4a4.sh
        4. LSQ HQOD w2a2                   1r   d3  tools/starter_scripts/yolox_s/qat/coco/dist_lsq_HQOD_yolox_s_coco_w2a2.sh
      4. RetinaNet18_AQD                             20G+
        1. fp32 pretrain                   0r   d7  tools/starter_scripts/retinanet_18/dist_train_retinanet_18_coco_AQD.sh
        2. LSQ HQOD w2a2                   1r   d4  tools/starter_scripts/retinanet_18/qat/coco/dist_lsq_retinanet_18_coco_w2a2_HQOD_AQD.sh
        3. LSQ HQOD w4a4                   1r   d5  tools/starter_scripts/retinanet_18/qat/coco/dist_lsq_retinanet_18_coco_w4a4_HQOD_AQD.sh
      5. ATSS50                                      40G+
        1. TQT w4a4                        1r   d6  tools/starter_scripts/atss_50/qat/coco/dist_tqt_atss_50_coco_w4a4.sh
        2. TQT w2a2                        2r   d1  tools/starter_scripts/atss_50/qat/coco/dist_tqt_atss_50_coco_w2a2.sh
        3. TQT HQOD w4a4                   1r   d7  tools/starter_scripts/atss_50/qat/coco/dist_tqt_HQOD_atss_50_coco_w4a4.sh
        4. TQT HQOD w2a2                   2r   d2  tools/starter_scripts/atss_50/qat/coco/dist_tqt_HQOD_atss_50_coco_w2a2.sh
      6. ATSS50                                      40G+
        1. LSQ w4a4                        2r   d3  tools/starter_scripts/atss_50/qat/coco/dist_lsq_atss_50_coco_w4a4.sh
        2. LSQ w2a2                        2r   d4  tools/starter_scripts/atss_50/qat/coco/dist_lsq_atss_50_coco_w2a2.sh
        3. LSQ HQOD w4a4                   2r   d5  tools/starter_scripts/atss_50/qat/coco/dist_lsq_HQOD_atss_50_coco_w4a4.sh
        4. LSQ HQOD w2a2                   2r   d6  tools/starter_scripts/atss_50/qat/coco/dist_lsq_HQOD_atss_50_coco_w2a2.sh

检查一遍pretrain是否加载了  AQD的要提前设置一下 √
batchsize √
device √

注意一下，要把master_port设置一下 √


commands_file=tools/starter_scripts/experiment1.txt tmux_s_name=hqod1 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment2.txt tmux_s_name=hqod2 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment3.txt tmux_s_name=hqod3 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment41.txt tmux_s_name=hqod41 bash tools/starter_scripts/main.sh





commands_file=tools/starter_scripts/experiment52.txt tmux_s_name=hqod52 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment53.txt tmux_s_name=hqod53 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment54.txt tmux_s_name=hqod54 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment55.txt tmux_s_name=hqod55 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment61.txt tmux_s_name=hqod61 bash tools/starter_scripts/main.sh
: 记得换文件
commands_file=tools/starter_scripts/experiment62.txt tmux_s_name=hqod62 bash tools/starter_scripts/main.sh


commands_file=tools/starter_scripts/experiment71.txt tmux_s_name=hqod71 bash tools/starter_scripts/main.sh

commands_file=tools/starter_scripts/experiment72.txt tmux_s_name=hqod72 bash tools/starter_scripts/main.sh

<!-- commands_file=tools/starter_scripts/experiment73.txt tmux_s_name=hqod73 bash tools/starter_scripts/main.sh -->





