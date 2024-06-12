# Introduction :dizzy:
  VP-QOD is a versatile playground for quantizing 2D object detectors based on MQBench and MMDetection.

  This project is built upon [MQBench](https://github.com/ModelTC/MQBench)(V0.0.7) and [MMDetection](https://github.com/open-mmlab/mmdetection/tree/2.x)(V2.28.2), with adjustments made to the codebases of both and deep integration achieved. Experimental setups for Vanilla PTQ, Advanced PTQ, and QAT, among others, can be conducted based on existing Detector & Config. However, this framework is only suitable for quantizing detectors in an **academic setting**. Alignment and deployment across various hardware platforms are not supported.

If you appreciate this project, remember to give it a star!

<details open>
<summary>Major features</summary>

- **Supported Quant Algorithms**

  The project currently supports Quantization-Aware Training (QAT) algorithms for object detection models. The available QAT algorithms include:
  
| Model | Paper URL |
| :------:  | :-------------------------------------------------------------------- |
| DSQ      <br> [ICCV'2019]  | [Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks](https://arxiv.org/abs/1908.05033) |
| LSQ   <br> [ICLR'2020]  | [Learned step size quantization](https://arxiv.org/abs/1902.08153) |
| TQT <br> [MLSys'2020]  | [Trained quantization thresholds for accurate and efficient fixed-point inference of deep neural networks](https://arxiv.org/abs/1903.08066) |
| AQD     <br> [CVPR'2021] | [Aqd: Towards accurate quantized object detection](https://arxiv.org/abs/2007.06919) |
| HQOD (Ours) <br> [ICME'2024] | [HQOD: Harmonious Quantization for Object Detection](https://arxiv.org/abs/2107.08430) |


- **Supported Object Detectors**

  The project currently supports the following object detectors to be quantized:
  
| Model | Paper URL |
| :------:  | :-------------------------------------------------------------------- |
| SSDLite   <br> [ECCV'2016]  | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) |
| RetinaNet <br> [ICCV'2017]  | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) |
| ATSS      <br> [CVPR'2020]  | [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424) |
| YOLOX     <br> [ArXiv'2021] | [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430) |


- **Flexible development**

  This project effectively assimilates and integrates the characteristics of MQBench and MMDetection, streamlining the adaptation process between detectors and torch.fx, It facilitates the convenient extension of other quantization algorithms and the extension of detectors.

- **Future plans**

  - [ ] Support vanilla PTQ (which only conduct calibration).
  - [ ] Support advance PTQ (which extra utilize reconstruction), such as AdaRound, BRECQ and PD-Quant.

</details>

# Get Started
**Step 0.** Prerequisites:
 - torch with 1.9.0
 - python with 3.8.8
 - cuda with 11.2
 - dataset
   - MS COCO 2017
   - (Optional) PASCAL VOC 2007 + 2012
**Step 1.** `cd` to the **root directory of the project** and install the local MMDet.
```
# Prepare mmcv-full.
pip install -U openmim --use-feature=2020-resolver
mim install mmcv-full

# install local MMDet project.
pip install -v -e .

# install other packages.

```

**Step 2.** Modify the dataset path. 
Open `configs/_base_/datasets/coco_detection.py` then modify:
```
...
data_root = 'your/path/to/coco2017/'
...
```
**Step 3.** Modify the pretrained ckpt path. Treat Retinanet-resnet18 as example: 
Open `configs/_base_/datasets/coco_detection.py` then modify:
```
...
# Modify the backbone ckpt path.
init_cfg=dict(type='Pretrained', checkpoint='your/path/to/resnet18-5c106cde.pth')),
...
# Modify the fp32 ckpt path.
load_from = 'your/path/to/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth'
```
Note that all the pretrained ckpts are downloaded from https://mmdetection.readthedocs.io/en/v2.28.2/model_zoo.html.

**Step 4.** Launch your quant procedure. Treat Retinanet-resnet18 as example:
```
root@Long:/workspace/code/Quant/hqod# tools/starter_scripts/retinanet_18/qat/coco/dist_lsq_HQOD_retinanet_18_coco_w4a4.sh
```

**Step A.** If you want to modify the quant setting, please refer to `mqbconfig` folder.

**Step B.** If you want to modify the quant setting, please refer to `mqbconfig` folder.


To be continue :astonished:



如果你有学术论文上的引用需求，请别忘了引用HQOD and QFOD:

HQOD指标放榜（包括参数文件放链接）：
