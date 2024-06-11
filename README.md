# Introduction :dizzy:
  VP-QOD is a versatile playground for quantizing 2D object detectors based on MQBench and MMDetection.

  This project is built upon [MQBench](https://github.com/ModelTC/MQBench)(V0.0.7) and [MMDetection](https://github.com/open-mmlab/mmdetection/tree/2.x)(V2.28.2), with adjustments made to the codebases of both and deep integration achieved. Experimental setups for Vanilla PTQ, Advanced PTQ, and QAT, among others, can be conducted based on existing Detector & Config. However, this framework is only suitable for quantizing detectors in an **academic setting**. Alignment and deployment across various hardware platforms are not supported.

如果你喜欢该工程，记得点个星星！如果你有学术论文上的引用需求，请别忘了引用HQOD and QFOD:

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
:astonished:
项目描述：简洁地介绍项目的目的和内容。

安装指南：如何安装和设置项目以进行开发。

使用说明：如何使用项目的核心功能。

快速示例：提供一个简单的使用示例。

快捷键：如果是工具类项目，列出可用的快捷键。

路线图：展示项目未来的开发计划。

贡献指南：如何为项目贡献代码或报告问题。

学习资源：提供有关项目相关领域的学习资源链接。

版权和许可：说明项目的版权和使用许可。


HQOD指标放榜（包括参数文件放链接）：
