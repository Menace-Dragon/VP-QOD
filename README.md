# VP-QOD
:dizzy: A Versatile Playground for Quantizing 2D Object Detectors :dizzy:

This project is built upon [MQBench](https://github.com/ModelTC/MQBench)(V0.0.7) and [MMDetection](https://github.com/open-mmlab/mmdetection/tree/2.x)(V2.28.2), with adjustments made to the codebases of both and deep integration achieved. Experimental setups for Vanilla PTQ, Advanced PTQ, and QAT, among others, can be conducted based on existing Detector & Config. However, this framework is only suitable for quantizing detectors in an **academic setting**. Alignment and deployment across various hardware platforms are not currently supported.


[NOTE] Our code is currently under modification and the full version will be release no later than July. Please be patient and stay tuned. :astonished:

本项目的初衷是打造一个可以对目标检测模型进行QAT训练实验的工程。在经历过许多曲折后，终确定以MQBench和MMDetection耦合的方式实现工程。后来又扩展到了能支持Vanilla PTQ、Advance PTQ实验。目前能支持的模型有RetinaNet、ATSS、SSDLite、YOLOX；


如果你喜欢该工程，记得点个星星！如果你有学术论文上的引用需求，请别忘了引用HQOD and QFOD:
