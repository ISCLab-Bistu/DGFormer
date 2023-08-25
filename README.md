# DGFormer
A Novel Dual-Granularity Lightweight Transformer for Vision Tasks


## Introduction


Transformer-based networks have revolutionized visual tasks, driving substantial progress through continuous innovation. However, the widespread adoption of Vision Transformers (ViT) has been hindered by their high computational and parameter requirements, limiting their feasibility for resource-constrained mobile and edge computing devices. Moreover, existing lightweight ViTs encounter challenges in effectively capturing different granular features, efficiently extracting local features, and incorporating the inductive bias inherent in convolutional neural networks. These limitations have a noticeable impact on overall performance.

In this paper, we propose an efficient ViT called DGFormer (Dual-Granularity Former), which mitigates these limitations. Specifically, DGFormer consists of two novel modules: Dual-Granularity Attention (DG Attention) and Efficient Feed-Forward Network (Efficient FFN). These modules enable efficient extraction and modeling of global and local features, significantly enhancing the performance of lightweight Transformer-based models.

Through extensive evaluation across various vision tasks, such as image classification, object detection, and semantic segmentation, we demonstrate that DGFormer outperforms popular models like PVTv2 and Swin Transformer, even with a reduced parameter count. This not only showcases the robust performance of DGFormer but also highlights its potential applicability in real-world scenarios.

<p align="center">
    <img src="https://github.com/ISCLab-Bistu/DGFormer/blob/main/Backbone.png" />
</p>



<p align = "center">

Dual-Granularity Transformer (DGFormer). 

</p>



The structure of the DGFormer network is shown in above picture. 
