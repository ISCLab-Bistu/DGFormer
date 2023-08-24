# Applying DGFormer to Semantic Segmentation

Here, we take [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) as an example, applying PVT to SemanticFPN.



## Usage

Install MMSegmentation.


## Data preparation

Preparing ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) in MMSegmentation.


## Results and models

| Method       | Backbone   | Pretrain    | Iters | mIoU(code) | mIoU(paper) | Config                                                | Download                                                                                          |
|--------------|------------|-------------|:-----:|:----------:|:-----------:|-------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Semantic FPN | DGFormer   | ImageNet-1K |  80K  |    39.16   |     39.2    | [config](configs/sem_fpn/DGFormer/fpn_dgformer_ade20k_80k.py) | [log](work_dirs/fpn_dgformer_ade20k_80k/20230715_231918.log.json) & [model](https://github.com/ISCLab-Bistu/DGFormer/releases/tag/seg-model/latest.pth) |


## Evaluation
To evaluate DGFormer + Semantic FPN on a single node with 8 gpus run:
```
dist_test.sh configs/sem_fpn/DGFormer/fpn_dgformer_ade20k_80k.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```


## Training
To train DGFormer + Semantic FPN on a single node with 8 gpus run:

```
dist_train.sh configs/sem_fpn/DGFormer/fpn_dgformer_ade20k_80k.py 8
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
