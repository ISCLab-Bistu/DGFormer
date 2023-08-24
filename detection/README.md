# Applying DGFormer to Object Detection

Our detection code is developed on top of [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Usage

Install [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

or

```
pip install mmdet==2.13.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the configuration files:
```
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Results and models

- DGFormer on COCO

| Method       | Backbone   | Pretrain    | Lr schd |  box AP  | box AP-0.5  | Config                                              | Download                                                                                          |
|--------------|------------|-------------|:-----:|:----------:|:-----------:|-------------------------------------------------------|---------------------------------------------------------------------------------------------------|
|  RetinaNet   | DGFormer   | ImageNet-1K |   1x   |    37.6    |     58.0    | [config](configs/dgformer/retinanet_dgformer_dga_effn_fpn_1x_coco.py) | [log](work_dirs/retinanet_dgformer_dga_effn_fpn_1x_coco/20230721_082820.log.json) & [model](https://github.com/ISCLab-Bistu/DGFormer/releases/tag/seg-model/iter_80000.pth) |
| Mask R-CNN   | DGFormer   | ImageNet-1K |  1x  |    38.6    |     60.5    | [config](configs/dgformer/mask_rcnn_dgformer_fpn_1x_coco.py) | [log](work_dirs/mask_rcnn_dgformer_fpn_1x_coco/20230722_002453.log.json) & [model](https://github.com/ISCLab-Bistu/DGFormer/tree/main/detection/work_dirs/mask_rcnn_dgformer_fpn_1x_coco/latest.pth) |





## Evaluation
To evaluate DGFormer + RetinaNet on COCO val2017 on a single node with 8 gpus run:
```
dist_test.sh configs/dgformer/retinanet_dgformer_dga_effn_fpn_1x_coco.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```
This should give
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.580
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.401
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.226
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.407
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.497
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.549
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.549
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.549
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.361
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.585
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.707
```


## Training
To train DGFormer + RetinaNet on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```
dist_train.sh configs/dgformer/retinanet_dgformer_dga_effn_fpn_1x_coco.py 8
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/dgformer/retinanet_dgformer_dga_effn_fpn_1x_coco.py
```
This should give
```
Input shape: (3, 800, 1333)
Flops: 167.28 GFLOPs
Params: 12.64 M
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
