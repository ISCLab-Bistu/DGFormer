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

| Method       | Backbone   | Pretrain    | Lr schd | mIoU(code) | mIoU(paper) | Config                                              | Download                                                                                          |
|--------------|------------|-------------|:-----:|:----------:|:-----------:|-------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Semantic FPN | PVT-Tiny   | ImageNet-1K |   40K   |    36.6    |     35.7    | [config](configs/sem_fpn/PVT/fpn_pvt_t_ade20k_40k.py) | [log](https://drive.google.com/file/d/18NodMVuLWSHGjbUz6oMbtDnV2EddQEkC/view?usp=sharing) & [model](https://drive.google.com/file/d/13SaiOJ9hH7Wwg_AyeQ158LNV9vtjq6Lu/view?usp=sharing) |
| Semantic FPN | PVT-Small  | ImageNet-1K |  40K  |    41.9    |     39.8    | [config](configs/sem_fpn/PVT/fpn_pvt_s_ade20k_40k.py) | [log](https://drive.google.com/file/d/12FnAEQHWFa5K0wurEn1LcI6BZD7vexJV/view?usp=sharing) & [model](https://drive.google.com/file/d/13fy-FXAfYnHgHRaUiJWVBON670wFLIiD/view?usp=sharing) |
| Semantic FPN | PVT-Medium | ImageNet-1K |  40K  |    43.5    |     41.6    | [config](configs/sem_fpn/PVT/fpn_pvt_m_ade20k_40k.py) | [log](https://drive.google.com/file/d/1yNQLCax2Qx7xOQVL0v84KwhcNkWbp_s8/view?usp=sharing) & [model](https://drive.google.com/file/d/10ErJJZCcucnjjo8et2ivuHRzxbwc04y2/view?usp=sharing) |
| Semantic FPN | PVT-Large  | ImageNet-1K |  40K  |    43.5    |     42.1    | [config](configs/sem_fpn/PVT/fpn_pvt_l_ade20k_40k.py) | [log](https://drive.google.com/file/d/11-gMPyz19ExtfT3Tp8P40EYUKHd11ntA/view?usp=sharing) & [model](https://drive.google.com/file/d/1JkaXbTorIWLj9Oe5Dh6kzH_1vtrRFJRL/view?usp=sharing) |

| Method           | Backbone  | Pretrain    | Lr schd | box AP | Config                                               | Download                                                                                    |
|------------------|-----------|-------------|:-------:|:------:|:-------:|------------------------------------------------------|---------------------------------------------------------------------------------------------|
| RetinaNet        | DGFormer  | ImageNet-1K |    1x   |  37.6  ||[config](configs/dgformer/retinanet_dgformer_dga_effn_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1w5giZkGZ0raFnl6TE8V7G3vbFoeRDqWC/view?usp=sharing) & [model](https://drive.google.com/file/d/1ZbS-g3oqAChkYiDYTiZLzw61lal2pgbl/view?usp=sharing) |
| Mask R-CNN        | DGFormer  | ImageNet-1K |    1x   |  38.6  |    [config](configs/mask_rcnn_pvt_t_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1PE__Sp2tgKIYkJaUa0V8q-GSxyLkcWke/view?usp=sharing) & [model](https://drive.google.com/file/d/1JGcl7ZnDIf-qQjrCXVtb71XeRWOIu7Xf/view?usp=sharing) |



## Evaluation
To evaluate PVT-Small + RetinaNet (640x) on COCO val2017 on a single node with 8 gpus run:
```
dist_test.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```
This should give
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.593
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.408
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.212
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.416
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.544
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.329
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.583
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.721
```

## Training
To train PVT-Small + RetinaNet (640x) on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```
dist_train.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py 8
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/gfl_pvt_v2_b2_fpn_3x_mstrain_fp16.py
```
This should give
```
Input shape: (3, 1280, 800)
Flops: 260.65 GFLOPs
Params: 33.11 M
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
