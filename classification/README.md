# Dual-Granularity Transformer: A Novel Dual-Granularity Lightweight Transformer for Vision Tasks

Our classification code is developed on top of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [deit](https://github.com/facebookresearch/deit).




## Todo List
- DGFormer + ImageNet-22K pre-training.

## Usage

First, clone the repository locally:
```
git clone https://github.com/ISCLab-Bistu/DGFormer.git
```
Then, install PyTorch 1.6.0+ and torchvision 0.7.0+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Model Zoo

- DGFormer on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) | Config                                   | Download                                                                                   |
|------------------|:----:|:-----:|:-----------:|------------------------------------------|--------------------------------------------------------------------------------------------|
| PVT-V2-B0        |  224 |  72.8 |     3.4     | [config](configs/dgformer/dgformer_dga_effn.py)    | 12.9M  [[GitHub]](https://github.com/ISCLab-Bistu/DGFormer/tree/main/classification/checkpoints/DGFormer_DGA_EFFN/dgformer.pth) |




## Evaluation
To evaluate a pre-trained DGFormer on ImageNet val with a single GPU run:
```
sh dist_train.sh configs/dgformer/dgformer_dga_effn.py 1 --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```
This should give
```
* Acc@1 72.823 Acc@5 91.496 loss 1.288
Accuracy of the network on the 50000 test images: 72.8%
```

## Training
To train DGFormer on ImageNet on a single node with 8 gpus for 300 epochs run:

```
sh dist_train.sh configs/dgformer/dgformer_dga_effn.py 8 --data-path /path/to/imagenet
```

## Calculating FLOPS & Params

```
python get_flops.py dgformer
```
This should give
```
Input shape: (3, 224, 224)
Flops: 0.4 GFLOPs
Params: 3.4 M
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
