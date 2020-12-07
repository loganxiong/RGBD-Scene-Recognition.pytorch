# PyTorch Image Classification

Following RGB-D Scene recognition networks are implemented using PyTorch.

* ResNet RGB-D
* ResNet-attention RGB-D
* TODO
  1. GCN based-- ACM: Adaptive Cross-Modal Graph Convolutional Neural Networks for RGB-D Scene Recognition
  2. ASK model-- ASK: Adaptively Selecting Key Local Features for RGB-D Scene Recognition
...


## Requirements

* Python >= 3.6
* PyTorch >= 1.0.0
* torchvision
* [tensorboardX](https://github.com/lanpa/tensorboardX) (optional)
* [NVIDIA Apex](https://github.com/NVIDIA/apex) (optional)



## Usage

```
$ ./train.py --arch resnet_preact --depth 56 --outdir results
```

### Use Cutout

```
$ ./train.py --arch resnet_preact --depth 56 --outdir results --use_cutout
```

### Use RandomErasing

```
$ ./train.py --arch resnet_preact --depth 56 --outdir results --use_random_erasing
```

### Use Mixup

```
$ ./train.py --arch resnet_preact --depth 56 --outdir results --use_mixup
```

### Use cosine annealing

```
$ ./train.py --arch wrn --outdir results --scheduler cosine
```



## Results on NYUD-v2

### Results using almost same settings as papers


| Model                                  | Mean-class Accuracy        | Pixel Accuracy    | Training Time |
|:---------------------------------------|:--------------------------:|:-----------------:|--------------:|
| ResNet-18 RGB                          |           62.91%           | 67.5%             |      20m    |
| ResNet-18 RGB-D                        |           66.93%           | 69.3%             |      36m    |
| ResNet-Attention-18 RGB-D              |           68.47%           | 71.7%             |      36m    |



Main codes are adapted from https://github.com/hysts/pytorch_image_classification


