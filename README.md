# PyTorch Image Classification

Following papers are implemented using PyTorch.

* ResNet (1512.03385)
* ResNet-preact (1603.05027)
* WRN (1605.07146)
* DenseNet (1608.06993)
* PyramidNet (1610.02915)
* ResNeXt (1611.05431)
* shake-shake (1705.07485)
* LARS (1708.03888, 1801.03137)
* Cutout (1708.04552)
* Random Erasing (1708.04896)
* SENet (1709.01507)
* Mixup (1710.09412)
* Dual-Cutout (1802.07426)
* RICAP (1811.09030)

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

![](figures/cifar10/test_errors.png)

| Model                                  | Mean-class Accuracy        | Pixel Accuracy    | Training Time |
|:---------------------------------------|:--------------------------:|:-----------------:|--------------:|
| ResNet-18 RGB                          |           67.29%           | 73.5%             |      1h20m    |
| ResNet-18 RGB-D                        |           66.52%           | 74.3%             |      3h06m    |
| ResNet-preact-18 RGB-D                 |           66.47%           | 76.3%             |      3h05m    |





