# Self Supervised Pretraining
A repository containing methods to pre train DNNs for image tasks
using self supervised learning.

## Self Supervised Methods:
1. [SimCLR](https://arxiv.org/abs/2002.05709)
2. [VICReg](https://arxiv.org/pdf/2105.04906.pdf)
2. [InPainting](https://arxiv.org/pdf/1604.07379.pdf)

## Run Examples
1. Train Resnet18 using SimCLR on stl10 dataset with 3 views
```
python main.py -type SimCLR --arch Resnet18 --dataset stl10 --n_views 3
```
2. Train Resnet43 using InPainting on cifar10 with mask size of 8
```
python main.py -type InPainting --arch Resnet34 --dataset cifar10 --mask_size 8
```

## Softwere Requirments
```
numpy=1.19.2=py38h54aff64_0
numpy-base=1.19.2=py38hfa32c7d_0
pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0
scikit-image=0.17.2=py38hdf5156a_0
scipy=1.4.1=py38h0b6359f_0
torchvision=0.9.1=py38_cu111

```