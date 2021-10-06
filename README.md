# Self Supervised Pretraining
A repository containing methods to pre train DNNs for image tasks
using self supervised learning.

## Self Supervised Methods:
1. [SimCLR](https://arxiv.org/abs/2002.05709)
2. [VICReg](https://arxiv.org/pdf/2105.04906.pdf)
3. [SimSiam](https://arxiv.org/pdf/2011.10566v1.pdf)
2. [InPainting](https://arxiv.org/pdf/1604.07379.pdf)

## Run Examples
1. Train Resnet18 using SimCLR on stl10 dataset with 3 views
```
python main.py -type SimCLR --arch Resnet18 --dataset stl10 --n_views 3
```
2. Train Resnet34using InPainting on cifar10 with mask size of 8
```
python main.py -type InPainting --arch Resnet34 --dataset cifar10 --mask_size 8
```
3. Train Resnet18 using VICReg and LARS optimizer
```
 python main.py -type VICReg --arch Resnet18 --dataset cifar10  --batch_size 512 --lambd 1000 --mu 3 --nu 1e5 --opti LARS
```

HELP:
```
  -h, --help            show this help message and exit
  -type {VICReg,SimCLR,SimSiam,InPainting}
                        Select the pretraining type
  --dataset DATASET     dataset name: stl10 or cifar10, if costum place it in the data folder
  --arch {Resnet18,Resnet34}
                        achitecture type
  --workers WORKERS     number of data loading workers for each loader
  --epochs EPOCHS       number of total epochs to run
  --batch_size BATCH_SIZE
                        mini-batch size (default: 512), this is the total
  --opti {SGD,Adam,LARS}
                        Select the otimizer type
  --lr LR               initial learning rate
  --grade_scale {True,False}
                        To use or not to use grade scaling (defualt: False)
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 1e-4)
  --momentum MOMENTUM   SGD optimizer momentum (default:0.9)
  --seed SEED           seed for initializing training.
  --temperature TEMPERATURE
                        softmax temperature (default: 0.07)
  --n_views N_VIEWS     Number of views for contrastive learning.
  --projection_size PROJECTION_SIZE
                        projection head size (default: 128)
  --mask_size MASK_SIZE
                        Maksed area size (size, size).
  --lambd LAMBD         VICReg lambd (default: 1000)
  --mu MU               VICReg mu (default: 2)
  --nu NU               VICReg nu (default: 1e5)
  --device DEVICE       Gpu index.
```

## Software Requirments
```
numpy=1.19.2=py38h54aff64_0
numpy-base=1.19.2=py38hfa32c7d_0
pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0
scikit-image=0.17.2=py38hdf5156a_0
scipy=1.4.1=py38h0b6359f_0
torchvision=0.9.1=py38_cu111

```
