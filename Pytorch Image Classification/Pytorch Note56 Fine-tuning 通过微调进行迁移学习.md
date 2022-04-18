# Pytorch Note56 Fine-tuning 通过微调进行迁移学习

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

## 通过微调进行迁移学习

前面我们介绍了如何训练卷积神经网络进行图像分类，可能你已经注意到了，训练一个卷积网络是特别耗费时间的，特别是一个比较深的卷积网络，而且可能因为训练方法不当导致训练不收敛的问题，就算训练好了网络，还有可能出现过拟合的问题，所以由此可见能够得到一个好的模型有多么困难。

有的时候，我们的数据集还特别少，这对于我们来讲无异于雪上加霜，因为少的数据集意味着非常高的风险过拟合，那么我们有没有办法在某种程度上避免这个问题呢？其实现在有一种方法特别流行，大家一直在使用，那就是微调(fine-tuning)，在介绍微调之前，我们先介绍一个数据集 ImageNet。

## ImageNet
ImageNet 是一个计算机视觉系统识别项目，是目前世界上最大的图像识别数据库，由斯坦福大学组织建立，大约有 1500 万张图片，2.2 万中类别，其中 ISLVRC 作为其子集是学术界中使用最为广泛的公开数据集，一共有 1281167 张图片作为训练集，50000 张图片作为验证集，一共是 1000 分类，是目前测试网络性能的标杆。

我们说的这个数据集有什么用呢？我们又不关心这个数据集，但是对于我们自己的问题，我们有没有办法借助 ImageNet 中的数据集来提升模型效果，比如我们要做一个猫狗分类器，但是我们现在只有几百张图片，肯定不够，ImageNet 中有很多关于猫狗的图片，我们如果能够把这些图片拿过来训练，不就能够提升模型性能了吗？

但是这种做法太麻烦了，从 ImageNet 中寻找这些图片就很困难，如果做另外一个问题又要去找新的图片，所以直接找图片并不靠谱，那么有没有办法能够让我们不去找这些图片，又能使用这些图片呢？

非常简单，我们可以使用在 ImageNet 上训练好的网路，然后把这个网络在放到我们自己的数据集上进行训练不就好了。这个方法就叫做微调，这十分形象，相当于把一个已经很厉害的模型再微调到我们自己的数据集上来，也可称为迁移学习。

迁移学习的方法非常简单，将预训练的模型导入，然后将最后的分类全连接层换成适合我们自己问题的全连接层，然后开始训练，可以固定卷积层的参数，也可以不固定进行训练，最后能够非常有效的得到结果

pytorch 一直为我们内置了前面我们讲过的那些著名网络的预训练模型，不需要我们自己去 ImageNet 上训练了，模型都在 `torchvision.models` 里面，比如我们想使用预训练的 50 层 resnet，就可以用 `torchvision.models.resnet50(pretrained=True)` 来得到

下面我们用一个例子来演示一些微调

```python
import sys
sys.path.append('..')

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder
```

首先我们点击下面的[链接](https://download.pytorch.org/tutorial/hymenoptera_data.zip)获得数据集，终端可以使用

`wget https://download.pytorch.org/tutorial/hymenoptera_data.zip`

下载完成之后，我们将其解压放在程序的目录下，这是一个二分类问题，区分蚂蚁和蜜蜂

我们可以可视化一下图片，看看你能不能区分出他们来

```python
import os
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
root_path = './hymenoptera_data/train/'
im_list = [os.path.join(root_path, 'ants', i) for i in os.listdir(root_path + 'ants')[:4]]
im_list += [os.path.join(root_path, 'bees', i) for i in os.listdir(root_path + 'bees')[:5]]

nrows = 3
ncols = 3
figsize = (8, 8)
_, figs = plt.subplots(nrows, ncols, figsize=figsize)
for i in range(nrows):
    for j in range(ncols):
        figs[i][j].imshow(Image.open(im_list[nrows*i+j]))
        figs[i][j].axes.get_xaxis().set_visible(False)
        figs[i][j].axes.get_yaxis().set_visible(False)
plt.show()
```

> ![在这里插入图片描述](https://img-blog.csdnimg.cn/565a27379e39430bb2196a86974d8e4f.png)

## 定义数据预处理

```python
# 定义数据预处理
train_tf = tfs.Compose([
    tfs.RandomResizedCrop(224),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 使用 ImageNet 的均值和方差
])

valid_tf = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

```python
# 使用 ImageFolder 定义数据集
train_set = ImageFolder('./hymenoptera_data/train/', train_tf)
valid_set = ImageFolder('./hymenoptera_data/val/', valid_tf)
# 使用 DataLoader 定义迭代器
train_data = DataLoader(train_set, 25, True, num_workers=4)
valid_data = DataLoader(valid_set, 32, False, num_workers=4)
```

## 使用预训练的模型

```python
# 使用预训练的模型
net = models.resnet50(pretrained=True)
print(net)
```

> ```python
> ResNet(
>   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
>   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>   (relu): ReLU(inplace=True)
>   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
>   (layer1): Sequential(
>     (0): Bottleneck(
>       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>       (downsample): Sequential(
>         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       )
>     )
>     (1): Bottleneck(
>       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (2): Bottleneck(
>       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>   )
>   (layer2): Sequential(
>     (0): Bottleneck(
>       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>       (downsample): Sequential(
>         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
>         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       )
>     )
>     (1): Bottleneck(
>       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (2): Bottleneck(
>       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (3): Bottleneck(
>       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>   )
>   (layer3): Sequential(
>     (0): Bottleneck(
>       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>       (downsample): Sequential(
>         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
>         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       )
>     )
>     (1): Bottleneck(
>       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (2): Bottleneck(
>       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (3): Bottleneck(
>       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (4): Bottleneck(
>       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (5): Bottleneck(
>       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>   )
>   (layer4): Sequential(
>     (0): Bottleneck(
>       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>       (downsample): Sequential(
>         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
>         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       )
>     )
>     (1): Bottleneck(
>       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>     (2): Bottleneck(
>       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
>       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>       (relu): ReLU(inplace=True)
>     )
>   )
>   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
>   (fc): Linear(in_features=2048, out_features=1000, bias=True)
> )
> ```

```python
# 打出第一层的权重
print(net.conv1.weight)
```

## 将最后的全连接层改成二分类

```python
# 将最后的全连接层改成二分类
net.fc = nn.Linear(2048, 2)
```

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
```

## 训练模型

```python
from utils import old_train
Acc, Loss, Lr = old_train(net, train_data, valid_data, 20, optimizer, criterion)
```

> ```python
> Epoch [  1/ 20]  Train Loss:0.590047  Train Acc:68.32% Test Loss:0.414566  Test Acc:79.43%  Learning Rate:0.010000	Time 00:20
> Epoch [  2/ 20]  Train Loss:0.401846  Train Acc:81.60% Test Loss:0.198879  Test Acc:93.58%  Learning Rate:0.010000	Time 00:13
> Epoch [  3/ 20]  Train Loss:0.234594  Train Acc:90.82% Test Loss:0.170969  Test Acc:94.83%  Learning Rate:0.010000	Time 00:14
> Epoch [  4/ 20]  Train Loss:0.222330  Train Acc:90.80% Test Loss:0.167808  Test Acc:95.28%  Learning Rate:0.010000	Time 00:15
> Epoch [  5/ 20]  Train Loss:0.156525  Train Acc:93.87% Test Loss:0.154961  Test Acc:95.28%  Learning Rate:0.010000	Time 00:15
> Epoch [  6/ 20]  Train Loss:0.101500  Train Acc:97.07% Test Loss:0.151929  Test Acc:94.65%  Learning Rate:0.010000	Time 00:14
> Epoch [  7/ 20]  Train Loss:0.125870  Train Acc:95.87% Test Loss:0.135658  Test Acc:95.45%  Learning Rate:0.010000	Time 00:14
> Epoch [  8/ 20]  Train Loss:0.104806  Train Acc:97.20% Test Loss:0.128553  Test Acc:95.45%  Learning Rate:0.010000	Time 00:13
> Epoch [  9/ 20]  Train Loss:0.134583  Train Acc:94.00% Test Loss:0.140324  Test Acc:94.83%  Learning Rate:0.010000	Time 00:14
> Epoch [ 10/ 20]  Train Loss:0.102864  Train Acc:96.40% Test Loss:0.136546  Test Acc:94.83%  Learning Rate:0.010000	Time 00:16
> Epoch [ 11/ 20]  Train Loss:0.069651  Train Acc:98.00% Test Loss:0.140122  Test Acc:94.65%  Learning Rate:0.010000	Time 00:17
> Epoch [ 12/ 20]  Train Loss:0.091097  Train Acc:96.80% Test Loss:0.126392  Test Acc:96.08%  Learning Rate:0.010000	Time 00:15
> Epoch [ 13/ 20]  Train Loss:0.099716  Train Acc:94.80% Test Loss:0.131765  Test Acc:94.65%  Learning Rate:0.010000	Time 00:16
> Epoch [ 14/ 20]  Train Loss:0.084522  Train Acc:97.87% Test Loss:0.124556  Test Acc:95.45%  Learning Rate:0.010000	Time 00:15
> Epoch [ 15/ 20]  Train Loss:0.077379  Train Acc:96.40% Test Loss:0.130561  Test Acc:96.08%  Learning Rate:0.010000	Time 00:14
> Epoch [ 16/ 20]  Train Loss:0.066754  Train Acc:97.60% Test Loss:0.117638  Test Acc:95.45%  Learning Rate:0.010000	Time 00:15
> Epoch [ 17/ 20]  Train Loss:0.093839  Train Acc:97.07% Test Loss:0.107425  Test Acc:94.83%  Learning Rate:0.010000	Time 00:16
> Epoch [ 18/ 20]  Train Loss:0.044568  Train Acc:98.40% Test Loss:0.117701  Test Acc:95.45%  Learning Rate:0.010000	Time 00:16
> Epoch [ 19/ 20]  Train Loss:0.040038  Train Acc:99.60% Test Loss:0.125794  Test Acc:95.45%  Learning Rate:0.010000	Time 00:14
> Epoch [ 20/ 20]  Train Loss:0.046786  Train Acc:98.40% Test Loss:0.129580  Test Acc:95.45%  Learning Rate:0.010000	Time 00:14
> ```

```python
from utils import plot_history
plot_history(20, Acc, Loss, Lr)
```

> ![在这里插入图片描述](https://img-blog.csdnimg.cn/e8721860272f4940afe6609dbf96db9a.png)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20ca3dd64e964cdc93a007a3fc04b241.png)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/11409cd0e5d44f80a59e8c2c7804a328.png)

## 预测

下面我们来可视化预测的结果

```python
net = net.eval() # 将网络改为预测模式
```

读一张蚂蚁的图片

```python
im1 = Image.open('./hymenoptera_data/train/ants/0013035.jpg')
im1
```

> ![在这里插入图片描述](https://img-blog.csdnimg.cn/09a5e60d03c4433798bef024ab6bd721.png)

```python
im = valid_tf(im1) # 做数据预处理
out = net(Variable(im.unsqueeze(0)).cuda())
pred_label = out.max(1)[1].data[0]
print('predict label: {}'.format(train_set.classes[pred_label]))
```

> ```python
> predict label: ants
> ```



可以看到预测的结果是对的

## 小练习

**小练习：看看上面的网络预测过程，多尝试几张图片进行预测**

### 使用预训练的模型

```python
# 保持前面的卷积层参数不变
net = models.resnet50(pretrained=True)
for param in net.parameters():
    param.requires_grad = False # 将模型的参数设置为不求梯度
net.fc = nn.Linear(2048, 2)

optimizer = torch.optim.SGD(net.fc.parameters(), lr=1e-2, weight_decay=1e-4)
```

```python
Acc, Loss, Lr = old_train(net, train_data, valid_data, 20, optimizer, criterion)
```

> ```python
> Epoch [  1/ 20]  Train Loss:0.698645  Train Acc:58.46% Test Loss:0.337921  Test Acc:89.38%  Learning Rate:0.010000	Time 00:10
> Epoch [  2/ 20]  Train Loss:0.405959  Train Acc:78.97% Test Loss:0.367483  Test Acc:81.25%  Learning Rate:0.010000	Time 00:11
> Epoch [  3/ 20]  Train Loss:0.498462  Train Acc:75.60% Test Loss:0.242778  Test Acc:94.47%  Learning Rate:0.010000	Time 00:10
> Epoch [  4/ 20]  Train Loss:0.260202  Train Acc:92.15% Test Loss:0.230207  Test Acc:90.45%  Learning Rate:0.010000	Time 00:10
> Epoch [  5/ 20]  Train Loss:0.264511  Train Acc:92.27% Test Loss:0.339598  Test Acc:84.50%  Learning Rate:0.010000	Time 00:10
> Epoch [  6/ 20]  Train Loss:0.235563  Train Acc:95.07% Test Loss:0.191060  Test Acc:94.83%  Learning Rate:0.010000	Time 00:10
> Epoch [  7/ 20]  Train Loss:0.261545  Train Acc:87.35% Test Loss:0.198911  Test Acc:93.85%  Learning Rate:0.010000	Time 00:09
> Epoch [  8/ 20]  Train Loss:0.305734  Train Acc:87.22% Test Loss:0.178782  Test Acc:95.45%  Learning Rate:0.010000	Time 00:09
> Epoch [  9/ 20]  Train Loss:0.186689  Train Acc:92.40% Test Loss:0.187927  Test Acc:93.68%  Learning Rate:0.010000	Time 00:10
> Epoch [ 10/ 20]  Train Loss:0.267986  Train Acc:87.62% Test Loss:0.193842  Test Acc:93.68%  Learning Rate:0.010000	Time 00:09
> Epoch [ 11/ 20]  Train Loss:0.186078  Train Acc:94.40% Test Loss:0.176513  Test Acc:95.28%  Learning Rate:0.010000	Time 00:10
> Epoch [ 12/ 20]  Train Loss:0.163462  Train Acc:95.35% Test Loss:0.190156  Test Acc:93.85%  Learning Rate:0.010000	Time 00:10
> Epoch [ 13/ 20]  Train Loss:0.177957  Train Acc:93.87% Test Loss:0.189092  Test Acc:93.22%  Learning Rate:0.010000	Time 00:09
> Epoch [ 14/ 20]  Train Loss:0.158989  Train Acc:95.35% Test Loss:0.183712  Test Acc:92.95%  Learning Rate:0.010000	Time 00:10
> Epoch [ 15/ 20]  Train Loss:0.207236  Train Acc:92.00% Test Loss:0.178122  Test Acc:91.70%  Learning Rate:0.010000	Time 00:10
> Epoch [ 16/ 20]  Train Loss:0.213432  Train Acc:91.35% Test Loss:0.174111  Test Acc:92.95%  Learning Rate:0.010000	Time 00:10
> Epoch [ 17/ 20]  Train Loss:0.165621  Train Acc:93.47% Test Loss:0.223445  Test Acc:90.20%  Learning Rate:0.010000	Time 00:11
> Epoch [ 18/ 20]  Train Loss:0.173697  Train Acc:95.60% Test Loss:0.189305  Test Acc:92.43%  Learning Rate:0.010000	Time 00:10
> Epoch [ 19/ 20]  Train Loss:0.161865  Train Acc:94.82% Test Loss:0.195919  Test Acc:92.43%  Learning Rate:0.010000	Time 00:10
> Epoch [ 20/ 20]  Train Loss:0.201205  Train Acc:90.80% Test Loss:0.147868  Test Acc:95.45%  Learning Rate:0.010000	Time 00:09
> ```

可以看到只训练验证集的准确率也可以达到比较高，但是 loss 跳动比较大，因为更新的参数太少了，只有全连接层的参数

### 不使用预训练的模型

```python
# 不使用预训练的模型
net = models.resnet50()
net.fc = nn.Linear(2048, 2)

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
```

```python
# 打出第一层的权重
print(net.conv1.weight)
```

```python
Acc, Loss, Lr = old_train(net, train_data, valid_data, 20, optimizer, criterion)
```

> ```python
> Epoch [  1/ 20]  Train Loss:3.789299  Train Acc:48.63% Test Loss:3.342654  Test Acc:56.25%  Learning Rate:0.010000	Time 00:13
> Epoch [  2/ 20]  Train Loss:2.933025  Train Acc:52.04% Test Loss:1.512564  Test Acc:51.95%  Learning Rate:0.010000	Time 00:14
> Epoch [  3/ 20]  Train Loss:1.754535  Train Acc:56.44% Test Loss:1.211804  Test Acc:57.03%  Learning Rate:0.010000	Time 00:16
> Epoch [  4/ 20]  Train Loss:1.517341  Train Acc:56.17% Test Loss:1.086264  Test Acc:58.10%  Learning Rate:0.010000	Time 00:17
> Epoch [  5/ 20]  Train Loss:1.025034  Train Acc:58.21% Test Loss:2.929851  Test Acc:56.25%  Learning Rate:0.010000	Time 00:15
> Epoch [  6/ 20]  Train Loss:1.249750  Train Acc:55.26% Test Loss:1.022647  Test Acc:56.25%  Learning Rate:0.010000	Time 00:14
> Epoch [  7/ 20]  Train Loss:0.941940  Train Acc:60.99% Test Loss:1.043641  Test Acc:61.40%  Learning Rate:0.010000	Time 00:14
> Epoch [  8/ 20]  Train Loss:0.674788  Train Acc:67.39% Test Loss:0.872586  Test Acc:66.65%  Learning Rate:0.010000	Time 00:15
> Epoch [  9/ 20]  Train Loss:0.652783  Train Acc:67.09% Test Loss:1.197592  Test Acc:68.15%  Learning Rate:0.010000	Time 00:14
> Epoch [ 10/ 20]  Train Loss:0.796914  Train Acc:57.71% Test Loss:1.370682  Test Acc:52.75%  Learning Rate:0.010000	Time 00:14
> Epoch [ 11/ 20]  Train Loss:0.772965  Train Acc:67.14% Test Loss:3.495006  Test Acc:56.25%  Learning Rate:0.010000	Time 00:16
> Epoch [ 12/ 20]  Train Loss:1.378554  Train Acc:54.74% Test Loss:2.399912  Test Acc:71.72%  Learning Rate:0.010000	Time 00:15
> Epoch [ 13/ 20]  Train Loss:1.027114  Train Acc:65.24% Test Loss:1.286589  Test Acc:72.97%  Learning Rate:0.010000	Time 00:16
> Epoch [ 14/ 20]  Train Loss:0.920177  Train Acc:64.69% Test Loss:1.315045  Test Acc:70.05%  Learning Rate:0.010000	Time 00:16
> Epoch [ 15/ 20]  Train Loss:0.641859  Train Acc:70.06% Test Loss:0.649205  Test Acc:62.48%  Learning Rate:0.010000	Time 00:15
> Epoch [ 16/ 20]  Train Loss:0.603328  Train Acc:68.84% Test Loss:0.824622  Test Acc:68.53%  Learning Rate:0.010000	Time 00:15
> Epoch [ 17/ 20]  Train Loss:0.722805  Train Acc:61.54% Test Loss:1.698193  Test Acc:56.25%  Learning Rate:0.010000	Time 00:14
> Epoch [ 18/ 20]  Train Loss:0.707589  Train Acc:66.84% Test Loss:0.767266  Test Acc:62.65%  Learning Rate:0.010000	Time 00:15
> Epoch [ 19/ 20]  Train Loss:0.846025  Train Acc:63.52% Test Loss:0.677937  Test Acc:69.30%  Learning Rate:0.010000	Time 00:16
> Epoch [ 20/ 20]  Train Loss:0.638594  Train Acc:71.12% Test Loss:0.693861  Test Acc:68.53%  Learning Rate:0.010000	Time 00:16
> ```

通过上面的结果可以看到，使用预训练的模型能够非常快的达到 95% 左右的验证集准确率，而不使用预训练模型只能到 70% 左右的验证集准确率，所以使用一个预训练的模型能够在较小的数据集上也取得一个非常好的效果，因为对于图片识别分类任务，最底层的卷积层识别的都是一些通用的特征，比如形状、纹理等等，所以对于很多图像分类、识别任务，都可以使用预训练的网络得到更好的结果。

