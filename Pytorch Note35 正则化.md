# Pytorch Note35 正则化

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

## 正则化

前面我们讲了数据增强和 dropout，而在实际使用中，现在的网络往往不使用 dropout，而是用另外一个技术，叫正则化。

正则化是机器学习中提出来的一种方法，有 L1 和 L2 正则化，目前使用较多的是 L2 正则化，引入正则化相当于在 loss 函数上面加上一项，比如

$$
f = loss + \lambda \sum_{p \in params} ||p||_2^2
$$

就是在 loss 的基础上加上了参数的二范数作为一个正则化，我们在训练网络的时候，不仅要最小化 loss 函数，同时还要最小化参数的二范数，也就是说我们会对参数做一些限制，不让它变得太大。

如果我们对新的损失函数 f 求导进行梯度下降，就有

$$
\frac{\partial f}{\partial p_j} = \frac{\partial loss}{\partial p_j} + 2 \lambda p_j
$$

那么在更新参数的时候就有

$$
p_j \rightarrow p_j - \eta (\frac{\partial loss}{\partial p_j} + 2 \lambda p_j) = p_j - \eta \frac{\partial loss}{\partial p_j} - 2 \eta \lambda p_j
$$
可以看到 $p_j - \eta \frac{\partial loss}{\partial p_j}$ 和没加正则项要更新的部分一样，而后面的 $2\eta \lambda p_j$ 就是正则项的影响，可以看到加完正则项之后会对参数做更大程度的更新，这也被称为权重衰减(weight decay)，在 pytorch 中正则项就是通过这种方式来加入的，比如想在随机梯度下降法中使用正则项，或者说权重衰减，`torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)` 就可以了，这个 `weight_decay` 系数就是上面公式中的 $\lambda$，非常方便

注意正则项的系数的大小非常重要，如果太大，会极大的抑制参数的更新，导致欠拟合，如果太小，那么正则项这个部分基本没有贡献，所以选择一个合适的权重衰减系数非常重要，这个需要根据具体的情况去尝试，初步尝试可以使用 `1e-4` 或者 `1e-3` 

### 加正则项

下面我们在训练 cifar 10 中添加正则项

```python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision import transforms as tfs
from train import *
```

```python
def data_tf(x):
    im_aug = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)


net = ResNet18()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-4) # 增加正则项
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=0.5,min_lr=1e-6)
```

```python
train(net, train_data, test_data, 20, optimizer, criterion,scheduler)
```

```python
 Epoch [  1/ 20 ]  train_loss:1.612872  train_acc:40.76%  test_loss:1.389403  test_acc:49.19%  	Time 02:08
 Epoch [  2/ 20 ]  train_loss:1.208912  train_acc:56.29%  test_loss:1.171614  test_acc:57.79%  	Time 02:02
 Epoch [  3/ 20 ]  train_loss:0.994738  train_acc:64.65%  test_loss:1.062282  test_acc:61.93%  	Time 02:01
 Epoch [  4/ 20 ]  train_loss:0.833706  train_acc:70.53%  test_loss:1.089088  test_acc:61.54%  	Time 02:02
 Epoch [  5/ 20 ]  train_loss:0.686806  train_acc:76.15%  test_loss:1.028853  test_acc:64.65%  	Time 02:04
 Epoch [  6/ 20 ]  train_loss:0.553232  train_acc:81.16%  test_loss:0.900885  test_acc:69.09%  	Time 02:05
 Epoch [  7/ 20 ]  train_loss:0.405403  train_acc:86.78%  test_loss:0.980572  test_acc:67.38%  	Time 02:01
 Epoch [  8/ 20 ]  train_loss:0.264943  train_acc:92.11%  test_loss:1.192778  test_acc:65.36%  	Time 02:03
 Epoch [  9/ 20 ]  train_loss:0.139099  train_acc:96.55%  test_loss:1.221482  test_acc:65.70%  	Time 02:01
 Epoch [ 10/ 20 ]  train_loss:0.060923  train_acc:99.01%  test_loss:1.050609  test_acc:70.47%  	Time 02:00
 Epoch [ 11/ 20 ]  train_loss:0.020524  train_acc:99.91%  test_loss:1.064708  test_acc:71.45%  	Time 01:59
 Epoch [ 12/ 20 ]  train_loss:0.009629  train_acc:99.98%  test_loss:1.073604  test_acc:71.80%  	Time 01:58
 Epoch [ 13/ 20 ]  train_loss:0.006375  train_acc:99.99%  test_loss:1.061980  test_acc:72.16%  	Time 02:07
 Epoch [ 14/ 20 ]  train_loss:0.004555  train_acc:100.00%  test_loss:1.079033  test_acc:72.42%  	Time 02:11
 Epoch [ 15/ 20 ]  train_loss:0.003731  train_acc:100.00%  test_loss:1.102795  test_acc:72.35%  	Time 02:14
 Epoch [ 16/ 20 ]  train_loss:0.003174  train_acc:100.00%  test_loss:1.105457  test_acc:72.14%  	Time 02:10
 Epoch [ 17/ 20 ]  train_loss:0.002749  train_acc:100.00%  test_loss:1.118599  test_acc:71.98%  	Time 02:10
 Epoch [ 18/ 20 ]  train_loss:0.002433  train_acc:100.00%  test_loss:1.136706  test_acc:71.98%  	Time 02:05
 Epoch [ 19/ 20 ]  train_loss:0.002248  train_acc:100.00%  test_loss:1.134177  test_acc:72.08%  	Time 02:08
 Epoch [ 20/ 20 ]  train_loss:0.001928  train_acc:100.00%  test_loss:1.144748  test_acc:72.04%  	Time 02:14
```



### 不加正则项

```python
net = ResNet18()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01) # 不增加正则项
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=0.5,min_lr=1e-6)
train(net, train_data, test_data, 20, optimizer, criterion,scheduler)
```



```python
 Epoch [  1/ 20 ]  train_loss:1.627564  train_acc:39.68%  test_loss:1.410907  test_acc:47.35%  	Time 02:09
 Epoch [  2/ 20 ]  train_loss:1.244046  train_acc:54.82%  test_loss:1.253459  test_acc:53.93%  	Time 02:08
 Epoch [  3/ 20 ]  train_loss:1.028898  train_acc:63.24%  test_loss:1.108610  test_acc:60.21%  	Time 02:11
 Epoch [  4/ 20 ]  train_loss:0.863521  train_acc:69.29%  test_loss:1.056871  test_acc:62.63%  	Time 02:12
 Epoch [  5/ 20 ]  train_loss:0.721911  train_acc:74.87%  test_loss:1.054512  test_acc:63.37%  	Time 02:08
 Epoch [  6/ 20 ]  train_loss:0.579459  train_acc:80.15%  test_loss:1.066591  test_acc:63.54%  	Time 02:10
 Epoch [  7/ 20 ]  train_loss:0.441732  train_acc:85.41%  test_loss:0.999685  test_acc:66.69%  	Time 02:04
 Epoch [  8/ 20 ]  train_loss:0.306258  train_acc:90.63%  test_loss:1.143444  test_acc:64.26%  	Time 02:09
 Epoch [  9/ 20 ]  train_loss:0.177991  train_acc:95.18%  test_loss:0.985517  test_acc:70.31%  	Time 02:12
 Epoch [ 10/ 20 ]  train_loss:0.074963  train_acc:98.74%  test_loss:1.142383  test_acc:68.49%  	Time 02:08
 Epoch [ 11/ 20 ]  train_loss:0.030187  train_acc:99.70%  test_loss:1.056057  test_acc:71.48%  	Time 02:14
 Epoch [ 12/ 20 ]  train_loss:0.011685  train_acc:99.98%  test_loss:1.075722  test_acc:72.21%  	Time 02:08
 Epoch [ 13/ 20 ]  train_loss:0.006838  train_acc:100.00%  test_loss:1.099632  test_acc:72.13%  	Time 02:06
 Epoch [ 14/ 20 ]  train_loss:0.006079  train_acc:99.97%  test_loss:1.108780  test_acc:72.57%  	Time 02:08
 Epoch [ 15/ 20 ]  train_loss:0.004120  train_acc:100.00%  test_loss:1.126062  test_acc:72.10%  	Time 02:09
 Epoch [ 16/ 20 ]  train_loss:0.003369  train_acc:100.00%  test_loss:1.137595  test_acc:72.12%  	Time 02:10
 Epoch [ 17/ 20 ]  train_loss:0.002886  train_acc:100.00%  test_loss:1.147825  test_acc:72.19%  	Time 02:09
 Epoch [ 18/ 20 ]  train_loss:0.002590  train_acc:100.00%  test_loss:1.158138  test_acc:72.22%  	Time 02:14
 Epoch [ 19/ 20 ]  train_loss:0.002202  train_acc:100.00%  test_loss:1.171842  test_acc:72.01%  	Time 02:06
 Epoch [ 20/ 20 ]  train_loss:0.001993  train_acc:100.00%  test_loss:1.179933  test_acc:72.22%  	Time 02:09

```

结果我们可以发现，不加正则项的会对模型造成一些影响

其实在很多深度学习模型中，我们发现我们的`weight_decay`取5e-4是一个比较好的参数

