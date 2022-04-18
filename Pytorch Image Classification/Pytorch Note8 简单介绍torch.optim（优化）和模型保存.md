# Pytorch Note8 简单介绍torch.optim（优化）和模型保存
[TOC]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)
## nn.Module（模组）
在PyTorch里面编写神经网络，所有的层结构和损失函数都来自于torch.nn，所有的模型构建都是从这个基类nn.Module继承的，于是有了下面这个模板。

```python
import torch.nn as nn
 
class net_name(nn.Module):
    def __init__(self, other_arguments):
        super(net_name, self).__init__()
        self.convl = nn.Conv2d(in_channels, out_channels, kernel_size)
        # 其他网路层
        
    def forward(self, x):
        x = self.convl(x)
        return x
```
这样就建立了一个计算图，并且这个结构可以复用多次，每次调用就相当于用该计算图定义的相同参数做一次前向传播，这得益于PyTorch的自动求导功能，所以我们不需要自己编写反向传播，而所有的网络层都是由nn这个包得到的，比如线性层nn.Linear。

定义完模型之后，我们需要通过nn这个包来定义损失函数。常见的损失函数都已经定义在了nn中，比如均方误差、多分类的交叉熵，以及二分类的交叉熵等等，调用这些已经定义好的损失函数也很简单：

```python
criterion = nn.CrossEntropyLoss()

loss = criterion(output, target)
```

## torch.optim（优化）
在机器学习或者深度学习中，我们需要通过修改参数使得损失函数最小化（或最大化），优化算法就是一种调整模型参数更新的策略。优化算法分为两大类。

1. 一阶优化算法
这种算法使用各个参数的梯度值来更新参数，最常用的一阶优化算法是梯度下降。所谓的梯度就是导数的多变量表达式，函数的梯度形成了一个向量场，同时也是一个方向，这个方向上方向导数最大，且等于梯度。梯度下降的功能是通过寻找最小值，控制方差，更新模型参数，最终使模型收敛
>在我们的最优化学习中，有介绍梯度下降法，并且有详细的推导
2. 二阶优化算法
二阶优化算法使用了二阶导数（也叫做Hessian方法）来最小化或最大化损失函数，主要基于牛顿法，但是由于二阶导数的计算成本很高，所以这种方法并没有广泛使用。torch.optim是一个实现各种优化算法的包，大多数常见的算法都能够直接通过这个包来调用，比如随机梯度下降，以及添加动量的随机梯度下降，自适应学习率等。在调用的时候将需要优化的参数传入，这些参数都必须是Variable，然后传入一些基本的设定，比如学习率和动量等。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210611213750991.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)


举个例子
```python
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
```
这样我们就将模型的参数作为需要更新的参数传入优化器，设定学习率是0.01，动量是0.9随机梯度下降，在优化之前需要先将梯度归零，即optimizer.zeros()，然后通过loss.backward()反向传播，自动求导得到每个参数的梯度，最后只需要optimizer.step()就可以通过梯度作一步参数更新。

这里只是粗略简单介绍一下，后面会详细的介绍一下机器学习的算法

## 模型的保存和加载
在PyTorch里面使用`torch.save`来保存模型的结构和参数，有两种保存方式：
- 保存整个模型的结构信息和参数信息，保存的对象是模型` model`；
- 保存模型的参数，保存的对象是模型的状态`model.state_dict()`。

可以这样保存，save的第一个参数是保存对象，第二个参数是保存路径及名称：

```python
torch.save(model, './model.pth')
 
torch.save(model.state_dict(), './model_state.pth')
```

加载模型有两种方式对应于保存模型的方式：
- 加载完整的模型结构和参数信息，使用 `load_model=torch.load('model.pth')`，在网络较大的时候加载的时间比较长，同时存储空间也比较大；
- 加载模型参数信息，需要先导入模型的结构，然后通过 `model.load_state_dic(torch.load('model_state.pth'))`来导入。