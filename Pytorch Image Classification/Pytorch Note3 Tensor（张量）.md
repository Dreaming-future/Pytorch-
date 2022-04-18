# Pytorch Note3 Tensor（张量）

[TOC]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)
## Tensor
Tensor，又名张量，读者可能对这个名词似曾相识，因它不仅在PyTorch中出现过，它也是Theano、TensorFlow、
Torch和MxNet中重要的数据结构。关于张量的本质不乏深度的剖析，但从工程角度来讲，可简单地认为它就是一个数组，且支持高效的科学计算。它可以是一个数（标量）、一维数组（向量）、二维数组（矩阵）和更高维的数组（高阶数据）。Tensor和Numpy的ndarrays类似，但PyTorch的tensor支持GPU加速。

常用的不同数据类型的Tensor：
- 32位浮点型torch.FloatTensor
- 64位浮点型 torch.DoubleTensor
- 16位整型torch.ShortTensor
- 32位整型torch.IntTensor
- 64位整型torch.LongTensor
## 把 PyTorch 当做 NumPy 用
PyTorch 的官方介绍是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构件是张量，所以我们可以把 PyTorch 当做 NumPy 来用，PyTorch 的很多操作好 NumPy 都是类似的，但是因为其能够在 GPU 上运行，所以有着比 NumPy 快很多倍的速度。

##  基础操作

学习过Numpy的读者会对本节内容感到非常熟悉，因tensor的接口有意设计成与Numpy类似，以方便用户使用。但不熟悉Numpy也没关系，本节内容并不要求先掌握Numpy。

从接口的角度来讲，对tensor的操作可分为两类：

1. `torch.function`，如`torch.save`等。
2. 另一类是`tensor.function`，如`tensor.view`等。

为方便使用，对tensor的大部分操作同时支持这两类接口，在本书中不做具体区分，如`torch.sum (torch.sum(a, b))`与`tensor.sum (a.sum(b))`功能等价。

而从存储的角度来讲，对tensor的操作又可分为两类：

1. 不会修改自身的数据，如 `a.add(b)`， 加法的结果会返回一个新的tensor。
2. 会修改自身的数据，如 `a.add_(b)`， 加法的结果仍存储在a中，a被修改了。

函数名以`_`结尾的都是inplace方式, 即会修改调用者自己的数据，在实际应用中需加以区分。

## 创建Tensor

在PyTorch中新建tensor的方法有很多，具体如表3-1所示。

表: 常见新建tensor的方法

|               函数                |           功能            |
| :-------------------------------: | :-----------------------: |
|          Tensor(\*sizes)          |       基础构造函数        |
|           tensor(data,)           |  类似np.array的构造函数   |
|           ones(\*sizes)           |         全1Tensor         |
|          zeros(\*sizes)           |         全0Tensor         |
|           eye(\*sizes)            |    对角线为1，其他为0     |
|          arange(s,e,step          |    从s到e，步长为step     |
|        linspace(s,e,steps)        | 从s到e，均匀切分成steps份 |
|        rand/randn(\*sizes)        |       均匀/标准分布       |
| normal(mean,std)/uniform(from,to) |     正态分布/均匀分布     |
|            randperm(m)            |         随机排列          |

这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu).


其中使用`Tensor`函数新建tensor是最复杂多变的方式，它既可以接收一个list，并根据list的数据新建tensor，也能根据指定的形状新建tensor，还能传入其他的tensor。

```python
import torch
import numpy as np
```

```python
# 创建一个 numpy ndarray
numpy_tensor = np.random.randn(10, 20)
```
## numpy --> tensor
我们可以使用下面两种方式将numpy的ndarray转换到tensor上

```python
pytorch_tensor1 = torch.Tensor(numpy_tensor)
pytorch_tensor2 = torch.from_numpy(numpy_tensor)
```
使用以上两种方法进行转换的时候，会直接将 NumPy ndarray 的数据类型转换为对应的 PyTorch Tensor 数据类型
## tensor --> numpy

同时我们也可以使用下面的方法将 pytorch tensor 转换为 numpy ndarray

```python
# 如果 pytorch tensor 在 cpu 上
numpy_array = pytorch_tensor1.numpy()

# 如果 pytorch tensor 在 gpu 上
numpy_array = pytorch_tensor1.cpu().numpy()
```
需要注意 GPU 上的 Tensor 不能直接转换为 NumPy ndarray，需要使用`.cpu()`先将 GPU 上的 Tensor 转到 CPU 上


PyTorch Tensor 使用 GPU 加速

我们可以使用以下两种方式将 Tensor 放到 GPU 上

```python
# 第一种方式是定义 cuda 数据类型
dtype = torch.cuda.FloatTensor # 定义默认 GPU 的 数据类型
gpu_tensor = torch.randn(10, 20).type(dtype)

# 第二种方式更简单，推荐使用
gpu_tensor = torch.randn(10, 20).cuda(0) # 将 tensor 放到第一个 GPU 上
gpu_tensor = torch.randn(10, 20).cuda(0) # 将 tensor 放到第二个 GPU 上
```
使用第一种方式将 tensor 放到 GPU 上的时候会将数据类型转换成定义的类型，而是用第二种方式能够直接将 tensor 放到 GPU 上，类型跟之前保持一致

推荐在定义 tensor 的时候就明确数据类型，然后直接使用第二种方法将 tensor 放到 GPU 上
而将 tensor 放回 CPU 的操作非常简单

```python
cpu_tensor = gpu_tensor.cpu()
```
## tensor 基本属性
我们也能够访问到 Tensor 的一些属性

```python
# 可以通过下面两种方式得到 tensor 的大小
print(pytorch_tensor1.shape)
print(pytorch_tensor1.size())
```
>torch.Size([10, 20])
>torch.Size([10, 20])

```python
# 得到 tensor 的数据类型
print(pytorch_tensor1.type())
```

```python
# 得到 tensor 的维度
print(pytorch_tensor1.dim())
```
>2

```python
# 得到 tensor 的所有元素个数
print(pytorch_tensor1.numel())
```
>200

## 小练习1

查阅以下[文档](http://pytorch.org/docs/0.3.0/tensors.html)了解 tensor 的数据类型，创建一个 float64、大小是 3 x 2、随机初始化的 tensor，将其转化为 numpy 的 ndarray，输出其数据类型

参考输出: float64

```python
# 答案
x = torch.randn(3, 2)
x = x.type(torch.DoubleTensor)
x_array = x.numpy()
print(x_array.dtype)
```
>float64
## Tensor的操作
Tensor 操作中的 api 和 NumPy 非常相似，如果你熟悉 NumPy 中的操作，那么 tensor 基本是一致的，下面我们来列举其中的一些操作

```python
x = torch.ones(2, 2)
print(x) # 这是一个float tensor
```

```python
x = torch.randn(4, 3)
print(x)
```
>tensor([[-1.0270,  0.8912,  0.4995],
>  [-0.1699,  1.1554, -1.6936],
>  [-0.7028, -0.5200,  1.2046],
>  [-1.2708,  1.2331,  0.5092]])

```python
# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)
```

```python
# 增加维度或者减少维度
print(x.shape)
x = x.unsqueeze(0) # 在第一维增加
print(x.shape)
```
>torch.Size([4, 3])
>torch.Size([1, 4, 3])
```python
x = x.unsqueeze(1) # 在第二维增加
print(x.shape)
```
>torch.Size([1, 1, 4, 3])

```python
x = x.squeeze(0) # 减少第一维
print(x.shape)
```
>torch.Size([1, 4, 3])

```python
x = x.squeeze() # 将 tensor 中所有的一维全部都去掉
print(x.shape)
```
>torch.Size([4, 3])
```python
x = torch.randn(3, 4, 5)
print(x.shape)

# 使用permute和transpose进行维度交换
x = x.permute(1, 0, 2) # permute 可以重新排列 tensor 的维度
print(x.shape)

x = x.transpose(0, 2)  # transpose 交换 tensor 中的两个维度
print(x.shape)
```
>torch.Size([3, 4, 5])
>torch.Size([4, 3, 5])
>torch.Size([5, 3, 4])

```python
# 使用 view 对 tensor 进行 reshape
x = torch.randn(3, 4, 5)
print(x.shape)

x = x.view(-1, 5) # -1 表示任意的大小，5 表示第二维变成 5
print(x.shape)

x = x.view(3, 20) # 重新 reshape 成 (3, 20) 的大小
print(x.shape)
```
>torch.Size([3, 4, 5])
>torch.Size([12, 5])
>torch.Size([3, 20])

## 小练习2

访问[文档](http://pytorch.org/docs/0.3.0/tensors.html)了解 tensor 更多的 api，实现下面的要求

创建一个 float32、4 x 4 的全为1的矩阵，将矩阵正中间 2 x 2 的矩阵，全部修改成2

参考输出
$$
\left[
\begin{matrix}
1 & 1 & 1 & 1 \\
1 & 2 & 2 & 1 \\
1 & 2 & 2 & 1 \\
1 & 1 & 1 & 1
\end{matrix}
\right] \\
[torch.FloatTensor\ of\ size\ 4x4]
$$

```python
# 答案
x = torch.ones(4, 4).float()
x[1:3, 1:3] = 2
print(x)
```
>tensor([[1., 1., 1., 1.],
>  [1., 2., 2., 1.],
>  [1., 2., 2., 1.],
>  [1., 1., 1., 1.]])



## 几种形状的Tensor

* 0: scalar
* 1: vector
* 2: matrix
* 3: n-dimensional tensor

![Scalar, vector, matrix, tensor - a drawing by Tai-Danae Bradley](https://uploads-ssl.webflow.com/5b1d427ae0c922e912eda447/5cd99a73f8ce4494ad86852e_arraychart.jpg)

