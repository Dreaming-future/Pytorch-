# Pytorch Note6 自动求导Autograd

[TOC]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)
## 自动求导
用Tensor训练网络很方便，但有时候反向传播过程需要手动实现。这对于像线性回归等较为简单的模型来说，还可以应付，但实际使用中经常出现非常复杂的网络结构，此时如果手动实现反向传播，不仅费时费力，而且容易出错，难以检查。torch.autograd就是为方便用户使用，而专门开发的一套自动求导引擎，它能够根据输入和前向传播过程自动构建计算图，并执行反向传播。

计算图(Computation Graph)是现代深度学习框架如PyTorch和TensorFlow等的核心，其为高效自动求导算法——反向传播(Back Propogation)提供了理论支持，了解计算图在实际写程序过程中会有极大的帮助。

```python
import torch
from torch.autograd import Variable
```

## 简单情况的自动求导
下面我们显示一些简单情况的自动求导，"简单"体现在计算的结果都是标量，也就是一个数，我们对这个标量进行自动求导。

```python
x = Variable(torch.Tensor([2]), requires_grad=True)
y = x + 2
z = y ** 2 + 3
print(z)
```
>tensor([19.], grad_fn=<AddBackward0>)

通过上面的一些列操作，我们从 x 得到了最后的结果out，我们可以将其表示为数学公式

$$
z = (x + 2)^2 + 3
$$

那么我们从 z 对 x 求导的结果就是 

$$
\frac{\partial z}{\partial x} = 2 (x + 2) = 2 (2 + 2) = 8
$$
如果你对求导不熟悉，可以查看以下[网址进行复习](https://baike.baidu.com/item/%E5%AF%BC%E6%95%B0#1)

```python
# 使用自动求导
z.backward()
print(x.grad)
```
>tensor([8.])
>
>对于上面这样一个简单的例子，我们验证了自动求导，同时可以发现发现使用自动求导非常方便。如果是一个更加复杂的例子，那么手动求导就会显得非常的麻烦，所以自动求导的机制能够帮助我们省去麻烦的数学计算，下面我们可以看一个更加复杂的例子。


```python
x = Variable(torch.randn(10, 20), requires_grad=True)
y = Variable(torch.randn(10, 5), requires_grad=True)
w = Variable(torch.randn(20, 5), requires_grad=True)

out = torch.mean(y - torch.matmul(x, w)) # torch.matmul 是做矩阵乘法
out.backward()
```
如果你对矩阵乘法不熟悉，可以查看下面的[网址进行复习](https://baike.baidu.com/item/%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95/5446029?fr=aladdin)

```python
# 得到 x 的梯度
print(x.grad)
# 得到 y 的的梯度
print(y.grad)
# 得到 w 的梯度
print(w.grad)
```
上面数学公式就更加复杂，矩阵乘法之后对两个矩阵对应元素相乘，然后所有元素求平均，有兴趣的同学可以手动去计算一下梯度，使用 PyTorch 的自动求导，我们能够非常容易得到 x, y 和 w 的导数，因为深度学习中充满大量的矩阵运算，所以我们没有办法手动去求这些导数，有了自动求导能够非常方便地解决网络更新的问题。


## 复杂情况的自动求导
上面我们展示了简单情况下的自动求导，都是对标量进行自动求导，可能你会有一个疑问，如何对一个向量或者矩阵自动求导了呢？感兴趣的同学可以自己先去尝试一下，下面我们会介绍对多维数组的自动求导机制。

```python
m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True) # 构建一个 1 x 2 的矩阵
n = Variable(torch.zeros(1, 2)) # 构建一个相同大小的 0 矩阵
print(m)
print(n)
```
>Variable containing:
> 2  3
>[torch.FloatTensor of size 1x2]<br>
>Variable containing:
> 0  0
>[torch.FloatTensor of size 1x2]


```python
# 通过 m 中的值计算新的 n 中的值
n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3
print(n)
```
>Variable containing:
>4  27
>[torch.FloatTensor of size 1x2]

将上面的式子写成数学公式，可以得到 
$$
n = (n_0,\ n_1) = (m_0^2,\ m_1^3) = (2^2,\ 3^3) 
$$
下面我们直接对 n 进行反向传播，也就是求 n 对 m 的导数。

这时我们需要明确这个导数的定义，即如何定义

$$
\frac{\partial n}{\partial m} = \frac{\partial (n_0,\ n_1)}{\partial (m_0,\ m_1)}
$$
在 PyTorch 中，如果要调用自动求导，需要往`backward()`中传入一个参数，这个参数的形状和 n 一样大，比如是 $(w_0,\ w_1)$，那么自动求导的结果就是：
$$
\frac{\partial n}{\partial m_0} = w_0 \frac{\partial n_0}{\partial m_0} + w_1 \frac{\partial n_1}{\partial m_0}
$$
$$
\frac{\partial n}{\partial m_1} = w_0 \frac{\partial n_0}{\partial m_1} + w_1 \frac{\partial n_1}{\partial m_1}
$$

```python
n.backward(torch.ones_like(n)) # 将 (w0, w1) 取成 (1, 1)
```

```python
print(m.grad)
```
>Variable containing:
>4  27
>[torch.FloatTensor of size 1x2]

通过自动求导我们得到了梯度是 4 和 27，我们可以验算一下
$$
\frac{\partial n}{\partial m_0} = w_0 \frac{\partial n_0}{\partial m_0} + w_1 \frac{\partial n_1}{\partial m_0} = 2 m_0 + 0 = 2 \times 2 = 4
$$
$$
\frac{\partial n}{\partial m_1} = w_0 \frac{\partial n_0}{\partial m_1} + w_1 \frac{\partial n_1}{\partial m_1} = 0 + 3 m_1^2 = 3 \times 3^2 = 27
$$
通过验算我们可以得到相同的结果

## 多次自动求导
通过调用 backward 我们可以进行一次自动求导，如果我们再调用一次 backward，会发现程序报错，没有办法再做一次。这是因为 PyTorch 默认做完一次自动求导之后，计算图就被丢弃了，所以两次自动求导需要手动设置一个东西，我们通过下面的小例子来说明。

```python
x = Variable(torch.FloatTensor([3]), requires_grad=True)
y = x * 2 + x ** 2 + 3
print(y)
```
>Variable containing:
> 18
>[torch.FloatTensor of size 1]

```python
y.backward(retain_graph=True) # 设置 retain_graph 为 True 来保留计算图
```

```python
print(x.grad)
```
>Variable containing:
> 8
>[torch.FloatTensor of size 1]
```python
y.backward() # 再做一次自动求导，这次不保留计算图
```

```python
print(x.grad)
```
>Variable containing:
> 16
>[torch.FloatTensor of size 1]


可以发现 x 的梯度变成了 16，因为这里做了两次自动求导，所以讲第一次的梯度 8 和第二次的梯度 8 加起来得到了 16 的结果。


## 小练习

定义

$$
x = 
\left[
\begin{matrix}
x_0 \\
x_1
\end{matrix}
\right] = 
\left[
\begin{matrix}
2 \\
3
\end{matrix}
\right]
$$

$$
k = (k_0,\ k_1) = (x_0^2 + 3 x_1,\ 2 x_0 + x_1^2)
$$

我们希望求得

$$
j = \left[
\begin{matrix}
\frac{\partial k_0}{\partial x_0} & \frac{\partial k_0}{\partial x_1} \\
\frac{\partial k_1}{\partial x_0} & \frac{\partial k_1}{\partial x_1}
\end{matrix}
\right]
$$

参考答案：

$$
\left[
\begin{matrix}
4 & 3 \\
2 & 6 \\
\end{matrix}
\right]
$$

```python
x = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
k = Variable(torch.zeros(2))

k[0] = x[0] ** 2 + 3 * x[1]
k[1] = x[1] ** 2 + 2 * x[0]
```

```python
j = torch.zeros(2, 2)

k.backward(torch.FloatTensor([1, 0]), retain_graph=True)
j[0] = x.grad.data

x.grad.data.zero_() # 归零之前求得的梯度

k.backward(torch.FloatTensor([0, 1]))
j[1] = x.grad.data
```

```python
print(j)
```
> 4  3
>  2  6
> [torch.FloatTensor of size 2x2]

下一章传送门：[Note7 Dataset（数据集）](https://blog.csdn.net/weixin_45508265/article/details/117818268)