# Pytorch Note57 Pytorch可视化网络结构

[toc]
全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)



> 随着深度神经网络做的的发展，网络的结构越来越复杂，我们也很难确定每一层的输入结构，输出结构以及参数等信息，这样导致我们很难在短时间内完成debug。因此掌握一个可以用来可视化网络结构的工具是十分有必要的。类似的功能在另一个深度学习库Keras中可以调用一个叫做model.summary()的API来很方便地实现，调用后就会显示我们的模型参数，输入大小，输出大小，模型的整体参数等，但是在PyTorch中没有这样一种便利的工具帮助我们可视化我们的模型结构。



对于pytorch来说，模型结构的可视化还是比较重要的，这样能够方便我们对数据的理解，并且也能加深对数据每一层的卷积变化的理解。今天这篇就简单介绍一下，一些模型的可视化，是我平常写代码常用的，也可以用来检测代码是否能够正确输出。

## 使用print打印

其实最简单的就是可以使用print打印，比如我们不懂其中一个网络的官方实现，我们可以从torchvision导入我们的模型

我简单使用torchvision中的alexnet模型进行测试

```python
from torchvision import models
net = models.alexnet()
print(net)
```

然后直接使用print打印，我们就可以直接看到内部的实现的参数，我们可以利用这个对我们的网络模型有个更好的理解，我们也可以利用这些模型进行迁移学习，只需要改变最后一层分类层即可。

> AlexNet(
> (features): Sequential(
>  (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
>  (1): ReLU(inplace=True)
>  (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
>  (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
>  (4): ReLU(inplace=True)
>  (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
>  (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
>  (7): ReLU(inplace=True)
>  (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
>  (9): ReLU(inplace=True)
>  (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
>  (11): ReLU(inplace=True)
>  (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
> )
> (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
> (classifier): Sequential(
>  (0): Dropout(p=0.5, inplace=False)
>  (1): Linear(in_features=9216, out_features=4096, bias=True)
>  (2): ReLU(inplace=True)
>  (3): Dropout(p=0.5, inplace=False)
>  (4): Linear(in_features=4096, out_features=4096, bias=True)
>  (5): ReLU(inplace=True)
>  (6): Linear(in_features=4096, out_features=1000, bias=True)
> )
> )

不过单纯的`print(model)`，只能得出基础构件的信息，既不能显示出每一层的shape，也不能显示对应参数量的大小，

## torchinfo可视化

实际上之前我用的很多都是torchsummary，但是后面好像发现，`torchsummary`和`torchsummaryX`已经许久没更新了，而`torchinfo`是由`torchsummary`和`torchsummaryX`重构出的库。

并且来说，torchsummary有时候会显得有些臃肿，输出所有层的维度和数量，对深层的网络结构就有些臃肿了。



### 安装torchinfo或者torchsummary

这个其实很简单，就是利用pip安装即可，打开命令行，输入以下命令就可以安装了

```python
pip install torchinfo torchsummary
```



### 使用torchinfo

无论是对于我们的torchinfo还是torchsummary来说，我们都是使用库里面的summary函数，不过这两个参数有些不同

大家也可以根据自己的喜好选取自己的喜好的summary



首先我们可以使用我们的torchinfo的summary函数

```python
from torchvision import models
net = models.alexnet()
from torchinfo import summary
summary(model, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/91e78b27a53e4129b77af75ff68ea251.png#pic_center)


`torchinfo`提供了更加详细的信息，包括

- 模块信息（每一层的类型、输出shape和参数量）
![在这里插入图片描述](https://img-blog.csdnimg.cn/3cabac91db134cbfa7fb1a07b1e7fe56.png)

- 模型整体的参数量以及大小
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/716d7a96285b438585ab8938dab9d48d.png)

- 一次前向或者反向传播需要的内存大小等
![在这里插入图片描述](https://img-blog.csdnimg.cn/c9ce4e50555c4f4585d04517dcd94ccf.png)




我们还可以看以前的summary函数，对于这一部分来说，就是Layer的可视化不同，这一部分可视化也给出了众多的参数，但是对于复杂模型的结果，就会不清晰

```python
from torchsummary import summary
summary(net, (3, 224, 224)) # 3:图片的通道数 224: 图片的高宽
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20fce0affeba46f382aa8cf196f7a267.png#pic_center)


> **注意**：
> 当使用的是colab或者jupyter notebook时，想要实现该方法，`summary()`一定是该单元（即notebook中的cell）的返回值，否则我们就需要使用`print(summary(...))`来可视化。
