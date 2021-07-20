# Pytorch Note1 Pytorch介绍

[toc]
全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

## PyTorch的诞生
2017年1月，Facebook人工智能研究院（FAIR）团队在GitHub上开源了PyTorch，并迅速占领GitHub热度榜榜首。

作为一个2017年才发布，具有先进设计理念的框架，PyTorch的历史可追溯到2002年就诞生于纽约大学的Torch。Torch使用了一种不是很大众的语言Lua作为接口。Lua简洁高效，但由于其过于小众，用的人不是很多，以至于很多人听说要掌握Torch必须新学一门语言就望而却步（其实Lua是一门比Python还简单的语言）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210611104315326.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
考虑到Python在计算科学领域的领先地位，以及其生态完整性和接口易用性，几乎任何框架都不可避免地要提供Python接口。终于，在2017年，Torch的幕后团队推出了PyTorch。PyTorch不是简单地封装Lua Torch提供Python接口，而是对Tensor之上的所有模块进行了重构，并新增了最先进的自动求导系统，成为当下最流行的动态图框架。

PyTorch一经推出就立刻引起了广泛关注，并迅速在研究领域流行起来。图1所示为Google指数，PyTorch自发布起关注度就在不断上升，截至2017年10月18日，PyTorch的热度已然超越了其他三个框架（Caffe、MXNet和Theano），并且其热度还在持续上升中。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210611104323590.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 常见的深度学习框架简介
随着深度学习的发展，深度学习框架如雨后春笋般诞生于高校和公司中。尤其是近两年，Google、Facebook、Microsoft等巨头都围绕深度学习重点投资了一系列新兴项目，他们也一直在支持一些开源的深度学习框架。

目前研究人员正在使用的深度学习框架不尽相同，有 TensorFlow 、Caffe、Theano、Keras等，常见的深度学习框架如图2所示。这些深度学习框架被应用于计算机视觉、语音识别、自然语言处理与生物信息学等领域，并获取了极好的效果。本部分主要介绍当前深度学习领域影响力比较大的几个框架

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210611104356317.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 为什么选择PyTorch

这么多深度学习框架，为什么选择PyTorch呢？

因为PyTorch是当前难得的简洁优雅且高效快速的框架。在笔者眼里，PyTorch达到目前深度学习框架的最高水平。当前开源的框架中，没有哪一个框架能够在灵活性、易用性、速度这三个方面有两个能同时超过PyTorch。下面是许多研究人员选择PyTorch的原因。

因为PyTorch是当前难得的简洁优雅且高效快速的框架。在笔者眼里，PyTorch达到目前深度学习框架的最高水平。当前开源的框架中，没有哪一个框架能够在灵活性、易用性、速度这三个方面有两个能同时超过PyTorch。下面是许多研究人员选择PyTorch的原因。


- ① 简洁：PyTorch的设计追求最少的封装，尽量避免重复造轮子。不像TensorFlow中充斥着session、graph、operation、name_scope、variable、tensor、layer等全新的概念，PyTorch的设计遵循tensor→variable(autograd)→nn.Module 三个由低到高的抽象层次，分别代表高维数组（张量）、自动求导（变量）和神经网络（层/模块），而且这三个抽象之间联系紧密，可以同时进行修改和操作。简洁的设计带来的另外一个好处就是代码易于理解。PyTorch的源码只有TensorFlow的十分之一左右，更少的抽象、更直观的设计使得PyTorch的源码十分易于阅读。在笔者眼里，PyTorch的源码甚至比许多框架的文档更容易理解。

- ② 速度：PyTorch的灵活性不以速度为代价，在许多评测中，PyTorch的速度表现胜过TensorFlow和Keras等框架 。框架的运行速度和程序员的编码水平有极大关系，但同样的算法，使用PyTorch实现的那个更有可能快过用其他框架实现的。

- ③易用：PyTorch是所有的框架中面向对象设计的最优雅的一个。PyTorch的面向对象的接口设计来源于Torch，而Torch的接口设计以灵活易用而著称，Keras作者最初就是受Torch的启发才开发了Keras。PyTorch继承了Torch的衣钵，尤其是API的设计和模块的接口都与Torch高度一致。PyTorch的设计最符合人们的思维，它让用户尽可能地专注于实现自己的想法，即所思即所得，不需要考虑太多关于框架本身的束缚。

- ④活跃的社区：PyTorch提供了完整的文档，循序渐进的指南，作者亲自维护的论坛 供用户交流和求教问题。Facebook 人工智能研究院对PyTorch提供了强力支持，作为当今排名前三的深度学习研究机构，FAIR的支持足以确保PyTorch获得持续的开发更新，不至于像许多由个人开发的框架那样昙花一现。

在PyTorch推出不到一年的时间内，各类深度学习问题都有利用PyTorch实现的解决方案在GitHub上开源。同时也有许多新发表的论文采用PyTorch作为论文实现的工具，PyTorch正在受到越来越多人的追捧 。

如果说 TensorFlow的设计是 **“Make It Complicated”** ，Keras的设计是 **“Make It Complicated And Hide It”** ，那么PyTorch的设计真正做到了 **“Keep it Simple，Stupid”** 。简洁即是美。

使用TensorFlow能找到很多别人的代码，使用PyTorch能轻松实现自己的想法。

PyTorch 的前身是 Torch，其是一个十分老牌、对多维矩阵数据进行操作的张量（tensor ）库，在机器学习和其他数学密集型应用有广泛应用，但由于其语言采用 Lua，导致在国内一直很小众，如今使用 Python 语言强势归来，快速的赢得了大量使用者。

PyTorch 提供了两种高层面的功能：
- 使用强大的 GPU 加速的 Tensor 计算（类似 numpy）
- 构建于基于 autograd 系统的深度神经网络

所以使用 PyTorch 的原因通常有两个：
- 作为 numpy 的替代，以便使用强大的 GPU 加速；
- 将其作为一个能提供最大灵活性和速度的深度学习研究平台
## 总结一下Pytorch的特点
### Python 优先
PyTorch 不是简单地在整体 C++ 框架上绑定 Python，他深入构建在 Python 之上，你可以像使用 numpy/scipy/scikit-learn 那样轻松地使用 PyTorch，也可以用你喜欢的库和包在 PyTorch 中编写新的神经网络层，尽量让你不用重新发明轮子。

### 命令式体验
PyTorch 的设计思路是线性、直观且易于使用。当你需要执行一行代码时，它会忠实执行。PyTorch 没有异步的世界观。当你打开调试器，或接收到错误代码和 stack trace 时，你会发现理解这些信息是非常轻松的。Stack-trace 点将会直接指向代码定义的确切位置。我们不希望你在 debug 时会因为错误的指向或异步和不透明的引擎而浪费时间。

### 快速精益
PyTorch 具有轻巧的框架，集成了各种加速库，如 Intel MKL、英伟达的 CuDNN 和 NCCL 来优化速度。在其核心，它的 CPU 和 GPU Tensor 与神经网络后端（TH、THC、THNN、THCUNN）被编写成了独立的库，带有 C99 API。

## 安装
PyTorch 的安装非常方便，可以使用 Anaconda 进行安装，也可以使用 pip 进行安装，比如

使用 conda 进行安装   
`conda install pytorch torchvision -c pytorch`

或者使用 pip   
`pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl 
pip install torchvision`

目前只支持 Mac OSX 和 Linux 系统，Windows 系统在不久之后也会支持，更多详细信息可以访问[官网](http://pytorch.org/)

下一章传送门：[Note2 Pytorch环境配置](https://blog.csdn.net/weixin_45508265/article/details/117809016)