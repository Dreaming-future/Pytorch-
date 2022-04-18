# Pytorch Note

什么是快乐星球，让我用简单易懂的代码带你进入pytorch快乐星球

![](https://img-blog.csdnimg.cn/img_convert/ac4fe519487d89d342cb05d2f710a20c.png#pic_center)

这是我的Pytoch学习笔记，下面会慢慢的更新我的学习笔记

---

## part1: 深度学习基础

- PyTorch介绍和环境配置
  - [Note1 Pytorch介绍](https://blog.csdn.net/weixin_45508265/article/details/117808642)
  - [Note2 Pytorch环境配置](https://blog.csdn.net/weixin_45508265/article/details/117809016)
- Pytorch基础
  * [Note3 Tensor(张量)](https://blog.csdn.net/weixin_45508265/article/details/117811600)
  * [Note4 Variable（变量）](https://blog.csdn.net/weixin_45508265/article/details/117812880)
  * [Note5 动态图和静态图 dynamic-graph](https://blog.csdn.net/weixin_45508265/article/details/117816228)
  * [Note6 自动求导Autograd](https://blog.csdn.net/weixin_45508265/article/details/117816977)
  * [Note7 Dataset（数据集）](https://blog.csdn.net/weixin_45508265/article/details/117818268)
  * [Note8 简单介绍torch.optim(优化)和模型保存](https://blog.csdn.net/weixin_45508265/article/details/117819532)
- 神经网络
  - [Note9 线性模型和梯度下降](https://blog.csdn.net/weixin_45508265/article/details/117827063)
  - [Note10 多项式回归](https://blog.csdn.net/weixin_45508265/article/details/117827333)
  - [Note11 Logistic 回归模型](https://blog.csdn.net/weixin_45508265/article/details/117828669)
  - [Note12 多层神经网络](https://blog.csdn.net/weixin_45508265/article/details/117848000)
  - [Note13 反向传播算法](https://blog.csdn.net/weixin_45508265/article/details/117855631)
  - [Note14 激活函数(Activation Function)](https://blog.csdn.net/weixin_45508265/article/details/117856338)
  - 优化算法
    - [Note15 优化算法1 梯度下降（Gradient descent varients）](https://blog.csdn.net/weixin_45508265/article/details/117859824)
    - [Note16 优化算法2 动量法(Momentum)](https://blog.csdn.net/weixin_45508265/article/details/117874046)
    - [Note17 优化算法3 Adagrad算法](https://blog.csdn.net/weixin_45508265/article/details/117877596)
    - [Note18 优化算法4 RMSprop算法](https://blog.csdn.net/weixin_45508265/article/details/117885569)
    - [Note19 优化算法5 Adadelta算法](https://blog.csdn.net/weixin_45508265/article/details/118930950)
    - [Note20 优化算法6 Adam算法](https://blog.csdn.net/weixin_45508265/article/details/118931366)
    - [Note21 优化算法对比](https://blog.csdn.net/weixin_45508265/article/details/118931198)
  - 数据处理和过拟合的方法
    - [Note22 数据预处理](https://blog.csdn.net/weixin_45508265/article/details/118933624)
    - [Note23 参数初始化](https://blog.csdn.net/weixin_45508265/article/details/118945764)
    - [Note24 防止过拟合](https://blog.csdn.net/weixin_45508265/article/details/118946214)
  - [Note25 深层神经网络实现 MNIST 手写数字分类](https://blog.csdn.net/weixin_45508265/article/details/118960084)
- 卷积神经网络
  - [Note26 卷积神经网络](https://blog.csdn.net/weixin_45508265/article/details/118971022)
  - [Note27 卷积设计的一些经验总结](https://blog.csdn.net/weixin_45508265/article/details/118971229)
  - [Note28 Pytorch的卷积模块](https://blog.csdn.net/weixin_45508265/article/details/118972098)
  - [Note29 使用重复元素的深度网络 VGG](https://blog.csdn.net/weixin_45508265/article/details/118974104)
  - [Note30 更加丰富化结构的网络 GoogLeNet](https://blog.csdn.net/weixin_45508265/article/details/119040170)
  - [Note31 深度残差网络 ResNet](https://blog.csdn.net/weixin_45508265/article/details/119087199)
  - [Note32 稠密连接的卷积网络 DenseNet](https://blog.csdn.net/weixin_45508265/article/details/119184861)
  - 更好的训练卷积网络
    - [Note33 数据增强](https://blog.csdn.net/weixin_45508265/article/details/119047348)
    - [Note34 学习率衰减](https://blog.csdn.net/weixin_45508265/article/details/119089705)
    - [Note35 正则化](https://blog.csdn.net/weixin_45508265/article/details/119123529)
- 循环神经网络
  - [Note36 循环神经网络的变式：LSTM和GRU](https://blog.csdn.net/weixin_45508265/article/details/119191003)
  - [Note37 PyTorch 中的循环神经网络模块](https://blog.csdn.net/weixin_45508265/article/details/119194403)
  - [Note38 RNN 做图像分类](https://blog.csdn.net/weixin_45508265/article/details/119210533)
  - [Note39 RNN 序列预测](https://blog.csdn.net/weixin_45508265/article/details/119347418)
  - 自然语言处理的应用
    - [Note40 词嵌入（word embedding）](https://blog.csdn.net/weixin_45508265/article/details/119362381)
    - [Note41 N-Gram 模型](https://blog.csdn.net/weixin_45508265/article/details/119361565)
    - [Note42 LSTM 做词性预测](https://blog.csdn.net/weixin_45508265/article/details/119428061)
- 生成对抗网络
  - [Note43 自动编码器(Autoencoder)](https://blog.csdn.net/weixin_45508265/article/details/119582615)
  - [Note44 变分自动编码器（VAE）](https://blog.csdn.net/weixin_45508265/article/details/119594085)
  - [Note45 生成对抗网络（GAN）](https://blog.csdn.net/weixin_45508265/article/details/119684311)
  - [Note46 生成对抗网络的数学原理](https://blog.csdn.net/weixin_45508265/article/details/119811489)
  - [Note47 Imporving GAN](https://blog.csdn.net/weixin_45508265/article/details/119830209)
  - [Note48 DCGAN生成人脸](https://blog.csdn.net/weixin_45508265/article/details/119830209)
- 深度强化学习
  - [Note49 Q-learning](https://blog.csdn.net/weixin_45508265/article/details/119937575)
  - [Note50 Gym 介绍](https://redamancy.blog.csdn.net/article/details/120072032)
  - [Note51 Deep Q Networks](https://redamancy.blog.csdn.net/article/details/120082922)

- Pytorch 高级应用
  - [Note52 灵活的数据读取介绍](https://redamancy.blog.csdn.net/article/details/120151183)
  - [Note53 TensorBoard可视化](https://redamancy.blog.csdn.net/article/details/120156777)

## part2: 深度学习的应用

- 迁移学习
  - [Note54 迁移学习简介](https://redamancy.blog.csdn.net/article/details/120159251)
  - [Note55 迁移学习实战猫狗分类](https://redamancy.blog.csdn.net/article/details/120173576)

- 计算机视觉
  - 
- 自然语言处理
  - 


---

参考:

- 《深度学习入门之pytorch》
- [https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch)

