# Pytorch Note29 使用重复元素的深度网络 VGG

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

计算机视觉是一直深度学习的主战场，从这里我们将接触到近几年非常流行的卷积网络结构，网络结构由浅变深，参数越来越多，网络有着更多的跨层链接，首先我们先介绍一个数据集 cifar10，我们将以此数据集为例介绍各种卷积网络的结构。

## CIFAR 10
cifar 10 这个数据集一共有 50000 张训练集，10000 张测试集，两个数据集里面的图片都是 png 彩色图片，图片大小是 32 x 32 x 3，一共是 10 分类问题，分别为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。这个数据集是对网络性能测试一个非常重要的指标，可以说如果一个网络在这个数据集上超过另外一个网络，那么这个网络性能上一定要比另外一个网络好，目前这个数据集最好的结果是 95% 左右的测试集准确率。

![](https://tva1.sinaimg.cn/large/006tNc79ly1fmpjxxq7wcj30db0ae7ag.jpg)

你能用肉眼对这些图片进行分类吗？

cifar 10 已经被 pytorch 内置了，使用非常方便，只需要调用 `torchvision.datasets.CIFAR10` 就可以了

## VGGNet

vggNet 是第一个真正意义上的深层网络结构，其是 ImageNet2014年的冠军，得益于 python 的函数和循环，我们能够非常方便地构建重复结构的深层网络。总结起来就是它使用了更小的滤波器，同时使用了更深的结构

vgg 的网络结构非常简单，就是不断地堆叠卷积层和池化层，下面是一个简单的图示

![](https://tva1.sinaimg.cn/large/006tNc79ly1fmpk5smtidj307n0dx3yv.jpg)

vgg 几乎全部使用 3 x 3 的卷积核以及 2 x 2 的池化层，使用小的卷积核进行多层的堆叠和一个大的卷积核的感受野是相同的，同时小的卷积核还能减少参数，同时可以有更深的结构。

vgg 的一个关键就是使用很多层 3 x 3 的卷积然后再使用一个最大池化层，这个模块被使用了很多次，下面我们照着这个结构来写一写

```python
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```



```python
class VGG(nn.Module):
    
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512,10),
        )
#         self.classifier = nn.Linear(512,10)

        self._initialize_weight()
        
    def forward(self, x):
        out = self.features(x)
        # 在进入
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    # make layers
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3 # RGB 初始通道为3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # kernel_size 为 2 x 2,然后步长为2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), # 都是(3.3)的卷积核
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]  # RelU
                in_channels = x  # 重定义通道
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    # 初始化参数
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
```

其实可以看出，VGG只是对网络层的不断堆叠，并没有进行太多创新，而增加深度确实可以在一定程度改善模型的效果

> ```
> [  1/200, 82 seconds]|	 loss: 1.992001, accuaracy: 18.77%	|	val_loss: 1.957387, val_accuaracy: 19.77%	train_lr:0.100000
> test_acc: 20.31% ,  test_loss : 1.954457
> [  2/200, 80 seconds]|	 loss: 1.687947, accuaracy: 36.07%	|	val_loss: 1.672445, val_accuaracy: 36.62%	train_lr:0.100000
> test_acc: 36.50% ,  test_loss : 1.663611
> [  3/200, 79 seconds]|	 loss: 1.382243, accuaracy: 48.09%	|	val_loss: 1.399725, val_accuaracy: 48.74%	train_lr:0.100000
> test_acc: 48.53% ,  test_loss : 1.395514
> [  4/200, 82 seconds]|	 loss: 1.120093, accuaracy: 60.88%	|	val_loss: 1.078191, val_accuaracy: 62.43%	train_lr:0.100000
> test_acc: 62.55% ,  test_loss : 1.091310
> [  5/200, 81 seconds]|	 loss: 1.302385, accuaracy: 58.28%	|	val_loss: 1.335634, val_accuaracy: 58.30%	train_lr:0.100000
> test_acc: 57.02% ,  test_loss : 1.358443
> [  6/200, 79 seconds]|	 loss: 1.007863, accuaracy: 66.49%	|	val_loss: 1.037014, val_accuaracy: 65.87%	train_lr:0.100000
> test_acc: 66.24% ,  test_loss : 1.054015
> [  7/200, 78 seconds]|	 loss: 0.986602, accuaracy: 67.15%	|	val_loss: 1.013462, val_accuaracy: 67.57%	train_lr:0.100000
> test_acc: 66.71% ,  test_loss : 1.030985
> [  8/200, 82 seconds]|	 loss: 0.952175, accuaracy: 68.93%	|	val_loss: 1.003335, val_accuaracy: 67.87%	train_lr:0.100000
> test_acc: 67.56% ,  test_loss : 1.014390
> [  9/200, 80 seconds]|	 loss: 0.917347, accuaracy: 69.83%	|	val_loss: 0.936724, val_accuaracy: 69.37%	train_lr:0.100000
> test_acc: 69.66% ,  test_loss : 0.924132
> [ 10/200, 80 seconds]|	 loss: 0.846754, accuaracy: 71.26%	|	val_loss: 0.852604, val_accuaracy: 71.14%	train_lr:0.100000
> test_acc: 70.89% ,  test_loss : 0.869030
> [ 11/200, 81 seconds]|	 loss: 0.821646, accuaracy: 72.51%	|	val_loss: 0.824014, val_accuaracy: 72.72%	train_lr:0.100000
> test_acc: 72.26% ,  test_loss : 0.845656
> [ 12/200, 81 seconds]|	 loss: 0.785699, accuaracy: 74.08%	|	val_loss: 0.785188, val_accuaracy: 73.77%	train_lr:0.100000
> test_acc: 73.38% ,  test_loss : 0.796236
> [ 13/200, 81 seconds]|	 loss: 0.612665, accuaracy: 80.28%	|	val_loss: 0.645819, val_accuaracy: 79.94%	train_lr:0.050000
> test_acc: 78.99% ,  test_loss : 0.667976
> [ 14/200, 80 seconds]|	 loss: 0.562087, accuaracy: 81.59%	|	val_loss: 0.593609, val_accuaracy: 80.51%	train_lr:0.050000
> test_acc: 80.19% ,  test_loss : 0.620839
> [ 15/200, 79 seconds]|	 loss: 0.653961, accuaracy: 78.09%	|	val_loss: 0.700986, val_accuaracy: 77.19%	train_lr:0.050000
> test_acc: 76.92% ,  test_loss : 0.698349
> [ 16/200, 80 seconds]|	 loss: 0.641346, accuaracy: 78.90%	|	val_loss: 0.694877, val_accuaracy: 77.62%	train_lr:0.050000
> test_acc: 76.56% ,  test_loss : 0.723869
> [ 17/200, 81 seconds]|	 loss: 0.570819, accuaracy: 81.40%	|	val_loss: 0.640584, val_accuaracy: 79.23%	train_lr:0.050000
> test_acc: 78.94% ,  test_loss : 0.652237
> [ 18/200, 81 seconds]|	 loss: 0.647558, accuaracy: 78.38%	|	val_loss: 0.725724, val_accuaracy: 75.47%	train_lr:0.050000
> test_acc: 76.21% ,  test_loss : 0.733160
> [ 19/200, 81 seconds]|	 loss: 0.576662, accuaracy: 81.03%	|	val_loss: 0.633693, val_accuaracy: 78.92%	train_lr:0.050000
> test_acc: 78.71% ,  test_loss : 0.663808
> [ 20/200, 82 seconds]|	 loss: 0.532334, accuaracy: 81.80%	|	val_loss: 0.576019, val_accuaracy: 80.60%	train_lr:0.050000
> test_acc: 79.69% ,  test_loss : 0.610959
> ```

我们可以从结果看到，我们大约训练了20次以后，我们的准确率就达到了80%左右

如果想详细了解VGG的图像分类问题，可以参考我的另一篇博客[VGG 系列的探索与pytorch实现 (CIFAR10 分类问题)](https://blog.csdn.net/weixin_45508265/article/details/117071577)

