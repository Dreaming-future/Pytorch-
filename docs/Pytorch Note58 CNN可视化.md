# Pytoch Note58 CNN可视化

[toc]
全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)



> 卷积神经网络（CNN）是深度学习中非常重要的模型结构，它广泛地用于图像处理，极大地提升了模型表现，推动了计算机视觉的发展和进步。但CNN是一个“黑盒模型”，人们并不知道CNN是如何获得较好表现的，由此带来了深度学习的可解释性问题。如果能理解CNN工作的方式，人们不仅能够解释所获得的结果，提升模型的鲁棒性，而且还能有针对性地改进CNN的结构以获得进一步的效果提升。



理解CNN的重要一步是可视化，包括可视化特征是如何提取的、提取到的特征的形式以及模型在输入数据上的关注点等。

这一部分就会介绍一下

- 可视化CNN卷积核的方法
- 可视化CNN特征图的方法
- 可视化CNN显著图（class activation map）的方法



## 1.CNN卷积核可视化

我们在学习深度学习中，总会看到卷积核是用来提取我们的特征的，但是有时候，我们似乎不知道卷积核是如何提取我们的特征的，我们只能看到一个一个的张量，所以我们可以可视化我们的卷积核来加深理解，进而理解模型的工作原理。

在Zeiler和Fergus 2013年的[paper](https://arxiv.org/pdf/1311.2901.pdf)中就研究了CNN各个层的卷积核的不同，他们发现靠近输入的层提取的特征是相对简单的结构，而靠近输出的层提取的特征就和图中的实体形状相近了

下面给出几张论文中的图，从这几张图片也可以看出，越靠近输入层的一般是简单的结构，比如直线和圆的结构，靠近输出层的特征相对复杂，与实体相近。

![img](https://img-blog.csdnimg.cn/img_convert/1e9008ca3c15e64faabe564d20ed4903.png#pic_center)



### 1.1 layer1

![img](https://andrewhuman.github.io/images/cnn_first_layout_zeiler.png)

上图的每个格子同样代表一个卷积核,对于第一行第一列的格子可以粗略看出一条-45度直线,事实上这个格子的卷积核寻找的正是-45度左右的线条，它会对如下图片产生激活反应:

![img](https://andrewhuman.github.io/images/cnn_first_layout_0_0_filter_activation.png)

对于第三行第三列的格子而言，它寻找的是类似下图的色块

![img](https://andrewhuman.github.io/images/cnn_first_layout_activation_3_3.png)

所以总体来说对于一个训练好的模型来说，它的第一层总是在寻找这些简单的结构,不管是AleNet,ResNet还是DenseNet

### 1.2 layer2

![img](https://andrewhuman.github.io/images/cnn_second_layout_filter_activation.png)

可以看到输入相应图片后，网络激活输出了稍复杂的纹理结构,比如条纹(第一行),嵌套的圆环(第二行右面),色块等等。

### 1.3 layer3

![img](https://andrewhuman.github.io/images/cnn_third_layout_filter_activation.png)

第三层输出了第二层的组合，比如蜂巢，人,门窗和文字的轮廓

### 1.4 layer4

![img](https://andrewhuman.github.io/images/cnn_fourth_layout_filter_activation.png)

第四层，我们开始得到一些真实物品形状的东西，例如狗,准确说是狗的抽象形状

### 1.5 layer5

![img](https://andrewhuman.github.io/images/cnn_fifth_layout_filter_activation.png)

第五层，我们得到一些更高层次的抽象，比如右边第8行4列穿红衣服的女人，输出的是人脸部分，因为对于分类来说，神经网络使用人脸来区分图像是不是表示一个人,所以它只关心人脸部分

在PyTorch中可视化卷积核也非常方便，核心在于特定层的卷积核即特定层的模型权重，可视化卷积核就等价于可视化对应的权重矩阵。下面给出在PyTorch中可视化卷积核的实现方案，以torchvision自带的VGG11模型为例。

首先加载模型，并确定模型的层信息：

```python
import torch
from torchvision.models import vgg11

model = vgg11(pretrained=True)
print(dict(model.features.named_children()))
```

```python
{'0': Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '1': ReLU(inplace=True),
 '2': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '3': Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '4': ReLU(inplace=True),
 '5': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '6': Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '7': ReLU(inplace=True),
 '8': Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '9': ReLU(inplace=True),
 '10': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '11': Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '12': ReLU(inplace=True),
 '13': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '14': ReLU(inplace=True),
 '15': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '16': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '17': ReLU(inplace=True),
 '18': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '19': ReLU(inplace=True),
 '20': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)}
```

卷积核对应的应为卷积层（Conv2d），这里以第“3”层为例，可视化对应的参数：

```python
conv1 = dict(model.features.named_children())['3']
kernel_set = conv1.weight.detach()
num = len(conv1.weight.detach())
print(kernel_set.shape)
for i in range(0,num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20, 17))
    if (len(i_kernel)) > 1:
        for idx, filer in enumerate(i_kernel):
            plt.subplot(9, 9, idx+1) 
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='bwr')
```

```
torch.Size([128, 64, 3, 3])
```

由于第“3”层的特征图由64维变为128维，因此共有128*64个卷积核，其中部分卷积核可视化效果如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/7288d07573194f9585d8829549212fd5.png#pic_center)




对于这一部分来说，由于是3x3卷积，看不出太多的纹路，我们还可视化了一下AlexNet的5x5的卷积核，大概可以得到论文中的结果

```python
import torch
from torchvision.models import alexnet

model = alexnet(pretrained=True)
conv1 = dict(model.features.named_children())['3']
localw = conv1.weight.detach() 
print("total of number of filter : ", len(localw), localw.shape)
num = len(localw)
for i in range(1,num):
    localw0 = localw[i]
	# print(localw0.shape)    
    # mean of 3 channel.
    #localw0 = torch.mean(localw0,dim=0)
    # there should be 3(3 channels) 11 * 11 filter.
    plt.figure(figsize=(20, 17))
    if (len(localw0)) > 1:
        for idx, filer in enumerate(localw0):
            plt.subplot(9, 9, idx+1) 
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='gray')
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/56dda93d8772409fabbc3d9782021365.png#pic_center)

## 2.CNN特征图可视化方法

与卷积核相对应，输入的原始图像经过每次卷积层得到的数据称为特征图，可视化卷积核是为了看模型提取哪些特征，可视化特征图则是为了看模型提取到的特征是什么样子的。

获取特征图的方法有很多种，可以从输入开始，逐层做前向传播，直到想要的特征图处将其返回。尽管这种方法可行，但是有些麻烦了。在zfnethttps://arxiv.org/abs/1311.2901一篇论文中，使用转置卷积将特征图映射回原始图像空间。来观察每层的特征图。我们这里偷个懒直接将特征图从网络中拿出来，可视化。 在PyTorch中，提供了一个专用的接口使得网络在前向传播过程中能够获取到特征图，这个接口的名称非常形象，叫做hook。可以想象这样的场景，数据通过网络向前传播，网络某一层我们预先设置了一个钩子，数据传播过后钩子上会留下数据在这一层的样子，读取钩子的信息就是这一层的特征图。具体实现如下：

```python
class Hook(object):
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self,module, fea_in, fea_out):
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None
    

def plot_feature(model, idx, inputs):
    hh = Hook()
    model.features[idx].register_forward_hook(hh)
    
    # forward_model(model,False)
    model.eval()
    _ = model(inputs)
    print(hh.module_name)
    print((hh.features_in_hook[0][0].shape))
    print((hh.features_out_hook[0].shape))
    
    out1 = hh.features_out_hook[0]

    total_ft  = out1.shape[1]
    first_item = out1[0].cpu().clone()    

    plt.figure(figsize=(20, 17))
    

    for ftidx in range(total_ft):
        if ftidx > 99:
            break
        ft = first_item[ftidx]
        plt.subplot(10, 10, ftidx+1) 
        
        plt.axis('off')
        #plt.imshow(ft[ :, :].detach(),cmap='gray')
        plt.imshow(ft[ :, :].detach())
```

## 3.CNN class activation map可视化方法

class activation map （CAM）的作用是判断哪些变量对模型来说是重要的，在CNN可视化的场景下，即判断图像中哪些像素点对预测结果是重要的。除了确定重要的像素点，人们也会对重要区域的梯度感兴趣，因此在CAM的基础上也进一步改进得到了Grad-CAM（以及诸多变种）。CAM和Grad-CAM的示例如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/f6448184fae64d5c8ef8f52cbafe040e.png#pic_center)


相比可视化卷积核与可视化特征图，CAM系列可视化更为直观，能够一目了然地确定重要区域，进而进行可解释性分析或模型优化改进。CAM系列操作的实现可以通过开源工具包pytorch-grad-cam来实现。

- 安装

```bash
pip install grad-cam
```

- 一个简单的例子

```python
import torch
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model = vgg11(pretrained=True)
img_path = './dog.png'
# resize操作是为了和传入神经网络训练图片大小一致
img = Image.open(img_path).resize((224,224))
# 需要将原始图片转为np.float32格式并且在0-1之间 
rgb_img = np.float32(img)/255
plt.imshow(img)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/dea1022b061d442784f303a986661cc2.png#pic_center)


```python
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.features[-1]]
# 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
cam = GradCAM(model=model,target_layers=target_layers)
targets = [ClassifierOutputTarget(preds)]   
# 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
print(type(cam_img))
Image.fromarray(cam_img)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/f9e87cf884d9455582ab52baa6e78158.png#pic_center)

## 4.使用FlashTorch快速实现CNN可视化

其实随着时代的发展，我们也不需要一个一个写代码去可视化了，也封装成一个一个的库了。随着PyTorch社区的努力，目前已经有不少开源工具能够帮助我们快速实现CNN可视化。这里我们介绍其中的一个——[FlashTorch](https://github.com/MisaOgura/flashtorch)。



- 安装

```bash
pip install flashtorch
```

- 可视化梯度

```python
# Download example images
# !mkdir -p images
# !wget -nv \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/great_grey_owl.jpg \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/peacock.jpg   \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/toucan.jpg    \
#    -P /content/images

import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

model = models.alexnet(pretrained=True)
backprop = Backprop(model)

image = load_image('./great_grey_owl.jpg')
owl = apply_transforms(image)

target_class = 24
backprop.visualize(owl, target_class, guided=True, use_gpu=True)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/dd16f3fd3505419e83deb6f1d0bfa5e0.png#pic_center)

- 可视化卷积核

```python
import torchvision.models as models
from flashtorch.activmax import GradientAscent

model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)

# specify layer and filter info
conv5_1 = model.features[24]
conv5_1_filters = [45, 271, 363, 489]

g_ascent.visualize(conv5_1, conv5_1_filters, title="VGG16: conv5_1")
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/616b4d4df0a844958e60595c1f8094f3.png#pic_center)


参考

- [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [https://github.com/MisaOgura/flashtorch](https://github.com/MisaOgura/flashtorch)  
