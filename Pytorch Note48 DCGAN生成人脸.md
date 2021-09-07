# Pytorch Note48 DCGAN生成人脸

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

## 生成对抗网络

### 什么是 GAN？

之前已经对GAN有了一个简单的介绍，并且对生成对抗网络的数学原理进行了一个较为简单的推导。详细可以查看[Note45 生成对抗网络](https://blog.csdn.net/weixin_45508265/article/details/119684311)，里面对GAN进行了一个较为详细的介绍，这里还是粗略的介绍一下。

GAN 是用于教授 DL 模型以捕获训练数据分布的框架，因此我们可以从同一分布中生成新数据。 GAN 由 Ian Goodfellow 于 2014 年发明，并在论文[《生成对抗网络》](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)中首次进行了描述。 它们由两个不同的模型组成：*生成器*和*判别器*。 生成器的工作是生成看起来像训练图像的“假”图像。 判别器的工作是查看图像并从生成器输出它是真实的训练图像还是伪图像。 在训练过程中，生成器不断尝试通过生成越来越好的伪造品而使判别器的表现超过智者，而判别器正在努力成为更好的侦探并正确地对真实和伪造图像进行分类。 博弈的平衡点是当生成器生成的伪造品看起来像直接来自训练数据时，而判别器则总是猜测生成器输出是真实还是伪造品的 50% 置信度。

现在，让我们从判别器开始定义一些在整个教程中使用的符号。 令`x`为代表图像的数据。 `D(x)`是判别器网络，其输出`x`来自训练数据而不是生成器的（标量）概率。 在这里，由于我们要处理图像，因此`D(x)`的输入是 CHW 大小为`3x64x64`的图像。 直观地，当`x`来自训练数据时，`D(x)`应该为高，而当`x`来自生成器时，它应该为低。 `D(x)`也可以被认为是传统的二分类器。

对于生成器的表示法，令`z`是从标准正态分布中采样的潜在空间向量。 `G(z)`表示将隐向量`z`映射到数据空间的生成器函数。 `G`的目标是估计训练数据来自`p_data`的分布，以便它可以从该估计分布（`p_g`）生成假样本。

因此，`D(G(z))`是生成器`G`的输出是真实图像的概率（标量）。 如 [Goodfellow 的论文](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)中所述，`D`和`G`玩一个$ minimax $​​游戏，其中`D`试图最大化其正确分类实物和假物`log D(x)`，并且`G`尝试最小化`D`预测其输出为假的概率`log(1 - D(G(g(x))))`。 从论文中来看，GAN 损失为


$$
\min _{G} \max _{D} V(G, D) 
$$
$$
V(G, D)=E_{x \sim P_{\text {data }}}[\log D(X)]+E_{x \sim P_{G}}[\log (1-D(X))]
$$

从理论上讲，此极小极大游戏的解决方案是$p_g = p_{data}$​，判别器会随机猜测输入是真实的还是假的。 但是，GAN 的收敛理论仍在积极研究中，实际上，模型并不总是能达到这一目的。

### 什么是 DCGAN？

DCGAN 是上述 GAN 的直接扩展，不同之处在于，DCGAN 分别在判别器和生成器中分别使用`卷积`和`卷积转置层`。 它最早由 Radford 等人，在论文[《使用深度卷积生成对抗网络的无监督表示学习》](https://arxiv.org/pdf/1511.06434.pdf)中描述。 判别器由分层的[卷积层](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)，[批量规范层](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d)和 [LeakyReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU) 激活组成。 输入是`3x64x64`的输入图像，输出是输入来自真实数据分布的标量概率。 生成器由[转置卷积层](https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d)，批量规范层和 [ReLU](https://pytorch.org/docs/stable/nn.html#relu) 激活组成。 输入是从标准正态分布中提取的潜向量`z`，输出是`3x64x64` RGB 图像。 跨步的转置层使潜向量可以转换为具有与图像相同形状的体积。 在本文中，作者还提供了一些有关如何设置优化器，如何计算损失函数以及如何初始化模型权重的提示，所有这些都将在接下来的部分中进行解释。

### 导入所需要的库

首先导入所需要的库，并且设置随机种子为999

```python
from __future__ import print_function
%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
```

> ```python
> Random Seed:  999
> ```



## 输入

让我们为跑步定义一些输入：

- `dataroot`-数据集文件夹根目录的路径。 
- `worker`-使用`DataLoader`加载数据的工作线程数
- `batch_size`-训练中使用的批量大小。 DCGAN 使用的Batch大小为 128（可以根据自己的电脑性能进行修改）
- `image_size`-用于训练的图像的空间大小。 此实现默认为`64x64`。 如果需要其他尺寸，则必须更改`D`和`G`的结构。 
- `nc`-输入图像中的彩色通道数。 对于彩色图像，这是 3
- `nz`-潜向量的长度
- `ngf`-与通过生成器传送的特征映射的深度有关
- `ndf`-设置通过判别器传播的特征映射的深度
- `num_epochs`-要运行的训练周期数。 训练更长的时间可能会导致更好的结果，但也会花费更长的时间
- `lr`-训练的学习率。 如 DCGAN 论文中所述，此数字应为 0.0002
- `beta1`-Adam 优化器的`beta1`超参数。 如论文所述，该数字应为 0.5
- `ngpu`-可用的 GPU 数量。 如果为 0，则代码将在 CPU 模式下运行。 如果此数字大于 0，它将在该数量的 GPU 上运行

```python
# Root directory for dataset
dataroot = 'D:/data/img_align_celeba'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
```

## 数据

在本教程中，我们将使用 [Celeb-A Faces 数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)，该数据集可在链接的站点或 [Google 云端硬盘](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)中下载。 数据集将下载为名为`img_align_celeba.zip`的文件。 下载完成后，创建一个名为`celeba`的目录，并将 zip 文件解压缩到该目录中。 然后，将此笔记本的`dataroot`输入设置为刚创建的`celeba`目录。 结果目录结构应为：

```python
/path/to/img_align_celeba
    -> img_align_celeba
        -> 188242.jpg
        -> 173822.jpg
        -> 284702.jpg
        -> 537394.jpg
           ...
```

大约一共有202,601张图片，所以数据集还是比较大的，如果希望测试数据，可以适当对数据集减小，所以我也存在一个little_celaba，里面有50,000张图片，虽然结果不会很好，但是也可以看出结果。此外，如果数据集下载有问题，无法连接Google硬盘，也可以从我的CSDN链接进行下载，或者自制数据集都是可以的。这里给出[链接](http://download.csdn.net/download/weixin_45508265/16487463)

这是重要的一步，因为我们将使用`ImageFolder`数据集类，该类要求数据集的根文件夹中有子目录。 现在，我们可以创建数据集，创建数据加载器，将设备设置为可以运行，并最终可视化一些训练数据。

```python
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/116c1f86b2a748e998e44749af6905c2.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 实现

设置好输入参数并准备好数据集后，我们现在可以进入实现了。 我们将从权重初始化策略开始，然后详细讨论生成器，判别器，损失函数和训练循环。

### 权重初始化

在 DCGAN 论文中，作者指定所有模型权重均应从均值为 0，`stdev = 0.02`的正态分布中随机初始化。 `weights_init`函数采用已初始化的模型作为输入，并重新初始化所有卷积，卷积转置和批量归一化层以满足此标准。 初始化后立即将此函数应用于模型。

```python
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    # 存在卷积Conv层
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # 标准初始化权重
    # 存在Batch层
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # 标准初始化权重
        nn.init.constant_(m.bias.data, 0) # 偏置项为0
```

### 生成器

生成器`G`用于将潜在空间向量（`z`）映射到数据空间。 由于我们的数据是图像，因此将`z`转换为数据空间意味着最终创建与训练图像大小相同的 RGB 图像（即`3x64x64`）。 在实践中，这是通过一系列跨步的二维卷积转置层来完成的，每个层都与 2d 批量规范层和 relu 激活配对。 生成器的输出通过 tanh 函数馈送，以使其返回到输入数据范围`[-1,1]`。 值得注意的是，在卷积转置层之后存在批量规范函数，因为这是 DCGAN 论文的关键贡献。 这些层有助于训练过程中的梯度流动。 DCGAN 纸生成的图像如下所示。

![dcgan_generator](https://pytorch.apachecn.org/docs/1.7/img/85974d98be6202902f21ce274418953f.png)

请注意，我们在输入部分中设置的输入（`nz`，`ngf`和`nc`）如何影响代码中的生成器架构。 `nz`是`z`输入向量的长度，`ngf`与通过生成器传播的特征映射的大小有关， `nc`是输出图像中的通道（对于 RGB 图像设置为 3）。 下面是生成器的代码。

```python
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```

现在，我们可以实例化生成器并应用`weights_init`函数。 签出打印的模型以查看生成器对象的结构。

```python
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu))) # 如果ngpu > 1，可以利用此代码并行运算

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
```

```python
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace=True)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
```

### 判别器

如前所述，判别器`D`是一个二分类网络，将图像作为输入并输出标量概率，即输入图像是真实的（与假的相对）。 在这里，`D`拍摄`3x64x64`的输入图像，通过一系列的`Conv2d`，`BatchNorm2d`和`LeakyReLU`层对其进行处理，然后通过 Sigmoid 激活函数输出最终概率。 如果需要解决此问题，可以用更多层扩展此架构，但是使用跨步卷积，`BatchNorm`和`LeakyReLU`仍然很重要。 DCGAN 论文提到，使用跨步卷积而不是通过池化来进行下采样是一个好习惯，因为它可以让网络学习自己的池化特征。 批量规范和泄漏 ReLU 函数还可以促进健康的梯度流，这对于`G`和`D`的学习过程都是至关重要的。

```python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

现在，与生成器一样，我们可以创建判别器，应用`weights_init`函数，并打印模型的结构。

```python
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)Copy
```

```py
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
```

### 损失函数和优化器

使用`D`和`G`设置，我们可以指定它们如何通过损失函数和优化器学习。 我们将使用在 PyTorch 中定义的二进制交叉熵损失（[BCELoss](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss)）函数：
$$
l(x,y) = L = \{l_1,\dots,l_N\}^T, l_n=-[y_n * \log x_n + (1-y_n) * \log(1-x_n)]
$$
请注意，此函数如何提供目标函数中两个对数分量的计算（即`log D(x)`和`log(1 - D(G(z)))`）。 我们可以指定`y`输入使用 BCE 方程的哪一部分。 这是在即将到来的训练循环中完成的，但重要的是要了解我们如何仅通过更改`y`（即`GT`标签）即可选择希望计算的分量。

接下来，我们将实际标签定义为 1，将假标签定义为 0。这些标签将在计算`D`和`G`的损失时使用，这也是 GAN 原始论文中使用的惯例 。 最后，我们设置了两个单独的优化器，一个用于`D`，另一个用于`G`。 如 DCGAN 论文中所指定，这两个都是学习速度为 0.0002 和`Beta1 = 0.5`的 Adam 优化器。 为了跟踪生成器的学习进度，我们将生成一批固定的潜在向量，这些向量是从高斯分布（即`fixed_noise`）中提取的。 在训练循环中，我们将定期将此`fixed_noise`输入到`G`中，并且在迭代过程中，我们将看到图像形成于噪声之外。

```python
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

### 训练

最后，既然我们已经定义了 GAN 框架的所有部分，我们就可以对其进行训练。 请注意，训练 GAN 某种程度上是一种艺术形式，因为不正确的超参数设置会导致模式崩溃，而对失败的原因几乎没有解释。 在这里，我们将严格遵循 Goodfellow 论文中的算法 1，同时遵守[`ganhacks`](https://github.com/soumith/ganhacks)中显示的一些最佳做法。 即，我们将“为真实和伪造构建不同的小批量”图像，并调整`G`的目标函数以最大化`log D(G(z))`。 训练分为两个主要部分。 第 1 部分更新了判别器，第 2 部分更新了生成器。

**第 1 部分-训练判别器**

回想一下，训练判别器的目的是最大程度地提高将给定输入正确分类为真实或伪造的可能性。 就古德费罗而言，我们希望“通过提高其随机梯度来更新判别器”。 实际上，我们要最大化`log D(x) + log(1 - D(G(z))`。 由于 ganhacks 提出了单独的小批量建议，因此我们将分两步进行计算。 首先，我们将从训练集中构造一批真实样本，向前通过`D`，计算损失（`log D(x)`），然后在向后通过中计算梯度。 其次，我们将使用当前生成器构造一批假样本，将这批伪造通过`D`，计算损失（`log(1 - D(G(z)))`），然后*反向累积*梯度。 现在，利用全批量和全批量的累积梯度，我们称之为判别器优化程序的一个步骤。

**第 2 部分-训练生成器**

如原始论文所述，我们希望通过最小化`log(1 - D(G(z)))`来训练生成器，以产生更好的假货。 如前所述，Goodfellow 证明这不能提供足够的梯度，尤其是在学习过程的早期。 作为解决方法，我们希望最大化`log D(G(z))`。 在代码中，我们通过以下步骤来实现此目的：将第 1 部分的生成器输出与判别器进行分类，使用实数标签`GT`计算`G`的损失，反向计算`G`的梯度，最后使用优化器步骤更新`G`的参数。 将真实标签用作损失函数的`GT`标签似乎是违反直觉的，但这使我们可以使用 BCELoss 的`log(x)`部分（而不是`log(1 - x)`部分），这正是我们想要的。

最后，我们将进行一些统计报告，并在每个周期结束时，将我们的`fixed_noise`批量推送到生成器中，以直观地跟踪`G`的训练进度。 报告的训练统计数据是：

- `Loss_D`-判别器损失，计算为所有真实批量和所有假批量的损失总和（`log D(x) + log D(G(z))`）。
- `Loss_G`-生成器损失计算为`log D(G(z))`
- `D(x)`-所有真实批量的判别器的平均输出（整个批量）。 这应该从接近 1 开始，然后在`G`变得更好时理论上收敛到 0.5。 想想这是为什么。
- `D(G(z))`-所有假批量的平均判别器输出。 第一个数字在`D`更新之前，第二个数字在`D`更新之后。 这些数字应从 0 开始，并随着`G`的提高收敛到 0.5。 想想这是为什么。

**注意**：此步骤可能需要一段时间，具体取决于您运行了多少个周期以及是否从数据集中删除了一些数据。

```python
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # label is all true
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            # save the false images
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
```

```python
Starting Training Loop...
[0/5][0/1583]	Loss_D: 1.7603	Loss_G: 5.0981	D(x): 0.5610	D(G(z)): 0.5963 / 0.0094
[0/5][50/1583]	Loss_D: 0.1727	Loss_G: 5.3953	D(x): 0.9503	D(G(z)): 0.0264 / 0.0140
[0/5][100/1583]	Loss_D: 0.3148	Loss_G: 4.2363	D(x): 0.8191	D(G(z)): 0.0270 / 0.0202
[0/5][150/1583]	Loss_D: 1.0460	Loss_G: 8.6272	D(x): 0.9613	D(G(z)): 0.5313 / 0.0006
[0/5][200/1583]	Loss_D: 0.3595	Loss_G: 4.6256	D(x): 0.9286	D(G(z)): 0.1856 / 0.0223
[0/5][250/1583]	Loss_D: 0.3802	Loss_G: 6.3810	D(x): 0.8899	D(G(z)): 0.1775 / 0.0047
[0/5][300/1583]	Loss_D: 0.8038	Loss_G: 6.7089	D(x): 0.8833	D(G(z)): 0.4243 / 0.0028
[0/5][350/1583]	Loss_D: 0.4610	Loss_G: 3.1248	D(x): 0.7281	D(G(z)): 0.0379 / 0.0673
[0/5][400/1583]	Loss_D: 1.7346	Loss_G: 11.0075	D(x): 0.9585	D(G(z)): 0.7320 / 0.0009
[0/5][450/1583]	Loss_D: 0.3837	Loss_G: 7.2247	D(x): 0.9205	D(G(z)): 0.2195 / 0.0016
[0/5][500/1583]	Loss_D: 0.5408	Loss_G: 3.4679	D(x): 0.7647	D(G(z)): 0.1689 / 0.0536
[0/5][550/1583]	Loss_D: 1.3597	Loss_G: 3.5977	D(x): 0.3992	D(G(z)): 0.0149 / 0.0531
[0/5][600/1583]	Loss_D: 0.6044	Loss_G: 3.6962	D(x): 0.7361	D(G(z)): 0.1200 / 0.0458
[0/5][650/1583]	Loss_D: 0.5643	Loss_G: 4.0031	D(x): 0.7969	D(G(z)): 0.2265 / 0.0275
[0/5][700/1583]	Loss_D: 1.0412	Loss_G: 1.1770	D(x): 0.4787	D(G(z)): 0.0173 / 0.4303
[0/5][750/1583]	Loss_D: 0.9833	Loss_G: 6.3992	D(x): 0.9483	D(G(z)): 0.5194 / 0.0047
[0/5][800/1583]	Loss_D: 0.4343	Loss_G: 3.8997	D(x): 0.7903	D(G(z)): 0.1009 / 0.0374
[0/5][850/1583]	Loss_D: 0.4335	Loss_G: 3.5301	D(x): 0.8011	D(G(z)): 0.1233 / 0.0513
[0/5][900/1583]	Loss_D: 0.9858	Loss_G: 2.8850	D(x): 0.5010	D(G(z)): 0.0263 / 0.0954
[0/5][950/1583]	Loss_D: 0.3718	Loss_G: 5.2125	D(x): 0.8881	D(G(z)): 0.1704 / 0.0123
[0/5][1000/1583]	Loss_D: 0.7203	Loss_G: 3.7901	D(x): 0.7544	D(G(z)): 0.2288 / 0.0371
[0/5][1050/1583]	Loss_D: 0.9680	Loss_G: 6.4439	D(x): 0.4846	D(G(z)): 0.0055 / 0.0045
[0/5][1100/1583]	Loss_D: 0.3800	Loss_G: 4.4296	D(x): 0.9164	D(G(z)): 0.2316 / 0.0217
[0/5][1150/1583]	Loss_D: 0.3262	Loss_G: 3.4205	D(x): 0.7945	D(G(z)): 0.0432 / 0.0568
[0/5][1200/1583]	Loss_D: 0.2945	Loss_G: 5.1929	D(x): 0.9296	D(G(z)): 0.1745 / 0.0092
[0/5][1250/1583]	Loss_D: 0.6466	Loss_G: 5.3215	D(x): 0.9288	D(G(z)): 0.3762 / 0.0102
[0/5][1300/1583]	Loss_D: 0.2936	Loss_G: 4.4877	D(x): 0.8669	D(G(z)): 0.1055 / 0.0198
[0/5][1350/1583]	Loss_D: 0.4120	Loss_G: 3.3041	D(x): 0.8495	D(G(z)): 0.1753 / 0.0591
[0/5][1400/1583]	Loss_D: 0.3433	Loss_G: 2.8770	D(x): 0.8975	D(G(z)): 0.1734 / 0.0935
[0/5][1450/1583]	Loss_D: 0.8761	Loss_G: 1.6780	D(x): 0.5392	D(G(z)): 0.0250 / 0.2520
[0/5][1500/1583]	Loss_D: 0.8508	Loss_G: 6.6078	D(x): 0.9541	D(G(z)): 0.4790 / 0.0035
[0/5][1550/1583]	Loss_D: 0.4240	Loss_G: 4.3567	D(x): 0.7394	D(G(z)): 0.0280 / 0.0263
[1/5][0/1583]	Loss_D: 0.7572	Loss_G: 8.4950	D(x): 0.9621	D(G(z)): 0.4453 / 0.0011
[1/5][50/1583]	Loss_D: 0.8186	Loss_G: 2.7094	D(x): 0.5832	D(G(z)): 0.0720 / 0.1167
[1/5][100/1583]	Loss_D: 0.5986	Loss_G: 2.8633	D(x): 0.6314	D(G(z)): 0.0274 / 0.0982
[1/5][150/1583]	Loss_D: 0.4370	Loss_G: 4.1904	D(x): 0.8188	D(G(z)): 0.1653 / 0.0261
[1/5][200/1583]	Loss_D: 0.3838	Loss_G: 3.4224	D(x): 0.8047	D(G(z)): 0.1106 / 0.0501
[1/5][250/1583]	Loss_D: 0.3507	Loss_G: 3.2371	D(x): 0.7936	D(G(z)): 0.0543 / 0.0626
[1/5][300/1583]	Loss_D: 0.8295	Loss_G: 0.9354	D(x): 0.5447	D(G(z)): 0.0097 / 0.4863
[1/5][350/1583]	Loss_D: 0.8640	Loss_G: 6.2494	D(x): 0.9553	D(G(z)): 0.4936 / 0.0032
[1/5][400/1583]	Loss_D: 1.3040	Loss_G: 8.3122	D(x): 0.8442	D(G(z)): 0.5849 / 0.0010
[1/5][450/1583]	Loss_D: 0.5020	Loss_G: 4.4474	D(x): 0.6834	D(G(z)): 0.0187 / 0.0317
[1/5][500/1583]	Loss_D: 0.3361	Loss_G: 3.8919	D(x): 0.8829	D(G(z)): 0.1610 / 0.0323
[1/5][550/1583]	Loss_D: 0.3074	Loss_G: 3.9042	D(x): 0.8332	D(G(z)): 0.0963 / 0.0340
[1/5][600/1583]	Loss_D: 0.4614	Loss_G: 2.8913	D(x): 0.7971	D(G(z)): 0.1383 / 0.0920
[1/5][650/1583]	Loss_D: 0.4557	Loss_G: 3.2309	D(x): 0.7825	D(G(z)): 0.1129 / 0.0620
[1/5][700/1583]	Loss_D: 0.2976	Loss_G: 3.7391	D(x): 0.8507	D(G(z)): 0.0904 / 0.0347
[1/5][750/1583]	Loss_D: 0.8371	Loss_G: 1.5715	D(x): 0.5565	D(G(z)): 0.0781 / 0.3129
[1/5][800/1583]	Loss_D: 0.4768	Loss_G: 2.7674	D(x): 0.7554	D(G(z)): 0.1128 / 0.1002
[1/5][850/1583]	Loss_D: 0.5819	Loss_G: 3.4087	D(x): 0.6723	D(G(z)): 0.0202 / 0.0609
[1/5][900/1583]	Loss_D: 0.8694	Loss_G: 1.1044	D(x): 0.5140	D(G(z)): 0.0099 / 0.4134
[1/5][950/1583]	Loss_D: 0.3131	Loss_G: 4.4502	D(x): 0.8940	D(G(z)): 0.1512 / 0.0216
[1/5][1000/1583]	Loss_D: 0.5248	Loss_G: 2.5302	D(x): 0.7485	D(G(z)): 0.1450 / 0.1152
[1/5][1050/1583]	Loss_D: 0.9438	Loss_G: 7.1672	D(x): 0.9680	D(G(z)): 0.5247 / 0.0018
[1/5][1100/1583]	Loss_D: 0.4514	Loss_G: 3.2238	D(x): 0.8215	D(G(z)): 0.1739 / 0.0608
[1/5][1150/1583]	Loss_D: 0.7385	Loss_G: 3.0609	D(x): 0.5727	D(G(z)): 0.0074 / 0.0822
[1/5][1200/1583]	Loss_D: 0.5681	Loss_G: 1.8565	D(x): 0.6734	D(G(z)): 0.0924 / 0.2142
[1/5][1250/1583]	Loss_D: 0.3442	Loss_G: 2.9841	D(x): 0.9430	D(G(z)): 0.2129 / 0.0836
[1/5][1300/1583]	Loss_D: 0.2960	Loss_G: 3.2145	D(x): 0.8595	D(G(z)): 0.1122 / 0.0613
[1/5][1350/1583]	Loss_D: 0.5397	Loss_G: 4.2935	D(x): 0.8580	D(G(z)): 0.2767 / 0.0217
[1/5][1400/1583]	Loss_D: 0.7478	Loss_G: 1.6391	D(x): 0.6473	D(G(z)): 0.1689 / 0.2602
[1/5][1450/1583]	Loss_D: 0.8304	Loss_G: 3.8670	D(x): 0.8398	D(G(z)): 0.4063 / 0.0338
[1/5][1500/1583]	Loss_D: 0.5487	Loss_G: 2.5004	D(x): 0.6715	D(G(z)): 0.0695 / 0.1302
[1/5][1550/1583]	Loss_D: 0.7358	Loss_G: 1.8881	D(x): 0.6222	D(G(z)): 0.1460 / 0.2245
[2/5][0/1583]	Loss_D: 0.4687	Loss_G: 3.1785	D(x): 0.7594	D(G(z)): 0.1392 / 0.0584
[2/5][50/1583]	Loss_D: 0.4938	Loss_G: 2.0276	D(x): 0.7646	D(G(z)): 0.1532 / 0.1772
[2/5][100/1583]	Loss_D: 0.5880	Loss_G: 4.1535	D(x): 0.8937	D(G(z)): 0.3386 / 0.0249
[2/5][150/1583]	Loss_D: 0.3699	Loss_G: 2.8783	D(x): 0.7662	D(G(z)): 0.0728 / 0.0802
[2/5][200/1583]	Loss_D: 0.9458	Loss_G: 2.2540	D(x): 0.4786	D(G(z)): 0.0332 / 0.1409
[2/5][250/1583]	Loss_D: 1.3651	Loss_G: 5.2262	D(x): 0.9318	D(G(z)): 0.6522 / 0.0132
[2/5][300/1583]	Loss_D: 0.8559	Loss_G: 4.2634	D(x): 0.8428	D(G(z)): 0.4354 / 0.0217
[2/5][350/1583]	Loss_D: 0.8685	Loss_G: 1.0340	D(x): 0.5074	D(G(z)): 0.0445 / 0.4164
[2/5][400/1583]	Loss_D: 0.5277	Loss_G: 4.1390	D(x): 0.8899	D(G(z)): 0.2957 / 0.0242
[2/5][450/1583]	Loss_D: 0.3848	Loss_G: 3.8379	D(x): 0.8515	D(G(z)): 0.1614 / 0.0375
[2/5][500/1583]	Loss_D: 0.5812	Loss_G: 2.7140	D(x): 0.7485	D(G(z)): 0.2051 / 0.0894
[2/5][550/1583]	Loss_D: 0.6208	Loss_G: 4.0429	D(x): 0.8851	D(G(z)): 0.3508 / 0.0242
[2/5][600/1583]	Loss_D: 0.5614	Loss_G: 2.6051	D(x): 0.8797	D(G(z)): 0.3136 / 0.1036
[2/5][650/1583]	Loss_D: 1.1518	Loss_G: 5.0309	D(x): 0.9496	D(G(z)): 0.6211 / 0.0096
[2/5][700/1583]	Loss_D: 1.1758	Loss_G: 5.9429	D(x): 0.9595	D(G(z)): 0.6228 / 0.0059
[2/5][750/1583]	Loss_D: 1.4850	Loss_G: 6.7221	D(x): 0.9484	D(G(z)): 0.6921 / 0.0023
[2/5][800/1583]	Loss_D: 0.6603	Loss_G: 3.0022	D(x): 0.8330	D(G(z)): 0.3392 / 0.0677
[2/5][850/1583]	Loss_D: 0.5060	Loss_G: 2.4951	D(x): 0.8372	D(G(z)): 0.2476 / 0.1074
[2/5][900/1583]	Loss_D: 0.3748	Loss_G: 2.8606	D(x): 0.8703	D(G(z)): 0.1905 / 0.0764
[2/5][950/1583]	Loss_D: 1.0564	Loss_G: 5.3408	D(x): 0.9534	D(G(z)): 0.5815 / 0.0080
[2/5][1000/1583]	Loss_D: 0.4661	Loss_G: 2.4021	D(x): 0.7769	D(G(z)): 0.1510 / 0.1196
[2/5][1050/1583]	Loss_D: 0.5569	Loss_G: 1.9636	D(x): 0.6671	D(G(z)): 0.0863 / 0.1822
[2/5][1100/1583]	Loss_D: 0.4839	Loss_G: 2.0121	D(x): 0.7349	D(G(z)): 0.1298 / 0.1611
[2/5][1150/1583]	Loss_D: 0.4571	Loss_G: 3.7059	D(x): 0.9360	D(G(z)): 0.2998 / 0.0348
[2/5][1200/1583]	Loss_D: 0.8312	Loss_G: 1.8929	D(x): 0.6783	D(G(z)): 0.2812 / 0.1922
[2/5][1250/1583]	Loss_D: 1.3269	Loss_G: 0.5262	D(x): 0.3733	D(G(z)): 0.1073 / 0.6284
[2/5][1300/1583]	Loss_D: 1.0485	Loss_G: 1.0856	D(x): 0.4512	D(G(z)): 0.0832 / 0.3958
[2/5][1350/1583]	Loss_D: 0.4749	Loss_G: 3.0244	D(x): 0.8420	D(G(z)): 0.2211 / 0.0678
[2/5][1400/1583]	Loss_D: 0.7603	Loss_G: 1.4797	D(x): 0.5504	D(G(z)): 0.0599 / 0.2739
[2/5][1450/1583]	Loss_D: 0.4936	Loss_G: 2.7009	D(x): 0.7372	D(G(z)): 0.1412 / 0.0893
[2/5][1500/1583]	Loss_D: 0.9367	Loss_G: 3.4466	D(x): 0.9211	D(G(z)): 0.5239 / 0.0464
[2/5][1550/1583]	Loss_D: 1.1805	Loss_G: 0.7536	D(x): 0.3738	D(G(z)): 0.0252 / 0.5321
[3/5][0/1583]	Loss_D: 0.6461	Loss_G: 2.0928	D(x): 0.5776	D(G(z)): 0.0314 / 0.1521
[3/5][50/1583]	Loss_D: 0.4533	Loss_G: 3.1588	D(x): 0.8476	D(G(z)): 0.2258 / 0.0557
[3/5][100/1583]	Loss_D: 0.7107	Loss_G: 1.2674	D(x): 0.6316	D(G(z)): 0.1550 / 0.3356
[3/5][150/1583]	Loss_D: 0.5477	Loss_G: 2.3475	D(x): 0.7712	D(G(z)): 0.2169 / 0.1235
[3/5][200/1583]	Loss_D: 0.4624	Loss_G: 2.1223	D(x): 0.7432	D(G(z)): 0.1193 / 0.1535
[3/5][250/1583]	Loss_D: 0.4723	Loss_G: 3.3965	D(x): 0.9238	D(G(z)): 0.2916 / 0.0488
[3/5][300/1583]	Loss_D: 0.5094	Loss_G: 2.3130	D(x): 0.7707	D(G(z)): 0.1782 / 0.1244
[3/5][350/1583]	Loss_D: 1.3670	Loss_G: 4.3643	D(x): 0.9005	D(G(z)): 0.6446 / 0.0207
[3/5][400/1583]	Loss_D: 0.4813	Loss_G: 3.1625	D(x): 0.8903	D(G(z)): 0.2792 / 0.0582
[3/5][450/1583]	Loss_D: 0.7487	Loss_G: 1.8650	D(x): 0.6228	D(G(z)): 0.1675 / 0.1992
[3/5][500/1583]	Loss_D: 0.6106	Loss_G: 3.0764	D(x): 0.8729	D(G(z)): 0.3363 / 0.0625
[3/5][550/1583]	Loss_D: 0.5066	Loss_G: 3.5693	D(x): 0.8467	D(G(z)): 0.2606 / 0.0373
[3/5][600/1583]	Loss_D: 0.4646	Loss_G: 2.0340	D(x): 0.7695	D(G(z)): 0.1480 / 0.1662
[3/5][650/1583]	Loss_D: 0.9872	Loss_G: 0.8669	D(x): 0.4872	D(G(z)): 0.1353 / 0.4752
[3/5][700/1583]	Loss_D: 0.6353	Loss_G: 1.6368	D(x): 0.6242	D(G(z)): 0.1019 / 0.2336
[3/5][750/1583]	Loss_D: 0.5890	Loss_G: 1.8652	D(x): 0.7084	D(G(z)): 0.1728 / 0.1897
[3/5][800/1583]	Loss_D: 0.4900	Loss_G: 2.4562	D(x): 0.7961	D(G(z)): 0.1990 / 0.1126
[3/5][850/1583]	Loss_D: 0.7937	Loss_G: 3.8138	D(x): 0.8770	D(G(z)): 0.4414 / 0.0302
[3/5][900/1583]	Loss_D: 0.5494	Loss_G: 3.3731	D(x): 0.8802	D(G(z)): 0.3186 / 0.0438
[3/5][950/1583]	Loss_D: 0.7981	Loss_G: 1.0441	D(x): 0.5539	D(G(z)): 0.0979 / 0.3950
[3/5][1000/1583]	Loss_D: 0.6517	Loss_G: 3.8005	D(x): 0.8521	D(G(z)): 0.3445 / 0.0334
[3/5][1050/1583]	Loss_D: 0.6440	Loss_G: 2.5199	D(x): 0.8099	D(G(z)): 0.3027 / 0.1103
[3/5][1100/1583]	Loss_D: 0.3902	Loss_G: 2.8809	D(x): 0.7980	D(G(z)): 0.1274 / 0.0744
[3/5][1150/1583]	Loss_D: 0.4874	Loss_G: 2.5236	D(x): 0.8155	D(G(z)): 0.2093 / 0.1058
[3/5][1200/1583]	Loss_D: 1.2330	Loss_G: 4.9618	D(x): 0.9033	D(G(z)): 0.6108 / 0.0134
[3/5][1250/1583]	Loss_D: 0.7017	Loss_G: 1.7945	D(x): 0.6837	D(G(z)): 0.2207 / 0.2039
[3/5][1300/1583]	Loss_D: 0.5203	Loss_G: 3.3688	D(x): 0.9051	D(G(z)): 0.3195 / 0.0445
[3/5][1350/1583]	Loss_D: 0.7667	Loss_G: 3.6694	D(x): 0.8690	D(G(z)): 0.4155 / 0.0396
[3/5][1400/1583]	Loss_D: 0.7485	Loss_G: 2.1347	D(x): 0.7461	D(G(z)): 0.3164 / 0.1466
[3/5][1450/1583]	Loss_D: 0.4987	Loss_G: 2.8715	D(x): 0.8164	D(G(z)): 0.2274 / 0.0716
[3/5][1500/1583]	Loss_D: 0.5107	Loss_G: 2.0005	D(x): 0.7593	D(G(z)): 0.1800 / 0.1654
[3/5][1550/1583]	Loss_D: 0.6989	Loss_G: 3.6959	D(x): 0.9159	D(G(z)): 0.4117 / 0.0348
[4/5][0/1583]	Loss_D: 0.5676	Loss_G: 2.3827	D(x): 0.8238	D(G(z)): 0.2816 / 0.1144
[4/5][50/1583]	Loss_D: 1.2728	Loss_G: 1.1088	D(x): 0.3736	D(G(z)): 0.0797 / 0.3972
[4/5][100/1583]	Loss_D: 1.3056	Loss_G: 0.9638	D(x): 0.4166	D(G(z)): 0.1809 / 0.4643
[4/5][150/1583]	Loss_D: 0.9012	Loss_G: 1.2633	D(x): 0.4691	D(G(z)): 0.0410 / 0.3404
[4/5][200/1583]	Loss_D: 1.2636	Loss_G: 4.1848	D(x): 0.9224	D(G(z)): 0.6215 / 0.0242
[4/5][250/1583]	Loss_D: 0.3177	Loss_G: 2.8631	D(x): 0.8634	D(G(z)): 0.1405 / 0.0784
[4/5][300/1583]	Loss_D: 0.4207	Loss_G: 2.4761	D(x): 0.8009	D(G(z)): 0.1537 / 0.1083
[4/5][350/1583]	Loss_D: 2.9878	Loss_G: 0.1553	D(x): 0.1051	D(G(z)): 0.0638 / 0.8720
[4/5][400/1583]	Loss_D: 0.7927	Loss_G: 3.3412	D(x): 0.8631	D(G(z)): 0.4332 / 0.0482
[4/5][450/1583]	Loss_D: 0.6376	Loss_G: 1.0313	D(x): 0.6310	D(G(z)): 0.1168 / 0.3975
[4/5][500/1583]	Loss_D: 0.6364	Loss_G: 3.0330	D(x): 0.8155	D(G(z)): 0.3154 / 0.0649
[4/5][550/1583]	Loss_D: 0.4288	Loss_G: 3.2750	D(x): 0.8640	D(G(z)): 0.2178 / 0.0541
[4/5][600/1583]	Loss_D: 0.6667	Loss_G: 2.6836	D(x): 0.8197	D(G(z)): 0.3248 / 0.0859
[4/5][650/1583]	Loss_D: 0.5062	Loss_G: 1.5188	D(x): 0.7045	D(G(z)): 0.1055 / 0.2643
[4/5][700/1583]	Loss_D: 0.5691	Loss_G: 2.7826	D(x): 0.8282	D(G(z)): 0.2820 / 0.0825
[4/5][750/1583]	Loss_D: 1.0212	Loss_G: 4.2888	D(x): 0.9228	D(G(z)): 0.5562 / 0.0205
[4/5][800/1583]	Loss_D: 0.6097	Loss_G: 3.3780	D(x): 0.8366	D(G(z)): 0.3143 / 0.0464
[4/5][850/1583]	Loss_D: 0.3560	Loss_G: 3.2729	D(x): 0.8905	D(G(z)): 0.1903 / 0.0527
[4/5][900/1583]	Loss_D: 0.6433	Loss_G: 3.7473	D(x): 0.8545	D(G(z)): 0.3482 / 0.0339
[4/5][950/1583]	Loss_D: 0.9391	Loss_G: 0.9512	D(x): 0.4787	D(G(z)): 0.0734 / 0.4396
[4/5][1000/1583]	Loss_D: 0.6153	Loss_G: 2.1201	D(x): 0.7829	D(G(z)): 0.2685 / 0.1510
[4/5][1050/1583]	Loss_D: 0.4937	Loss_G: 2.2968	D(x): 0.7430	D(G(z)): 0.1459 / 0.1303
[4/5][1100/1583]	Loss_D: 0.9423	Loss_G: 4.5192	D(x): 0.9176	D(G(z)): 0.5317 / 0.0158
[4/5][1150/1583]	Loss_D: 1.8675	Loss_G: 0.4250	D(x): 0.2189	D(G(z)): 0.0618 / 0.6851
[4/5][1200/1583]	Loss_D: 1.1241	Loss_G: 5.0819	D(x): 0.9651	D(G(z)): 0.6029 / 0.0101
[4/5][1250/1583]	Loss_D: 2.0676	Loss_G: 0.3945	D(x): 0.1665	D(G(z)): 0.0162 / 0.7166
[4/5][1300/1583]	Loss_D: 0.4080	Loss_G: 2.5124	D(x): 0.8133	D(G(z)): 0.1605 / 0.1026
[4/5][1350/1583]	Loss_D: 0.5476	Loss_G: 1.5342	D(x): 0.6674	D(G(z)): 0.0878 / 0.2537
[4/5][1400/1583]	Loss_D: 0.5454	Loss_G: 3.1446	D(x): 0.8476	D(G(z)): 0.2815 / 0.0568
[4/5][1450/1583]	Loss_D: 0.5763	Loss_G: 1.9525	D(x): 0.7694	D(G(z)): 0.2344 / 0.1711
[4/5][1500/1583]	Loss_D: 0.7142	Loss_G: 1.0892	D(x): 0.6125	D(G(z)): 0.1304 / 0.3755
[4/5][1550/1583]	Loss_D: 0.7555	Loss_G: 1.6924	D(x): 0.5499	D(G(z)): 0.0639 / 0.2309
```

对于这个结果，我大概花了4h 10m 6s 所以说如果只是验证一下结果，可以利用小一点的数据集进行判断，同样的，有可能生成的图片不一定那么好。

## 结果

最后，让我们看看我们是如何做到的。 在这里，我们将看三个不同的结果。 首先，我们将了解`D`和`G`的损失在训练过程中如何变化。 其次，我们将在每个周期将`G`的输出显示为`fixed_noise`批量。 第三，我们将查看一批真实数据以及来自`G`的一批伪数据。

### **损失与训练迭代**

下面是`D&G`的损失与训练迭代的关系图。

```python
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/6cdf3df53ab74efabbc70fb9cc6dd816.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

### **可视化`G`的进度**

请记住，在每次训练之后，我们如何将生成器的输出保存为`fixed_noise`批量。 现在，我们可以用动画形象化`G`的训练进度。 按下播放按钮开始动画。

注：这个结果是利用小数据集得出来的。

```python
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
```

![../_img/sphx_glr_dcgan_faces_tutorial_003.png](https://pytorch.apachecn.org/docs/1.7/img/2a31b55ef7bfff0c24c35bc635656078.png)

### **真实图像和伪图像**

最后，让我们并排查看一些真实图像和伪图像。

```python
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/e531bb6dd4e74d09b3416f4ec4b2f3f0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)