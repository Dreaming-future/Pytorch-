# Pytorch Note45 生成对抗网络（GAN）

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

2014年，深度学习三巨头之一 lan Goodfellow 提出了生成对抗网络（ GenerativeAdversarial Networks, GANs）这一概念，刚开始并没有引起轰动，直到2016年，学界、业界对它的兴趣如“井喷”一样爆发，多篇重磅文章陆续发表，Lecun这样形容GANs`“adversarial training is the coolest thing since sliced bread”`。2016年12月NIPS大会上，Goodfellow 做了关于GANs的专题报告，使得GANs成为了当今最热门的研究领域之一，接下来介绍如今深度学习界的明星——生成对抗网络。

在我的博客之中有一个用通俗的语言讲清楚GANs，如果有兴趣可以去看看[一文看懂「生成对抗网络 - GAN」基本原理+10种典型算法+13种应用](https://blog.csdn.net/weixin_45508265/article/details/115446736)

## GANs

这种训练方式定义了一种全新的网络结构，就是生成对抗网络，也就是 GANs。这一部分，我们会形象地介绍生成对抗网络，以及用代码进行实现，也会在后面更加详细地介绍 GANs 的数学推导。

根据这个名字就可以知道这个网络是由两部分组成的，第一部分是生成，第二部分是对抗。简单来说，就是有一个生成网络和一个判别网络，通过训练让两个网络相互竞争，生成网络来生成假的数据，对抗网络通过判别器去判别真伪，最后希望生成器生成的数据能够以假乱真。

可以用这个图来简单的看一看这两个过程

![](https://img-blog.csdnimg.cn/img_convert/ef2470ad23a8463eff4f5665e7fbeeba.png)

## Discriminator Network

首先我们来讲一下对抗过程，因为这个过程更加简单。

对抗过程简单来说就是一个判断真假的判别器，相当于一个二分类问题，我们输入一张真的图片希望判别器输出的结果是1，输入一张假的图片希望判别器输出的结果是0。这其实已经和原图片的 label 没有关系了，不管原图片到底是一个多少类别的图片，他们都统一称为真的图片，label 是 1 表示真实的；而生成的假的图片的 label 是 0 表示假的。

我们训练的过程就是希望这个判别器能够正确的判出真的图片和假的图片，这其实就是一个简单的二分类问题，对于这个问题可以用我们前面讲过的很多方法去处理，比如 logistic 回归，深层网络，卷积神经网络，循环神经网络都可以。

## Generator Network

接着我们看看生成网络如何生成一张假的图片。首先给出一个简单的高维的正态分布的噪声向量，如上图所示的 D-dimensional noise vector，这个时候我们可以通过仿射变换，也就是 $xw+b$ 将其映射到一个更高的维度，然后将他重新排列成一个矩形，这样看着更像一张图片，接着进行一些卷积、转置卷积、池化、激活函数等进行处理，最后得到了一个与我们输入图片大小一模一样的噪音矩阵，这就是我们所说的假的图片。

这个时候我们如何去训练这个生成器呢？这就需要通过对抗学习，增大判别器判别这个结果为真的概率，通过这个步骤不断调整生成器的参数，希望生成的图片越来越像真的，而在这一步中我们不会更新判别器的参数，因为如果判别器不断被优化，可能生成器无论生成什么样的图片都无法骗过判别器。

生成器的效果可以看看下面的图示

![](https://img-blog.csdnimg.cn/img_convert/643d31d3738be745a328623704efeedd.png)

关于生成对抗网络，出现了很多变形，比如 WGAN，LS-GAN 等等，这一节我们只使用 mnist 举一些简单的例子来说明，更复杂的网络结构可以再 github 上找到相应的实现

## 简单版本的生成对抗网络

通过前面我们知道生成对抗网络有两个部分构成，一个是生成网络，一个是对抗网络，我们首先写一个简单版本的网络结构，生成网络和对抗网络都是简单的多层神经网络

### 判别网络

判别网络的结构非常简单，就是一个二分类器，结构如下:
* 全连接(784 -> 256)
* leakyrelu,  $\alpha$ 是 0.2
* 全连接(256 -> 256)
* leakyrelu, $\alpha$ 是 0.2
* 全连接(256 -> 1)

其中 leakyrelu 是指 f(x) = max($\alpha$ x, x)

```python
def discriminator():
    net = nn.Sequential(        
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    return net
```

### 生成网络
接下来我们看看生成网络，生成网络的结构也很简单，就是根据一个随机噪声生成一个和数据维度一样的张量，结构如下：
* 全连接(噪音维度 -> 1024)
* relu
* 全连接(1024 -> 1024)
* relu
* 全连接(1024 -> 784)
* tanh 将数据裁剪到 -1 ~ 1 之间

```python
def generator(noise_dim=NOISE_DIM):   
    net = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return net
```

接下来我们需要定义生成对抗网络的 loss，通过前面的讲解我们知道，对于对抗网络，相当于二分类问题，将真的判别为真的，假的判别为假的，作为辅助，可以参考一下论文中公式

$$
\ell_D = \mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]
$$
而对于生成网络，需要去骗过对抗网络，也就是将假的也判断为真的，作为辅助，可以参考一下论文中公式

$$
\ell_G  =  \mathbb{E}_{z \sim p(z)}\left[\log D(G(z))\right]
$$
如果你还记得前面的二分类 loss，那么你就会发现上面这两个公式就是二分类 loss

$$
bce(s, y) = y * \log(s) + (1 - y) * \log(1 - s)
$$
如果我们把 D(x) 看成真实数据的分类得分，那么 D(G(z)) 就是假数据的分类得分，所以上面判别器的 loss 就是将真实数据的得分判断为 1，假的数据的得分判断为 0，而生成器的 loss 就是将假的数据判断为 1

下面我们来实现一下

```python
bce_loss = nn.BCEWithLogitsLoss()

def discriminator_loss(logits_real, logits_fake): # 判别器的 loss
    size = logits_real.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()
    false_labels = torch.zeros(size, 1).float().cuda()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss
```

```python
def generator_loss(logits_fake): # 生成器的 loss  
    size = logits_fake.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()
    loss = bce_loss(logits_fake, true_labels)
    return loss
```

```python
# 使用 adam 来进行训练，学习率是 3e-4, beta1 是 0.5, beta2 是 0.999
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer
```

下面我们开始训练一个这个简单的生成对抗网络

```python
def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=250, 
                noise_size=96, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = x.view(bs, -1).cuda() # 真实数据
            logits_real = D_net(real_data) # 判别网络得分
            
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
            g_fake_seed = sample_noise.cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据
            logits_fake = D_net(fake_images) # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake) # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step() # 优化判别网络
            
            # 生成网络
            g_fake_seed = sample_noise.cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake) # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step() # 优化生成网络

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data, g_error.data))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
```

```python
D = discriminator().cuda()
G = generator().cuda()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss)
```



> Iter: 0, D: 1.433, G:0.7233
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/43f4a5c15b8149319f7deb1c12923629.png)
> Iter: 250, D: 1.578, G:0.7958
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/ec86ae2a5da843f68f716ad927cad3c7.png)
> ...
>
> Iter: 3500, D: 1.077, G:1.983
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/5a54e27f67c14104a95df4131cf6ddb2.png)
> Iter: 3750, D: 1.322, G:0.8375
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/39219ecea9c14b02b42693988c3f8863.png)



我们已经完成了一个简单的生成对抗网络，是不是非常容易呢。但是可以看到效果并不是特别好，生成的数字也不是特别完整，因为我们仅仅使用了简单的多层全连接网络。

除了这种最基本的生成对抗网络之外，还有很多生成对抗网络的变式，有结构上的变式，也有 loss 上的变式，接下来会看看。