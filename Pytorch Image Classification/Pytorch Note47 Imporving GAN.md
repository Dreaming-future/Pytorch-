# Pytorch Note47 Imporving GAN

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

在这一部分，我们介绍一些改善的生成对抗网络，因为简单的生成对抗网络存在一些问题，所以我们研究是否可以改善网络结构或者损失函数来解决这些问题。

## Least Squares GAN

[Least Squares GAN](https://arxiv.org/abs/1611.04076) 比最原始的 GANs 的 loss 更加稳定，通过名字我们也能够看出这种 GAN 是通过最小平方误差来进行估计，而不是通过二分类的损失函数，下面我们看看 loss 的计算公式
$$
\ell_G  =  \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[\left(D(G(z))-1\right)^2\right]
$$

$$
\ell_D = \frac{1}{2}\mathbb{E}_{x \sim p_\text{data}}\left[\left(D(x)-1\right)^2\right] + \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[ \left(D(G(z))\right)^2\right]
$$

可以看到 Least Squares GAN 通过最小二乘代替了二分类的 loss，下面我们定义一下 loss 函数

```python
def ls_discriminator_loss(scores_real, scores_fake):
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake ** 2).mean()
    return loss

def ls_generator_loss(scores_fake):
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss
```

```python
D = discriminator().cuda()
G = generator().cuda()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

train_a_gan(D, G, D_optim, G_optim, ls_discriminator_loss, ls_generator_loss)
```

> Iter: 0, D: 0.5524, G:0.4728
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/5b7f8eb09df7435abc7f0149168bd6fe.png)
>
> Iter: 250, D: 0.2155, G:0.1959
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/470bae2cf01b4d1a9a20ccc947d80684.png)
> ......
>
> Iter: 3500, D: 0.1186, G:0.3989
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/022273f0db7a4a2693a2c78fa4aa9726.png)
>
> Iter: 3750, D: 0.1621, G:0.228
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/5f6a494221df4456b754dc8595ca80a7.png)



## Deep Convolutional GANs
深度卷积生成对抗网络特别简单，就是将生成网络和对抗网络都改成了卷积网络的形式，下面我们来实现一下

### 卷积判别网络
卷积判别网络就是一个一般的卷积网络，结构如下

* 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
* Max Pool 2x2, Stride 2
* 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
* Max Pool 2x2, Stride 2
* Fully Connected size 4 x 4 x 64, Leaky ReLU(alpha=0.01)
* Fully Connected size 1



```python
class build_dc_classifier(nn.Module):
    def __init__(self):
        super(build_dc_classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
```



### 卷积生成网络

卷积生成网络需要将一个低维的噪声向量变成一个图片数据，结构如下

* Fully connected of size 1024, ReLU
* BatchNorm
* Fully connected of size 7 x 7 x 128, ReLU
* BatchNorm
* Reshape into Image Tensor
* 64 conv2d^T filters of 4x4, stride 2, padding 1, ReLU
* BatchNorm
* 1 conv2d^T filter of 4x4, stride 2, padding 1, TanH



```python
class build_dc_generator(nn.Module): 
    def __init__(self, noise_dim=NOISE_DIM):
        super(build_dc_generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7) # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        return x
```

```python
def train_dc_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=250, 
                noise_size=96, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = x.cuda() # 真实数据
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
D_DC = build_dc_classifier().cuda()
G_DC = build_dc_generator().cuda()

D_DC_optim = get_optimizer(D_DC)
G_DC_optim = get_optimizer(G_DC)

train_dc_gan(D_DC, G_DC, D_DC_optim, G_DC_optim, discriminator_loss, generator_loss, num_epochs=5)
```

> Iter: 0, D: 1.387, G:0.6381
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/d267c4b9fd7e44d2a083ac3686045274.png)
> Iter: 250, D: 0.7821, G:1.807
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/7bad9a7d6bd24ffea8c9ec7f7f15c0dd.png)
> ......
> Iter: 1500, D: 1.216, G:0.7218
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/38630561cf604a89a5cecb38bc138fd8.png)
> Iter: 1750, D: 1.143, G:1.092
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/785438d7d5f74c51a5fd2a0736a692bd.png)

可以看到，通过 DCGANs 能够得到更加清楚的结果，而且也可以更快地收敛

