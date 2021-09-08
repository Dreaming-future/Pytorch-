# Pytorch Note55 迁移学习实战猫狗分类

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

在这一部分，我会用迁移学习的方法，实现kaggle中的猫狗分类，这是一个二分类的问题，我们可以直接使用修改我们的预训练的网络卷积部分提取我们自己图片的特征，对于我们的猫狗二分类，我们就用自己的分类全连接层就可以了。

## 加载数据集

数据集可以去[https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)下载，一个是训练集文件夹，一个是测试集文件夹。这两个文件内部都有两个文件夹：一个文件夹放狗的图片，一个文件夹中放猫的图片

下载完成后，我这里需要将我们的图片移动成以下格式，方便我们加载数据，如果还不清楚加载数据的话，具体可以看[Note52 灵活的数据读取介绍](https://redamancy.blog.csdn.net/article/details/120151183)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2a31c1a4cbea46e09c4ce7d366598f47.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_14,color_FFFFFF,t_70,g_se,x_16)

接着我们就可以用我们的代码载入数据了

首先导入我们需要的包

```python
# import
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.models as models
import os
import time
```

定义对数据的transforms，标准差和均值都是取ImageNet的标准差和均值，保持一致

```python
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
```

读入数据，如果要换自己的路径，可以换root

```python
root = 'D:/data/dogs-vs-cats/'
trainset = torchvision.datasets.ImageFolder('D:/data/dogs-vs-cats/train',transform=transform)
valset = torchvision.datasets.ImageFolder('D:/data/dogs-vs-cats/val',transform=transform)
```

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True,num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=64,shuffle=False,num_workers=8)
```

我设定我们的训练集中，我们猫狗图片各10000张，验证集图片猫狗各2500张。总共就是训练集有20000张，验证集有5000张

```python
print(u"训练集个数:", len(trainset))
print(u"验证集个数:", len(valset))
```

> ```python
> 训练集个数: 20000
> 验证集个数: 5000
> ```

```python
trainset
```

> ```python
> Dataset ImageFolder
>     Number of datapoints: 20000
>     Root location: D:/data/dogs-vs-cats/train
>     StandardTransform
> Transform: Compose(
>                Resize(size=256, interpolation=bilinear)
>                CenterCrop(size=(224, 224))
>                ToTensor()
>                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
>            )
> ```

```python
trainset.class_to_idx
```

> ```python
> {'cat': 0, 'dog': 1}
> ```

```python
trainset.classes
```

> ```python
> ['cat', 'dog']
> ```

```python
trainset.imgs[0][0]
```

> ```python
> 'D:/data/dogs-vs-cats/train\\cat\\cat.0.jpg'
> ```

随机打开一张图片

```python
n = np.random.randint(0,20000)
Image.open(trainset.imgs[n][0])
```

> ![在这里插入图片描述](https://img-blog.csdnimg.cn/41700e1c5baa45a687487e391a672c6c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_9,color_FFFFFF,t_70,g_se,x_16#pic_center)

```python
trainset[0][0].shape
```

> ```python
> torch.Size([3, 224, 224])
> ```

## 迁移学习网络

```python
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### 定义训练模型的函数

```python
def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1) # 得到概率
    correct = (pred_y == label).sum().data
    return correct / total

def train(net,path = './model.pth',epoches = 10, writer = None, verbose = False):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=0.5,min_lr=1e-6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    train_acc_list, val_acc_list = [],[]
    train_loss_list, val_loss_list = [],[]
    lr_list  = []
    for i in range(epoches):
        start = time.time()
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        if torch.cuda.is_available():
            net = net.to(device)
        net.train()
        for step,data in enumerate(trainloader,start=0):
            im,label = data
            im = im.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # 释放内存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # formard
            outputs = net(im)
            loss = criterion(outputs,label)
            # backward
            loss.backward()
            # 更新参数
            optimizer.step()
            train_loss += loss.data
            train_acc += get_acc(outputs,label)
            # 打印下载进度
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * (50 - int(rate * 50))
            print('\r train {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
    #     print('train_loss:{:.6f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')  
        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        # 更新学习率
        scheduler.step(train_loss)

        net.eval()
        with torch.no_grad():
            for step,data in enumerate(valloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)
                # 释放内存
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                outputs = net(im)
                loss = criterion(outputs,label)
                val_loss += loss.data
                # probs, pred_y = outputs.data.max(dim=1) # 得到概率
                # test_acc += (pred_y==label).sum().item()
                # total += label.size(0)
                val_acc += get_acc(outputs,label)
                rate = (step + 1) / len(valloader)
                a = "*" * int(rate * 50)
                b = "." * (50 - int(rate * 50))
                print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
        val_loss = val_loss / len(valloader)
        val_acc = val_acc * 100 / len(valloader)
        if verbose:
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
        end = time.time()
        print(
            '\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Val Loss:{:>.6f}  Val Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                i + 1, epoches, train_loss, train_acc, val_loss, val_acc,lr), end='')
        
        time_ = int(end - start)
        h = time_ / 3600
        m = time_ % 3600 /60
        s = time_ % 60
        time_str = "\tTime %02d:%02d" % ( m, s)
        # ====================== 使用 tensorboard ==================
        if writer is not None:
            writer.add_scalars('Loss', {'train': train_loss,
                                    'valid': val_loss}, i+1)
            writer.add_scalars('Acc', {'train': train_acc ,
                                   'valid': val_acc}, i+1)
#             writer.add_scalars('Learning Rate',lr,i+1)
        # =========================================================
        # 打印所用时间
        print(time_str)
        # 如果取得更好的准确率，就保存模型
        if val_acc > best_acc:
            torch.save(net,path)
            best_acc = val_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['val_acc'] = val_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['val_loss'] = val_loss_list
    Lr = lr_list
    return Acc, Loss, Lr
```

### 定义一个测试的函数

```python
import matplotlib.pyplot as plt
def test(path, model):
    # 读取要预测的图片
    img = Image.open(path).convert('RGB') # 读取图像

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    class_indict = ["cat", "dog"]
    plt.imshow(img)
    img = data_transform(img)
    img = img.to(device)
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).data.cpu().numpy()
    print(class_indict[predict_cla], predict[predict_cla].data.cpu().numpy())
    plt.show()
```

```python
if not os.path.exists('./model/'):
    os.mkdir('./model')
```

### 1.AlexNet

```python
# 导入Pytorch封装的AlexNet网络模型
alexnet = models.alexnet(pretrained=True)
# 固定卷积层参数
for param in alexnet.parameters():
    param.requires_grad = False
# 获取最后一个全连接层的输入通道数
num_input = alexnet.classifier[6].in_features
# 获取全连接层的网络结构
feature_model = list(alexnet.classifier.children())
# 去掉原来的最后一层
feature_model.pop()
# 添加上适用于自己数据集的全连接层
feature_model.append(nn.Linear(num_input, 2))
# 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
# 还可以为网络添加新的层
# 重新生成网络的后半部分
alexnet.classifier = nn.Sequential(*feature_model)
for param in alexnet.classifier.parameters():
    param.requires_grad = True
alexnet = alexnet.to(device)
#打印一下
print(alexnet)
```

> ```python
> AlexNet(
>   (features): Sequential(
>     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
>     (1): ReLU(inplace=True)
>     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
>     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
>     (4): ReLU(inplace=True)
>     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
>     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
>     (7): ReLU(inplace=True)
>     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
>     (9): ReLU(inplace=True)
>     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
>     (11): ReLU(inplace=True)
>     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
>   )
>   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
>   (classifier): Sequential(
>     (0): Dropout(p=0.5, inplace=False)
>     (1): Linear(in_features=9216, out_features=4096, bias=True)
>     (2): ReLU(inplace=True)
>     (3): Dropout(p=0.5, inplace=False)
>     (4): Linear(in_features=4096, out_features=4096, bias=True)
>     (5): ReLU(inplace=True)
>     (6): Linear(in_features=4096, out_features=2, bias=True)
>   )
> )
> ```

```python
train(alexnet, "./model/alexnet.pth")
```

> ```python
> Epoch [  1/ 10]  Train Loss:0.192722  Train Acc:94.21% Val Loss:0.104133  Val Acc:96.02%  Learning Rate:0.010000	Time 05:26
> Epoch [  2/ 10]  Train Loss:0.080365  Train Acc:96.93% Val Loss:0.100261  Val Acc:96.22%  Learning Rate:0.010000	Time 04:34
> Epoch [  3/ 10]  Train Loss:0.052221  Train Acc:98.07% Val Loss:0.090963  Val Acc:96.80%  Learning Rate:0.010000	Time 04:01
> Epoch [  4/ 10]  Train Loss:0.040361  Train Acc:98.43% Val Loss:0.099036  Val Acc:96.86%  Learning Rate:0.010000	Time 02:34
> Epoch [  5/ 10]  Train Loss:0.032902  Train Acc:98.74% Val Loss:0.097670  Val Acc:96.97%  Learning Rate:0.010000	Time 04:21
> Epoch [  6/ 10]  Train Loss:0.024438  Train Acc:99.15% Val Loss:0.096651  Val Acc:96.82%  Learning Rate:0.010000	Time 03:53
> Epoch [  7/ 10]  Train Loss:0.017886  Train Acc:99.38% Val Loss:0.105171  Val Acc:97.11%  Learning Rate:0.010000	Time 01:39
> Epoch [  8/ 10]  Train Loss:0.014792  Train Acc:99.44% Val Loss:0.104840  Val Acc:96.91%  Learning Rate:0.010000	Time 01:52
> Epoch [  9/ 10]  Train Loss:0.013121  Train Acc:99.55% Val Loss:0.112335  Val Acc:96.93%  Learning Rate:0.010000	Time 01:37
> Epoch [ 10/ 10]  Train Loss:0.010472  Train Acc:99.60% Val Loss:0.122118  Val Acc:96.80%  Learning Rate:0.010000	Time 02:17
> ```

```python
n = np.random.randint(0,12500)
test(root + 'test/%d.jpg'% n ,alexnet)
```

> dog 1.0
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/58b2d39e9a174eba99d8bb170a7d1ecd.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_6,color_FFFFFF,t_70,g_se,x_16)

### 2.VGG16

```python
vgg16 = models.vgg16_bn(pretrained=True)
# 固定模型权重
for param in vgg16.parameters():
    param.requires_grad = False
    
# 最后加一个分类器
vgg16.classifier[6] = nn.Sequential(nn.Linear(4096, 2))
for param in vgg16.classifier[6].parameters():
    param.requires_grad = True
    
vgg16 = vgg16.to(device)
```

```python
train(vgg16, "./model/vgg16.pth")
```

> ```python
> Epoch [  1/ 10]  Train Loss:0.071487  Train Acc:97.33% Val Loss:0.047137  Val Acc:98.42%  Learning Rate:0.010000	Time 07:34
> Epoch [  2/ 10]  Train Loss:0.057413  Train Acc:97.89% Val Loss:0.037040  Val Acc:98.75%  Learning Rate:0.010000	Time 06:49
> Epoch [  3/ 10]  Train Loss:0.056943  Train Acc:97.99% Val Loss:0.040015  Val Acc:98.56%  Learning Rate:0.010000	Time 06:45
> Epoch [  4/ 10]  Train Loss:0.056988  Train Acc:97.90% Val Loss:0.044927  Val Acc:98.42%  Learning Rate:0.010000	Time 06:44
> Epoch [  5/ 10]  Train Loss:0.058226  Train Acc:97.89% Val Loss:0.040388  Val Acc:98.54%  Learning Rate:0.010000	Time 06:45
> Epoch [  6/ 10]  Train Loss:0.053336  Train Acc:98.06% Val Loss:0.040575  Val Acc:98.58%  Learning Rate:0.010000	Time 06:53
> Epoch [  7/ 10]  Train Loss:0.056317  Train Acc:97.95% Val Loss:0.043550  Val Acc:98.52%  Learning Rate:0.010000	Time 07:08
> Epoch [  8/ 10]  Train Loss:0.056145  Train Acc:97.96% Val Loss:0.038649  Val Acc:98.66%  Learning Rate:0.010000	Time 06:58
> Epoch [  9/ 10]  Train Loss:0.055673  Train Acc:97.93% Val Loss:0.052661  Val Acc:98.20%  Learning Rate:0.010000	Time 06:49
> Epoch [ 10/ 10]  Train Loss:0.061366  Train Acc:97.87% Val Loss:0.038335  Val Acc:98.64%  Learning Rate:0.010000	Time 07:10
> ```

```python
n = np.random.randint(0,12500)
test(root + 'test/%d.jpg'% n ,vgg16)
```

> cat 1.0
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/81bf40781777462a9cfe7a7a28df8408.png)

### 3.ResNet18

```python
resnet18 = models.resnet18(pretrained=True)

for param in resnet18.parameters():
    param.requires_grad = False
    
inchannel = resnet18.fc.in_features
resnet18.fc = nn.Linear(inchannel, 2)
for param in resnet18.fc.parameters():
    param.requires_grad = True
    
resnet18 = resnet18.to(device)
```

```python
train(resnet18, "./model/resnet18.pth")
```

> ```python
> Epoch [  1/ 10]  Train Loss:0.092101  Train Acc:96.34% Val Loss:0.058147  Val Acc:97.77%  Learning Rate:0.010000	Time 05:21
> Epoch [  2/ 10]  Train Loss:0.083978  Train Acc:96.88% Val Loss:0.081815  Val Acc:97.09%  Learning Rate:0.010000	Time 04:40
> Epoch [  3/ 10]  Train Loss:0.075946  Train Acc:97.08% Val Loss:0.063055  Val Acc:97.77%  Learning Rate:0.010000	Time 04:57
> Epoch [  4/ 10]  Train Loss:0.072917  Train Acc:97.34% Val Loss:0.055050  Val Acc:98.18%  Learning Rate:0.010000	Time 05:54
> Epoch [  5/ 10]  Train Loss:0.064799  Train Acc:97.59% Val Loss:0.056419  Val Acc:98.02%  Learning Rate:0.010000	Time 04:41
> Epoch [  6/ 10]  Train Loss:0.061354  Train Acc:97.68% Val Loss:0.054933  Val Acc:98.10%  Learning Rate:0.010000	Time 04:29
> Epoch [  7/ 10]  Train Loss:0.071761  Train Acc:97.54% Val Loss:0.065232  Val Acc:97.55%  Learning Rate:0.010000	Time 04:04
> Epoch [  8/ 10]  Train Loss:0.065548  Train Acc:97.57% Val Loss:0.052685  Val Acc:98.30%  Learning Rate:0.010000	Time 04:29
> Epoch [  9/ 10]  Train Loss:0.065787  Train Acc:97.54% Val Loss:0.078557  Val Acc:97.23%  Learning Rate:0.010000	Time 02:26
> Epoch [ 10/ 10]  Train Loss:0.063050  Train Acc:97.54% Val Loss:0.055315  Val Acc:98.20%  Learning Rate:0.010000	Time 01:33
> ```

```python
n = np.random.randint(0,12500)
test(root + 'test/%d.jpg'% n ,resnet18)
```

> dog 1.0
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/bd9cab3eba5f4a20b9f13b539ca9cd43.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_6,color_FFFFFF,t_70,g_se,x_16)

### 4.DenseNet

```python
densenet121 = models.densenet121(pretrained=True)

for param in densenet121.parameters():
    param.requires_grad = False
    
inchannel = densenet121.classifier.in_features
densenet121.classifier = nn.Linear(inchannel, 2)
for param in densenet121.classifier.parameters():
    param.requires_grad = True
    
densenet121 = densenet121.to(device)
```

```python
train(densenet121,"./model/densenet121.pth",epoches=50,writer=writer)
```

> ```python
> Epoch [  1/ 50]  Train Loss:0.069421  Train Acc:97.06% Val Loss:0.045408  Val Acc:98.50%  Learning Rate:0.010000	Time 04:12
> Epoch [  2/ 50]  Train Loss:0.055100  Train Acc:97.94% Val Loss:0.045141  Val Acc:98.60%  Learning Rate:0.010000	Time 03:50
> Epoch [  3/ 50]  Train Loss:0.048683  Train Acc:98.20% Val Loss:0.046111  Val Acc:98.36%  Learning Rate:0.010000	Time 04:07
> Epoch [  4/ 50]  Train Loss:0.047130  Train Acc:98.17% Val Loss:0.039344  Val Acc:98.67%  Learning Rate:0.010000	Time 04:03
> Epoch [  5/ 50]  Train Loss:0.048824  Train Acc:98.09% Val Loss:0.047738  Val Acc:98.58%  Learning Rate:0.010000	Time 03:29
> Epoch [  6/ 50]  Train Loss:0.051442  Train Acc:98.11% Val Loss:0.044026  Val Acc:98.58%  Learning Rate:0.010000	Time 03:37
> Epoch [  7/ 50]  Train Loss:0.042843  Train Acc:98.42% Val Loss:0.057952  Val Acc:97.96%  Learning Rate:0.010000	Time 04:08
> Epoch [  8/ 50]  Train Loss:0.049442  Train Acc:98.26% Val Loss:0.054428  Val Acc:98.28%  Learning Rate:0.010000	Time 03:31
> Epoch [  9/ 50]  Train Loss:0.043310  Train Acc:98.29% Val Loss:0.044551  Val Acc:98.40%  Learning Rate:0.010000	Time 03:30
> Epoch [ 10/ 50]  Train Loss:0.044042  Train Acc:98.40% Val Loss:0.042966  Val Acc:98.52%  Learning Rate:0.010000	Time 03:31
> Epoch [ 11/ 50]  Train Loss:0.045892  Train Acc:98.23% Val Loss:0.044504  Val Acc:98.56%  Learning Rate:0.010000	Time 03:31
> Epoch [ 12/ 50]  Train Loss:0.040170  Train Acc:98.57% Val Loss:0.045590  Val Acc:98.48%  Learning Rate:0.005000	Time 03:29
> Epoch [ 13/ 50]  Train Loss:0.037383  Train Acc:98.66% Val Loss:0.044364  Val Acc:98.56%  Learning Rate:0.005000	Time 03:29
> Epoch [ 14/ 50]  Train Loss:0.036059  Train Acc:98.53% Val Loss:0.044473  Val Acc:98.54%  Learning Rate:0.005000	Time 03:37
> Epoch [ 15/ 50]  Train Loss:0.036368  Train Acc:98.63% Val Loss:0.045188  Val Acc:98.50%  Learning Rate:0.005000	Time 03:35
> Epoch [ 16/ 50]  Train Loss:0.036568  Train Acc:98.53% Val Loss:0.046154  Val Acc:98.46%  Learning Rate:0.005000	Time 03:44
> Epoch [ 17/ 50]  Train Loss:0.037295  Train Acc:98.54% Val Loss:0.047695  Val Acc:98.38%  Learning Rate:0.005000	Time 03:53
> Epoch [ 18/ 50]  Train Loss:0.034725  Train Acc:98.74% Val Loss:0.044237  Val Acc:98.48%  Learning Rate:0.005000	Time 03:36
> Epoch [ 19/ 50]  Train Loss:0.035841  Train Acc:98.62% Val Loss:0.045616  Val Acc:98.38%  Learning Rate:0.005000	Time 03:36
> Epoch [ 20/ 50]  Train Loss:0.034830  Train Acc:98.61% Val Loss:0.046552  Val Acc:98.54%  Learning Rate:0.005000	Time 03:44
> Epoch [ 21/ 50]  Train Loss:0.037384  Train Acc:98.60% Val Loss:0.044396  Val Acc:98.38%  Learning Rate:0.005000	Time 04:22
> Epoch [ 22/ 50]  Train Loss:0.037995  Train Acc:98.56% Val Loss:0.047278  Val Acc:98.50%  Learning Rate:0.005000	Time 03:39
> Epoch [ 23/ 50]  Train Loss:0.032673  Train Acc:98.69% Val Loss:0.043948  Val Acc:98.50%  Learning Rate:0.002500	Time 03:59
> Epoch [ 24/ 50]  Train Loss:0.037515  Train Acc:98.57% Val Loss:0.043362  Val Acc:98.56%  Learning Rate:0.002500	Time 03:35
> Epoch [ 25/ 50]  Train Loss:0.031254  Train Acc:98.75% Val Loss:0.043706  Val Acc:98.48%  Learning Rate:0.002500	Time 03:51
> Epoch [ 26/ 50]  Train Loss:0.032019  Train Acc:98.76% Val Loss:0.044048  Val Acc:98.46%  Learning Rate:0.002500	Time 03:36
> Epoch [ 27/ 50]  Train Loss:0.037767  Train Acc:98.54% Val Loss:0.045632  Val Acc:98.48%  Learning Rate:0.002500	Time 04:42
> Epoch [ 28/ 50]  Train Loss:0.033983  Train Acc:98.64% Val Loss:0.047578  Val Acc:98.38%  Learning Rate:0.002500	Time 04:36
> Epoch [ 29/ 50]  Train Loss:0.034914  Train Acc:98.65% Val Loss:0.043730  Val Acc:98.44%  Learning Rate:0.002500	Time 04:54
> Epoch [ 30/ 50]  Train Loss:0.031600  Train Acc:98.80% Val Loss:0.042117  Val Acc:98.54%  Learning Rate:0.001250	Time 04:39
> Epoch [ 31/ 50]  Train Loss:0.033425  Train Acc:98.73% Val Loss:0.042952  Val Acc:98.54%  Learning Rate:0.001250	Time 03:52
> Epoch [ 32/ 50]  Train Loss:0.032280  Train Acc:98.80% Val Loss:0.043329  Val Acc:98.46%  Learning Rate:0.001250	Time 04:32
> Epoch [ 33/ 50]  Train Loss:0.030588  Train Acc:98.86% Val Loss:0.043638  Val Acc:98.52%  Learning Rate:0.001250	Time 04:19
> Epoch [ 34/ 50]  Train Loss:0.033875  Train Acc:98.65% Val Loss:0.042842  Val Acc:98.52%  Learning Rate:0.001250	Time 04:13
> Epoch [ 35/ 50]  Train Loss:0.032511  Train Acc:98.73% Val Loss:0.043455  Val Acc:98.48%  Learning Rate:0.001250	Time 04:03
> Epoch [ 36/ 50]  Train Loss:0.031035  Train Acc:98.81% Val Loss:0.042359  Val Acc:98.52%  Learning Rate:0.001250	Time 04:00
> Epoch [ 37/ 50]  Train Loss:0.032186  Train Acc:98.71% Val Loss:0.043344  Val Acc:98.44%  Learning Rate:0.001250	Time 03:54
> Epoch [ 38/ 50]  Train Loss:0.032048  Train Acc:98.75% Val Loss:0.044101  Val Acc:98.44%  Learning Rate:0.000625	Time 04:22
> Epoch [ 39/ 50]  Train Loss:0.030956  Train Acc:98.73% Val Loss:0.042923  Val Acc:98.50%  Learning Rate:0.000625	Time 05:09
> Epoch [ 40/ 50]  Train Loss:0.029514  Train Acc:98.82% Val Loss:0.043631  Val Acc:98.44%  Learning Rate:0.000625	Time 04:46
> Epoch [ 41/ 50]  Train Loss:0.032194  Train Acc:98.78% Val Loss:0.041953  Val Acc:98.50%  Learning Rate:0.000625	Time 04:07
> Epoch [ 42/ 50]  Train Loss:0.029889  Train Acc:98.82% Val Loss:0.043006  Val Acc:98.52%  Learning Rate:0.000625	Time 03:46
> Epoch [ 43/ 50]  Train Loss:0.032727  Train Acc:98.72% Val Loss:0.043430  Val Acc:98.52%  Learning Rate:0.000625	Time 04:47
> Epoch [ 44/ 50]  Train Loss:0.032861  Train Acc:98.69% Val Loss:0.045076  Val Acc:98.46%  Learning Rate:0.000625	Time 05:00
> Epoch [ 45/ 50]  Train Loss:0.030219  Train Acc:98.79% Val Loss:0.043679  Val Acc:98.52%  Learning Rate:0.000313	Time 05:03
> Epoch [ 46/ 50]  Train Loss:0.036379  Train Acc:98.64% Val Loss:0.042752  Val Acc:98.48%  Learning Rate:0.000313	Time 05:28
> Epoch [ 47/ 50]  Train Loss:0.033099  Train Acc:98.74% Val Loss:0.044587  Val Acc:98.48%  Learning Rate:0.000313	Time 05:14
> Epoch [ 48/ 50]  Train Loss:0.030044  Train Acc:98.87% Val Loss:0.043721  Val Acc:98.46%  Learning Rate:0.000313	Time 04:31
> Epoch [ 49/ 50]  Train Loss:0.030164  Train Acc:98.82% Val Loss:0.043604  Val Acc:98.40%  Learning Rate:0.000156	Time 04:33
> Epoch [ 50/ 50]  Train Loss:0.032786  Train Acc:98.73% Val Loss:0.042702  Val Acc:98.52%  Learning Rate:0.000156	Time 04:48
> ```

```python
n = np.random.randint(0,12500)
test(root + 'test/%d.jpg'% n ,densenet121)
```

> dog 0.9999974
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/3aaad7634ae1424aa479cfe601ea21a9.png)

### 5.MobileNet V2

```python
mobilenet = models.mobilenet_v2(pretrained=True)

for param in mobilenet.parameters():
    param.requires_grad = False
    
# 最后加一个分类器
mobilenet.classifier[1] = nn.Sequential(nn.Linear(1280, 2))
for param in mobilenet.classifier[1].parameters():
    param.requires_grad = True
    
mobilenet = mobilenet.to(device)
```

```python
train(mobilenet,"./model/mobilenet.pth")
```

> ```python
> Epoch [  1/ 10]  Train Loss:0.092556  Train Acc:96.13% Val Loss:0.056114  Val Acc:98.00%  Learning Rate:0.010000	Time 02:55
> Epoch [  2/ 10]  Train Loss:0.073321  Train Acc:97.14% Val Loss:0.052979  Val Acc:98.18%  Learning Rate:0.010000	Time 02:06
> Epoch [  3/ 10]  Train Loss:0.075686  Train Acc:97.23% Val Loss:0.060839  Val Acc:97.78%  Learning Rate:0.010000	Time 01:57
> Epoch [  4/ 10]  Train Loss:0.079644  Train Acc:96.93% Val Loss:0.073229  Val Acc:97.53%  Learning Rate:0.010000	Time 01:59
> Epoch [  5/ 10]  Train Loss:0.081197  Train Acc:96.96% Val Loss:0.052441  Val Acc:98.20%  Learning Rate:0.010000	Time 01:58
> Epoch [  6/ 10]  Train Loss:0.072206  Train Acc:97.19% Val Loss:0.052156  Val Acc:98.10%  Learning Rate:0.010000	Time 02:01
> Epoch [  7/ 10]  Train Loss:0.074642  Train Acc:97.15% Val Loss:0.053458  Val Acc:98.04%  Learning Rate:0.010000	Time 03:22
> Epoch [  8/ 10]  Train Loss:0.072543  Train Acc:97.33% Val Loss:0.060096  Val Acc:97.92%  Learning Rate:0.010000	Time 02:07
> Epoch [  9/ 10]  Train Loss:0.076153  Train Acc:97.17% Val Loss:0.052286  Val Acc:98.20%  Learning Rate:0.010000	Time 02:04
> Epoch [ 10/ 10]  Train Loss:0.072429  Train Acc:97.36% Val Loss:0.059538  Val Acc:97.98%  Learning Rate:0.010000	Time 02:05
> ```

```python
n = np.random.randint(0,12500)
test(root + 'test/%d.jpg'% n ,mobilenet)
```

> dog 0.99671364
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/4f734c1927ea4a0d9a46acf0dfed76aa.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_6,color_FFFFFF,t_70,g_se,x_16)

### 6.ShuffleNetV2

```python
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

for param in mobilenet.parameters():
    param.requires_grad = False
    
# 最后加一个分类器
inchannel = shufflenet.fc.in_features
shufflenet.fc = nn.Linear(inchannel, 2)
for param in shufflenet.fc.parameters():
    param.requires_grad = True
    
shufflenet = shufflenet.to(device)
```

```python
train(shufflenet,"./model/shufflenet.pth")
```

> ```python
> Epoch [  1/ 10]  Train Loss:0.407648  Train Acc:88.84% Val Loss:0.096014  Val Acc:97.61%  Learning Rate:0.010000	Time 03:10
> Epoch [  2/ 10]  Train Loss:0.068225  Train Acc:97.93% Val Loss:0.050991  Val Acc:98.38%  Learning Rate:0.010000	Time 02:18
> Epoch [  3/ 10]  Train Loss:0.034774  Train Acc:98.88% Val Loss:0.044972  Val Acc:98.46%  Learning Rate:0.010000	Time 02:37
> Epoch [  4/ 10]  Train Loss:0.020964  Train Acc:99.38% Val Loss:0.045224  Val Acc:98.32%  Learning Rate:0.010000	Time 02:22
> Epoch [  5/ 10]  Train Loss:0.017840  Train Acc:99.42% Val Loss:0.044285  Val Acc:98.48%  Learning Rate:0.010000	Time 02:21
> Epoch [  6/ 10]  Train Loss:0.011427  Train Acc:99.66% Val Loss:0.042665  Val Acc:98.58%  Learning Rate:0.010000	Time 02:22
> Epoch [  7/ 10]  Train Loss:0.008015  Train Acc:99.80% Val Loss:0.048528  Val Acc:98.32%  Learning Rate:0.010000	Time 02:59
> Epoch [  8/ 10]  Train Loss:0.008493  Train Acc:99.79% Val Loss:0.042361  Val Acc:98.64%  Learning Rate:0.010000	Time 03:44
> Epoch [  9/ 10]  Train Loss:0.005134  Train Acc:99.87% Val Loss:0.047370  Val Acc:98.67%  Learning Rate:0.010000	Time 03:07
> Epoch [ 10/ 10]  Train Loss:0.008110  Train Acc:99.75% Val Loss:0.051268  Val Acc:98.32%  Learning Rate:0.010000	Time 02:32
> ```

```python
n = np.random.randint(0,12500)
test(root + 'test/%d.jpg'% n ,shufflenet)
```

> dog 0.9999373
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/b7658d6040264e7dae77496195bcbfcb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_6,color_FFFFFF,t_70,g_se,x_16)

## 总结

前面分别使用了Alexnet、VGGNet、ResNet、DenseNet、MobileNet、ShuffleNet 六种网络结构来试验迁移学习，发现效果其实是都差不多都在94%左右。
**注**：这里并没有使用Inception网络结构，因为他的图片输入大小为（299,299），且他的网络输出结果有三个值，所以对应训练过程的损失也有三部分：

```python
images, labels = data
optimizer.zero_grad()
logits, aux_logits2, aux_logits1 = net(images.to(device))
loss0 = loss_function(logits, labels.to(device))
loss1 = loss_function(aux_logits1, labels.to(device))
loss2 = loss_function(aux_logits2, labels.to(device))
loss = loss0 + loss1 * 0.3 + loss2 * 0.3
loss.backward()
optimizer.step()
```
总的来说迁移学习优点真的是太好了，速度快，准确率高，而且还不用担心GPU内存不够，CPU都能给你跑。



最后给一个集成的代码

```python
import torch.nn as nn
from torchvision import datasets, models

# 你要分类的种类数
num_classes = 2

# 训练的批量大小（根据您拥有的内存量而变化）
BATCH_SIZE = 128

# 用于特征提取的标志。当为False时，我们会微调整个模型；当为True时，我们只会更新重塑的图层参数
feature_extract = True

#是否改变卷及层的参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # 初始化这些变量，这些变量将在此if语句中设置。这些变量中的每一个都是特定于模型的。
        model_ft = None
        input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224
                
    elif model_name == "shufflenet":
        """ShuffleNetV2
        """
        model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet":
        """ MobileNet V2 
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"#调用模型的名字
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
#print(model_ft, input_size)
```

