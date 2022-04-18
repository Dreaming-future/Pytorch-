# Pytorch Note53 TensorBoard 可视化

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

TensorBoard是Tensorflow的可视化工具，它可以通过Tensorflow程序运行过程中输出的日志文件可视化Tensorflow程序的运行状态。TensorBoard和Tensorflow程序跑在不同的进程中，TensorBoard会自动读取最新的TensorFlow日志文件，并呈现当前TensorFlow程序运行的最新状态。

当然除了Tensorflow，也可以可视化我们的Pytorch



## 安装TensorBoard

首先安装TensorBoard是很简单的，我们用cmd打开我们的命令行，然后接着输入以下命令

```python
pip install tensorboard
```

接着成功安装即可

我们可以测试是否安装成功，可以在我们cmd命令中输入以下命令

```python
tensorboard --logdir=D:\
```

如果有结果，就说明我们安装成功了

## TensorBoard 的使用

安装完成后，我们用一个例子来演示一下TensorBoard的可视化

```python
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
```

我们将在该教程中定义一个类似的模型架构，仅需进行少量修改即可解决以下事实：图像现在是一个通道而不是三个通道，而图像是28x28而不是32x32：

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

我们将在之前定义相同的optimizer和criterion：

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### 1. TensorBoard 设置

现在，我们将设置 TensorBoard，从torch.utils导入tensorboard并定义SummaryWriter，这是将信息写入 TensorBoard 的关键对象。

```python
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```

请注意，仅此行会创建一个runs/fashion_mnist_experiment_1文件夹。

### 2. 写入 TensorBoard

现在，使用make_grid将图像写入到 TensorBoard 中，具体来说就是网格。

```python
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/7612751a1e1a43e8860dc7bbc7578fd1.png#pic_center)

```python
images.shape
```

> ```python
> torch.Size([4, 1, 28, 28])
> ```



### 启动TensorBoard

运行下面的命令可以启动TensorBoard

```python
load_ext tensorboard
```

```
tensorboard --logdir=runs
```

运行上面的命令会启动一个服务，这个父母的端口默认为6006。通过浏览器打开localhost：6006。使用--port参数可以改变启动服务的端口。
打开TensorBoard如下:

![在这里插入图片描述](https://img-blog.csdnimg.cn/79441ea6387a4a3386c85464e3e799b2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

### 3. 使用 TensorBoard 检查模型

TensorBoard 的优势之一是其可视化复杂模型结构的能力。 让我们可视化我们构建的模型。

```python
writer.add_graph(net, images)
writer.close()
```

继续并双击Net以展开它，查看构成模型的各个操作的详细视图。

TensorBoard 具有非常方便的功能，可在低维空间中可视化高维数据，例如图像数据。 接下来我们将介绍这一点。

![image-20210907144142750](C:\Users\86137\AppData\Roaming\Typora\typora-user-images\image-20210907144142750.png)

### 4. 在 TensorBoard 中添加“投影仪”

我们可以通过add_embedding方法可视化高维数据的低维表示

```python
import tensorboard as tb
import tensorflow as tf
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/73ab66d0f7ad4959bbc378453ae201c8.gif#pic_center)

### 5. 使用 TensorBoard 跟踪模型训练

在前面的示例中，我们仅每 2000 次迭代打印该模型的运行损失。 现在，我们将运行损失记录到 TensorBoard 中，并通过plot_classes_preds函数查看模型所做的预测。

```python
# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
```

```python
running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/308d0bdffc634d5ca1c121b56e90a5d3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

### 6. 使用 TensorBoard 评估经过训练的模型

可以得到每一类的概率和最后我们的PR曲线

```python
# 1\. gets the probability predictions in a test_size x num_classes Tensor
# 2\. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/a1d9326f7a054c509369e980ca2c31c9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 常见的问题 

这里会列出几个，我在使用TensorBoard中出现的问题，这样也方便大家不走弯路

### 1.杀死进程

我们可以利用以下命令杀死所有TensorBoard的进程

```python
taskkill /im tensorboard.exe /f
```

> ```python
> 成功: 已终止进程 "tensorboard.exe"，其 PID 为 6948。
> 成功: 已终止进程 "tensorboard.exe"，其 PID 为 25888。
> ```

或者会出现

> ```python
> 错误: 没有找到进程 "tensorboard.exe"。
> ```

说明已经没有tensorboard的进程了。

如果不可以，我们也可以换一个端口也就ok了

### 2.端口被占用

前面有说过，我们的默认端口是6006，当我们有多个程序的时候，我们可能需要换个端口，我们就可以修改我们启动TensorBoard的命令，比如我们将我们的端口改为6008

```python
tensorboard --logdir=runs --port=6008
```

这样即可，我们打开端口为6008的就成功了。

### 3.重启TensorBoard

```python
reload_ext tensorboard
```

## 例子

其实对于我们来说，有时候我们主要是用TensorBoard来可视化我们的损失和准确率，而不用那么多的结果，所以对我们来说，我们用的很简单，以下举个例子

```python
writer = SummaryWriter()

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data[0]
    return num_correct / total

if torch.cuda.is_available():
    net = net.cuda()
prev_time = datetime.now()
for epoch in range(30):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for im, label in train_data:
        if torch.cuda.is_available():
            im = Variable(im.cuda())  # (bs, 3, h, w)
            label = Variable(label.cuda())  # (bs, h, w)
        else:
            im = Variable(im)
            label = Variable(label)
        # forward
        output = net(im)
        loss = criterion(output, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        train_acc += get_acc(output, label)
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    valid_loss = 0
    valid_acc = 0
    net = net.eval()
    for im, label in valid_data:
        if torch.cuda.is_available():
            im = Variable(im.cuda(), volatile=True)
            label = Variable(label.cuda(), volatile=True)
        else:
            im = Variable(im, volatile=True)
            label = Variable(label, volatile=True)
        output = net(im)
        loss = criterion(output, label)
        valid_loss += loss.data[0]
        valid_acc += get_acc(output, label)
    epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
    prev_time = cur_time
    # ====================== 使用 tensorboard ==================
    writer.add_scalars('Loss', {'train': train_loss / len(train_data),
                                'valid': valid_loss / len(valid_data)}, epoch)
    writer.add_scalars('Acc', {'train': train_acc / len(train_data),
                               'valid': valid_acc / len(valid_data)}, epoch)
    # =========================================================
    print(epoch_str + time_str)
```

![img](img/tensorboard.jpg)