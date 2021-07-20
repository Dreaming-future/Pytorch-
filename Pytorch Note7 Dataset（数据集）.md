# Pytorch Note7 Dataset（数据集）

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)
## Dataset（数据集）
在处理任何机器学习问题之前都需要数据读取，并进行预处理。PyTorch提供了很多工具使得数据的读取和预处理变得很容易。


torchvision包 包含了目前流行的数据集，模型结构和常用的图片转换工具
torchvision.datasets中包含了以下数据集
- MNIST
- COCO（用于图像标注和目标检测）(Captioning and Detection)
- LSUN Classification
- ImageFolder
- Imagenet-12
- CIFAR10 and CIFAR100
- STL10

```python
from torch.utils.data import Dataset
import pandas as pd
 
class myDataset(Dataset):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        data = (self.csv_data[idx], self.txt_data[idx])
        return data
```

## DataLoader（数据加载器）

- dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；

- 使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问；

- 也可以使用`for inputs, labels in dataloaders`进行可迭代对象的访问；

- 一般我们实现一个datasets对象，传入到dataloader中；然后内部使用yeild返回每一次batch的数据；


输入数据PipeLine
pytorch 的数据加载到模型的操作顺序是这样的：


 1. 创建一个 Dataset 对象
 2. 创建一个 DataLoader 对象
 3. 循环这个 DataLoader 对象，将img, label加载到模型中进行训练

首先简单介绍一下DataLoader，它是PyTorch中数据读取的一个重要接口，该接口定义在dataloader.py中，只要是用PyTorch来训练模型基本都会用到该接口（除非用户重写…），该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。

官方对DataLoader的说明是：“数据加载由数据集和采样器组成，基于python的单、多进程的iterators来处理数据。”关于iterator和iterable的区别和概念请自行查阅，在实现中的差别就是iterators有__iter__和__next__方法，而iterable只有__iter__方法。

参数:
- Dataset: 加载数据的数据集
- batch_size(int, optional): 加载批训练的数据个数
- Shuffle(bool, optional): 如果为True，在每个epoch开始的时候，对数据进行重新排序
- Sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
- batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引）
- num_workers (int, optional): 用于数据加载的子进程数。0表示数据将在主进程中加载。
- collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数，合并样本列表
- pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
- drop_last (bool, optional): 如果数据集大小不能被batch_size整除，设置为True可删除最后一个不完整的批处理。如果设置为False并且数据集的大小不能被batch_size整除，则最后一个batch会更小


```python
from torch.utils.data import DataLoader
 
dataiter = DataLoader(myDataset,batch_size=32,shuffle=True,collate_fn=defaulf_collate)
```

其中的参数都很清楚，只有collate_fn是标识如何取样本的，我们可以定义自己的函数来准确地实现想要的功能，默认的函数在一般情况下都是可以使用的。


**（需要注意的是，Dataset类只相当于一个打包工具，包含了数据的地址。真正把数据读入内存的过程是由Dataloader进行批迭代输入的时候进行的。）**


## torchvision.datasets.ImageFolder
另外在torchvison这个包中还有一个更高级的有关于计算机视觉的数据读取类：ImageFolder，主要功能是处理图片，且要求图片是下面这种存放形式：

```python
root/dog/xxx.png
 
root/dog/xxy.png
 
root/dog/xxz.png
 
root/cat/123.png
 
root/cat/asd/png
 
root/cat/zxc.png
```
之后这样来调用这个类：

```python
from torchvision.datasets import ImageFolder
 
dset = ImageFolder(root='root_path', transform=None, loader=default_loader)
```
其中 root 需要是根目录，在这个目录下有几个文件夹，每个文件夹表示一个类别：transform 和 target_transform 是图片增强，后面我们会详细介绍；loader是图片读取的办法，因为我们读取的是图片的名字，然后通过 loader 将图片转换成我们需要的图片类型进入神经网络。



下一章传送门：[Note8 简单介绍torch.optim(优化)和模型保存](https://blog.csdn.net/weixin_45508265/article/details/117819532)