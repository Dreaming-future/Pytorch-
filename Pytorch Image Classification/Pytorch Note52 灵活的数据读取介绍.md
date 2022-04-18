# Pytorch Note52 灵活的数据读取介绍

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)



图片数据一般有两种情况：

1、所有图片放在一个文件夹内，另外有一个txt文件显示标签。

2、不同类别的图片放在不同的文件夹内，文件夹就是图片的类别。

针对这两种不同的情况，数据集的准备也不相同，第一种情况可以自定义一个Dataset，第二种情况直接调用`torchvision.datasets.ImageFolder`来处理。下面分别进行说明：

## 灵活的数据读取

首先导入我们需要的函数

```python
from torchvision.datasets import ImageFolder
```

文件中数据分布是这样的，每个文件夹中有三张图片

![在这里插入图片描述](https://img-blog.csdnimg.cn/53bc4ca381ef49e48348f1bdf515d18f.png)

### 读入数据

```python
# 三个文件夹，每个文件夹一共有 3 张图片作为例子
folder_set = ImageFolder('./example_data/image/')
```

```python
# 查看名称和类别下标的对应
folder_set.class_to_idx
```

> ```
> {'class_1': 0, 'class_2': 1, 'class_3': 2}
> ```

```python
# 得到所有的图片名字和标签
folder_set.imgs
```

> ```python
> [('./example_data/image/class_1/1.png', 0),
>  ('./example_data/image/class_1/2.png', 0),
>  ('./example_data/image/class_1/3.png', 0),
>  ('./example_data/image/class_2/10.png', 1),
>  ('./example_data/image/class_2/11.png', 1),
>  ('./example_data/image/class_2/12.png', 1),
>  ('./example_data/image/class_3/16.png', 2),
>  ('./example_data/image/class_3/17.png', 2),
>  ('./example_data/image/class_3/18.png', 2)]
> ```

```python
# 取出其中一个数据
im, label = folder_set[0]
im
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/54dd4832aec94da1bf82a49e6b16b8c2.png)

```python
label
```

> ```python
> 0
> ```

### 传入数据预处理方式

```python
from torchvision import transforms as tfs
```

```python
# 传入数据预处理方式
data_tf = tfs.ToTensor()

folder_set = ImageFolder('./example_data/image/', transform=data_tf)

im, label = folder_set[0]
```

```python
im
```

> ```python
> tensor([[[0.2314, 0.1686, 0.1961,  ..., 0.6196, 0.5961, 0.5804],
>          [0.0627, 0.0000, 0.0706,  ..., 0.4824, 0.4667, 0.4784],
>          [0.0980, 0.0627, 0.1922,  ..., 0.4627, 0.4706, 0.4275],
>          ...,
>          [0.8157, 0.7882, 0.7765,  ..., 0.6275, 0.2196, 0.2078],
>          [0.7059, 0.6784, 0.7294,  ..., 0.7216, 0.3804, 0.3255],
>          [0.6941, 0.6588, 0.7020,  ..., 0.8471, 0.5922, 0.4824]],
> 
>         [[0.2431, 0.1804, 0.1882,  ..., 0.5176, 0.4902, 0.4863],
>          [0.0784, 0.0000, 0.0314,  ..., 0.3451, 0.3255, 0.3412],
>          [0.0941, 0.0275, 0.1059,  ..., 0.3294, 0.3294, 0.2863],
>          ...,
>          [0.6667, 0.6000, 0.6314,  ..., 0.5216, 0.1216, 0.1333],
>          [0.5451, 0.4824, 0.5647,  ..., 0.5804, 0.2431, 0.2078],
>          [0.5647, 0.5059, 0.5569,  ..., 0.7216, 0.4627, 0.3608]],
> 
>         [[0.2471, 0.1765, 0.1686,  ..., 0.4235, 0.4000, 0.4039],
>          [0.0784, 0.0000, 0.0000,  ..., 0.2157, 0.1961, 0.2235],
>          [0.0824, 0.0000, 0.0314,  ..., 0.1961, 0.1961, 0.1647],
>          ...,
>          [0.3765, 0.1333, 0.1020,  ..., 0.2745, 0.0275, 0.0784],
>          [0.3765, 0.1647, 0.1176,  ..., 0.3686, 0.1333, 0.1333],
>          [0.4549, 0.3686, 0.3412,  ..., 0.5490, 0.3294, 0.2824]]])
> ```

```python
label
```

> ```python
> 0
> ```

可以看到通过这种方式能够非常方便的访问每个数据点

### Dataset

```python
from torch.utils.data import Dataset
```

```python
# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform # 传入数据预处理
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        self.img_list = [i.split()[0] for i in lines] # 得到所有的图像名字
        self.label_list = [i.split()[1] for i in lines] # 得到所有的 label 

    def __getitem__(self, idx): # 根据 idx 取出其中一个
        img = self.img_list[idx]
        label = self.label_list[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self): # 总数据的多少
        return len(self.label_list)
```

```python
txt_dataset = custom_dataset('./example_data/train.txt') # 读入 txt 文件
```

```python
# 取得其中一个数据
data, label = txt_dataset[0]
print(data)
print(label)
```

> ```python
> 1009_2.png
> YOU
> ```

```python
# 再取一个
data2, label2 = txt_dataset[34]
print(data2)
print(label2)
```

> ```python
> 1046_7.png
> LIFE
> ```

所以通过这种方式我们也能够非常方便的定义一个数据读入，同时也能够方便的定义数据预处理

### DataLoader

```python
from torch.utils.data import DataLoader
```

```python
train_data1 = DataLoader(folder_set, batch_size=2, shuffle=True) # 将 2 个数据作为一个 batch
```

```python
for im, label in train_data1: # 访问迭代器
    print(label)
```

> ```python
> tensor([0, 2])
> tensor([1, 1])
> tensor([1, 2])
> tensor([0, 0])
> tensor([2])
> ```

可以看到，通过训练我们可以访问到所有的数据，这些数据被分为了 5 个 batch，前面 4 个都有两个数据，最后一个 batch 只有一个数据，因为一共有 9 个数据，同时顺序也被打乱了

## 例子

下面我们用自定义的数据读入举例子

```python
train_data2 = DataLoader(txt_dataset, 8, True) # batch size 设置为 8
```

```python
im, label = next(iter(train_data2)) # 使用这种方式访问迭代器中第一个 batch 的数据
```

```python
im
```

> ```python
> ('377_10.png',
>  '178_1.png',
>  '5008_4.png',
>  '5050_5.png',
>  '716_3.png',
>  '415_8.png',
>  '858_6.png',
>  '5086_10.png')
> ```

```python
label
```

> ```python
> ('AUGUST',
>  'OTKRIJTE',
>  'ASTAIRE',
>  'BOONMEE',
>  'OF',
>  'CAUTION',
>  'PROPANE',
>  'PECC')
> ```

现在有一个需求，希望能够将上面一个 batch 输出的 label 补成相同的长度，短的 label 用 0 填充，我们就需要使用 `collate_fn` 来自定义我们 batch 的处理方式，下面直接举例子

```python
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True) # 将数据集按照 label 的长度从大到小排序
    img, label = zip(*batch) # 将数据和 label 配对取出
    # 填充
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = label[i]
        temp_label += '0' * (max_len - len(label[i]))
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    pad_label 
    return img, pad_label, lens # 输出 label 的真实长度
```

使用我们自己定义 collate_fn 看看效果

```python
train_data3 = DataLoader(txt_dataset, 8, True, collate_fn=collate_fn) # batch size 设置为 8
```

```python
im, label, lens = next(iter(train_data3))
```

```python
im
```

> ```python
> ('5016_1.png',
>  '2314_3.png',
>  '731_9.png',
>  '5019_4.png',
>  '208_4.png',
>  '5017_12.png',
>  '5190_1.png',
>  '855_12.png')
> ```

```python
label
```

> ```python
> ['LINDSAY',
>  'ADDRESS',
>  'MAIDEN0',
>  'EINER00',
>  'INDIA00',
>  'GERE000',
>  'JAWS000',
>  'TD00000']
> ```

```python
lens
```

> ```python
> [7, 7, 6, 5, 5, 4, 4, 2]
> ```

可以看到一个 batch 中所有的 label 都从长到短进行排列，同时短的 label 都被补长了，所以使用 collate_fn 能够非常方便的处理一个 batch 中的数据，一般情况下，没有特别的要求，使用 pytorch 中内置的 collate_fn 就可以满足要求了

## 接下来是第二种情况，也是较为复杂的情况

第二种情况就是，所有文件都在一个文件夹下有图片，还必须要有事先标注好的label标签文件。

制作个人分类用数据集具体步骤如下：
1、将个人收集的图片归到一个文件夹内如下图：

<img src="https://img-blog.csdnimg.cn/20190905091913135.png" />

```python
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt='D:/CIFAR-10/images/data/label.txt', transform=transforms.ToTensor())
data_loader = DataLoader(train_data, batch_size=1,shuffle=True)
print(len(data_loader))
```

```python
data_loader.dataset.imgs
```

> ```python
> [('D:/CIFAR-10/images/data/0.jpg', 0),
>  ('D:/CIFAR-10/images/data/1.jpg', 1),
>  ('D:/CIFAR-10/images/data/2.jpg', 0),
>  ('D:/CIFAR-10/images/data/3.jpg', 1),
>  ('D:/CIFAR-10/images/data/4.jpg', 1),
>  ('D:/CIFAR-10/images/data/5.jpg', 0),
>  ('D:/CIFAR-10/images/data/6.jpg', 1),
>  ('D:/CIFAR-10/images/data/7.jpg', 1),
>  ('D:/CIFAR-10/images/data/8.jpg', 0),
>  ('D:/CIFAR-10/images/data/9.jpg', 0)]
> ```

其实和例子是类似的

```python
Image.open(data_loader.dataset.imgs[0][0])
```

![](https://img-blog.csdnimg.cn/3a1e7e0489db4b95be1082d23a552dd1.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6aOO5L-h5a2Q55qE54yrUmVkYW1hbmN5,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)
