# Pytorch Note42 LSTM 做词性预测

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

## 模型介绍

对于一个单词，会有这不同的词性，首先能够根据一个单词的后缀来初步判断，比如 -ly 这种后缀，很大概率是一个副词，除此之外，一个相同的单词可以表示两种不同的词性，比如 book 既可以表示名词，也可以表示动词，所以到底这个词是什么词性需要结合前后文来具体判断。

根据这个问题，我们可以使用 lstm 模型来进行预测，首先对于一个单词，可以将其看作一个序列，比如 apple 是由 a p p l e 这 5 个单词构成，这就形成了 5 的序列，我们可以对这些字符构建词嵌入，然后输入 lstm，就像 lstm 做图像分类一样，只取最后一个输出作为预测结果，整个单词的字符串能够形成一种记忆的特性，帮助我们更好的预测词性。

![](https://tva1.sinaimg.cn/large/006tKfTcgy1fmxi67w0f7j30ap05qq2u.jpg)

接着我们把这个单词和其前面几个单词构成序列，可以对这些单词构建新的词嵌入，最后输出结果是单词的词性，也就是根据前面几个词的信息对这个词的词性进行分类。

下面我们用例子来简单的说明

我们使用下面简单的训练集

```python
training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), 
                  ["NN", "V", "DET", "NN"])]
```

## 编码

接下来我们需要对单词和标签进行编码

```python
word_to_idx = {}
tag_to_idx = {}
for context, tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)
```

```python
word_to_idx
```

> ```python
> {'the': 0,
>  'dog': 1,
>  'ate': 2,
>  'apple': 3,
>  'everybody': 4,
>  'read': 5,
>  'that': 6,
>  'book': 7}
> ```

```python
tag_to_idx
```

> ```python
> {'det': 0, 'nn': 1, 'v': 2}
> ```

然后我们对字母进行编码

```python
alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]] = i
```

接着我们可以构建训练数据

```python
def make_sequence(x, dic): # 字符编码
    idx = [dic[i.lower()] for i in x]
    idx = torch.LongTensor(idx)
    return idx
```

```python
make_sequence('apple', char_to_idx)
```

> ```python
> tensor([ 0, 15, 15, 11,  4])
> ```

```python
training_data[1][0]
```

> ```python
> ['Everybody', 'read', 'that', 'book']
> ```

```python
make_sequence(training_data[1][0], word_to_idx)
```

> ```python
> tensor([4, 5, 6, 7])
> ```



## 构建单个字符的 lstm 模型

```python
class char_lstm(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(char_lstm, self).__init__()
        
        self.char_embed = nn.Embedding(n_char, char_dim)
        self.lstm = nn.LSTM(char_dim, char_hidden)
        
    def forward(self, x):
        x = self.char_embed(x)
        out, _ = self.lstm(x)
        return out[-1] # (batch, hidden)
```

## 构建词性分类的 lstm 模型

```python
class lstm_tagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, word_dim, 
                 char_hidden, word_hidden, n_tag):
        super(lstm_tagger, self).__init__()
        self.word_embed = nn.Embedding(n_word, word_dim)
        self.char_lstm = char_lstm(n_char, char_dim, char_hidden)
        self.word_lstm = nn.LSTM(word_dim + char_hidden, word_hidden)
        self.classify = nn.Linear(word_hidden, n_tag)
        
    def forward(self, x, word):
        char = []
        for w in word: # 对于每个单词做字符的 lstm
            char_list = make_sequence(w, char_to_idx)
            char_list = char_list.unsqueeze(1) # (seq, batch, feature) 满足 lstm 输入条件
            char_infor = self.char_lstm(Variable(char_list)) # (batch, char_hidden)
            char.append(char_infor)
        char = torch.stack(char, dim=0) # (seq, batch, feature)
        
        x = self.word_embed(x) # (batch, seq, word_dim)
        x = x.permute(1, 0, 2) # 改变顺序
        x = torch.cat((x, char), dim=2) # 沿着特征通道将每个词的词嵌入和字符 lstm 输出的结果拼接在一起
        x, _ = self.word_lstm(x)
        
        s, b, h = x.shape
        x = x.view(-1, h) # 重新 reshape 进行分类线性层
        out = self.classify(x)
        return out
```

## 开始训练

```python
net = lstm_tagger(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
```

```python
# 开始训练
for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        word_list = make_sequence(word, word_to_idx).unsqueeze(0) # 添加第一维 batch
        tag = make_sequence(tag, tag_to_idx)
        word_list = Variable(word_list)
        tag = Variable(tag)
        # 前向传播
        out = net(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.data
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(training_data)))
```

## 预测

最后我们可以看看预测的结果

```python
net = net.eval()
test_sent = 'Everybody ate the apple'
test = make_sequence(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())
print(out)
```

> ```python
> tensor([[-0.9812,  1.6600, -0.8180],
>         [-0.8561, -0.5893,  1.5312],
>         [ 1.7873, -0.6825, -0.6958],
>         [-0.3254,  1.7655, -1.3631]], grad_fn=<AddmmBackward>)
> ```

```python
print(tag_to_idx)
```

> ```python
> {'det': 0, 'nn': 1, 'v': 2}
> ```

最后可以得到上面的结果，因为最后一层的线性层没有使用 softmax，所以数值不太像一个概率，但是每一行数值最大的就表示属于该类，可以看到第一个单词 'Everybody' 属于 nn，第二个单词 'ate' 属于 v，第三个单词 'the' 属于det，第四个单词 'apple' 属于 nn，所以得到的这个预测结果是正确的
