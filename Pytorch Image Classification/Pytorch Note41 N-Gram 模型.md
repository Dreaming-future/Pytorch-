# Pytorch Note41 N-Gram 模型

[toc]

全部笔记的汇总贴：[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

首先我们介绍一下 N-Gram 模型的原理和其要解决的问题。对于一句话，单词的排列顺序是非常重要的，所以我们能否由前面的几个词来预测后面的几个单词呢，比如 'I lived in France for 10 years, I can speak _' 这句话中，我们能够预测出最后一个词是 French。

对于一句话 T，其由 $w_1, w_2, \cdots, w_n$ 这 n 个词构成，

$$
P(T) = P(w_1)P(w_2 | w_1)P(w_3 |w_2 w_1) \cdots P(w_n |w_{n-1} w_{n-2}\cdots w_2w_1)
$$

我们可以再简化一下这个模型，比如对于一个词，它并不需要前面所有的词作为条件概率，也就是说一个词可以只与其前面的几个词有关，这就是马尔科夫假设。

对于这里的条件概率，传统的方法是统计语料中每个词出现的频率，根据贝叶斯定理来估计这个条件概率，这里我们就可以用词嵌入对其进行代替，然后使用 RNN 进行条件概率的计算，然后最大化这个条件概率。不仅修改词嵌入，同时能够使得模型可以依据计算的条件概率对其中的一个单词进行预测。

## 单词预测的 Pytorch 实现

```python
CONTEXT_SIZE = 2 # 依据的单词数
EMBEDDING_DIM = 10 # 词向量的维度
# 我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine  deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
```

这里的 `CONTEXT_SIZE` 表示我们希望由前面几个单词来预测这个单词，这里使用两个单词，`EMBEDDING_DIM` 表示词嵌入的维度。

接着我们建立训练集，便利整个语料库，将单词三个分组，前面两个作为输入，最后一个作为预测的结果。

```python
 trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2]) 
            for i in range(len(test_sentence)-2)]
```

```python
# 总的数据量
len(trigram)
```

> ```python
> 113
> ```

```python
# 取出第一个数据看看
trigram[0]
```

> ```python
> (('When', 'forty'), 'winters')
> ```

```python
# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence) # 使用 set 将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
```

```python
word_to_idx
```

> ```python
> {'old': 0,
>  'And': 1,
>  'child': 2,
>  'within': 3,
>  'where': 4,
>  'made': 5,
>  'sunken': 6,
>  'This': 7,
>  'thou': 8,
>  'treasure': 9,
>  'small': 10,
>  'dig': 11,
>  'gazed': 12,
>  'Thy': 13,
>  'weed': 14,
>  'held:': 15,
>  'answer': 16,
>  "deserv'd": 17,
>  'praise.': 18,
>  'brow,': 19,
>  'days;': 20,
>  'trenches': 21,
>  'blood': 22,
>  'to': 23,
>  'praise': 24,
>  "totter'd": 25,
>  'in': 26,
>  'say,': 27,
>  'beauty': 28,
>  'shall': 29,
>  'proud': 30,
>  'thy': 31,
>  'When': 32,
>  'see': 33,
>  'besiege': 34,
>  'thine': 35,
>  'being': 36,
>  'art': 37,
>  'lusty': 38,
>  'more': 39,
>  'lies,': 40,
>  "'This": 41,
>  'it': 42,
>  'Proving': 43,
>  'his': 44,
>  'To': 45,
>  'Shall': 46,
>  'be': 47,
>  'couldst': 48,
>  'succession': 49,
>  'all-eating': 50,
>  'fair': 51,
>  'sum': 52,
>  'count,': 53,
>  'when': 54,
>  'all': 55,
>  'new': 56,
>  'asked,': 57,
>  'Will': 58,
>  'a': 59,
>  'by': 60,
>  'forty': 61,
>  'my': 62,
>  'much': 63,
>  "youth's": 64,
>  "feel'st": 65,
>  'thriftless': 66,
>  'warm': 67,
>  'on': 68,
>  'use,': 69,
>  'so': 70,
>  'and': 71,
>  'worth': 72,
>  'the': 73,
>  'shame,': 74,
>  'winters': 75,
>  'thine!': 76,
>  'livery': 77,
>  'of': 78,
>  'Were': 79,
>  'own': 80,
>  'eyes,': 81,
>  'Where': 82,
>  'an': 83,
>  'now,': 84,
>  'mine': 85,
>  'field,': 86,
>  "excuse,'": 87,
>  'cold.': 88,
>  "beauty's": 89,
>  'Then': 90,
>  'If': 91,
>  'How': 92,
>  'old,': 93,
>  'make': 94,
>  'were': 95,
>  'deep': 96}
> ```

从上面可以看到每个词都对应一个数字，且这里的单词都各不相同

## 定义模型

接着我们定义模型，模型的输入就是前面的两个词，输出就是预测单词的概率

```python
# 定义模型
class n_gram(nn.Module):
    def __init__(self, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
        super(n_gram, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )
        
    def forward(self, x):
        voc_embed = self.embed(x) # 得到词嵌入
        voc_embed = voc_embed.view(1, -1) # 将两个词向量拼在一起
        out = self.classify(voc_embed)
        return out
```

最后我们输出就是条件概率，相当于是一个分类问题，我们可以使用交叉熵来方便地衡量误差

```python
net = n_gram(len(word_to_idx))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)
```

```python
for e in range(100):
    train_loss = 0
    for word, label in trigram: # 使用前 100 个作为训练集
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word])) # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # 前向传播
        out = net(word)
        loss = criterion(out, label)
        train_loss += loss.data
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(e + 1, train_loss / len(trigram)))
```

## 测试结果

最后我们可以测试一下结果

```python
net = net.eval()
```

```python
# 测试一下结果
word, label = trigram[19]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].item()
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))
```

> ```python
> input: ('so', 'gazed')
> label: on
> 
> real word is on, predicted word is on
> ```

```python
word, label = trigram[75]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].item()
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))
```

> ```python
> input: ("'This", 'fair')
> label: child
> 
> real word is child, predicted word is child
> ```

可以看到网络在训练集上基本能够预测准确，不过这里样本太少，特别容易过拟合。但是一定程度上也说明这个小模型能够处理N Gram模型的问题。除此之外，还有一种复杂一点的N Gram模型通过双边的单词来预测中间的单词，这种模型有个专门的名字，叫Continuous Bag-of-Words model(CBOW)，具体内容差别不大，就是不单单从单边，而是双边来预测。

