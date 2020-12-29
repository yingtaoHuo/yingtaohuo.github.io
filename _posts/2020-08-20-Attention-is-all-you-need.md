---
layout: post
title:  Attention is all you need
date:   2020-08-26 10:56:20 +0300
description: write some words # Add post description (optional)
img: post-2.jpg # Add image post (optional)
tags: [NLP-Basic]
author: Richard Huo # Add name author (optional)
comments: true
---
### Component Introduction

#### Input & Output Embedding
1. wordEmbedding：基于nn.Embedding实现，或者用one-hot vector与权重矩阵W相乘获得。nn.Embedding有两种权重矩阵可以选择。
- 使用pretrained的embeddings固化，可以用glove或者word2vec模型
- 初始化W权重矩阵，设置为trainable，在迭代的过程中进行训练

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) 
```
1. positionalEmbedding：体现词在句子的位置信息。在NMT任务中，之前的encoder模型是基于LSTM的，句子中的词按顺序逐步计算。但是在Transformer中，句子中的词同时处理，无法体现词在句子中的位置。比如：“我爱她”和“她爱我”，截然不同的意思。
   positionEmbeding也有两个获得方式。
- 使用根据公式计算好的的embeddings
- 初始化W权重矩阵，设置为trainable，在迭代的过程中进行训练
在后来的实验中，两种方法的效果相似。考虑到第一种方式不需要更新参数，也可以应对训练集中没有出现过的句子长度，故采用第一种方法。
计算positionEmbedding的公式为：


> $ PE_{(pos, 2i)}=\sin\left(pos / 10000^{2 i / d_{\text {modd }}}\right)$

> $ PE_{(pos, 2i+1)}=\cos\left(pos / 10000^{2 i / d_{\text {modd }}}\right)$


```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)
```
#### Encoder
1. encoder由六个结构相同，参数不同的的encodeBlock构成，每个block包含两个sub-layer，sub-layer分别是multi-head-attention mechanism和feed-forward network。每个sub-layer都加入了short-cut connection和normalisation。
   multi-head-attention结构如代码所示
```python
class multiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, headNum, dropout=0.1):
        super.__init__()
        assert d_model == d_feature * headNum
        self.d_model = d_model
        self.d_feature = d_feature
        self.headNum = headNum
        self.multiHeadAttention = nn.ModuleList([
            attentionHead(d_model, d_feature, 0.1) for _ in range(self.headNum)
        ])
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        x = [attn(query, key, value,mask=mask) for i,attn in enumerate(self.multiHeadAttention)]
        x = torch.cat(x, dim=-1)
        x = self.projection(x)
        return x
```
其中的attentionHead如下所示：
```python
class attentionHead(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        self.attn = scaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)

    def forward(self, query, key, value, mask=None):
        Q = self.query_tfm(query)
        K = self.key_tfm(key)
        V = self.value_tfm(value)
        x = self.attn(Q, K, V)
        return x
```
scaledDotProductAttention如下所示，与之前的dot attention不同的是加了scaled，作者认为，当d_k较大时，softmax后整个matrix的值都会偏小，通过scaled来扩大数值差异：

> $ \text{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right)V$


```python
class scaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout=0.1)

    def forward(self, query, key, value, mask=None, dropout=0.1):
        d_k = key.size(-1)
        scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, 0)
        p_attn = F.softmax(scores,dim=-1)
        if dropout is None:
            p_attn = self.dropout(p_attn)
        output = torch.matmul(p_attn, value)
        return output
```
feed-forward模块则包含在encode中：
```python
class encoderBlock(nn.Module):
    def __init__(self,d_model, d_feature, d_ff, headNum, dropout=0.1):
        super().__init__()
        self.headAttention = multiHeadAttention(d_model, d_feature, headNum, dropout)
        self.normLayer1 = normLayer(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Relu(),
            nn.Linear(d_ff, d_model),
        )
        self.normLayer2 = normLayer(d_model)

    def forward(self, x, mask=None):
        atten = self.headAttention(x,x,x,mask=None)
        x = x + self.dropout(self.normLayer1(atten))
        pos = self.position_wise_feed_forward(x)
        x = pos + self.dropout(self.normLayer2(pos))
        return x
```
将多个encoderBlock连在一块就是整个encoder：
```python
class encoder(nn.Module):
    def __init__(self, encodeNum, d_model, d_feature, d_ff, dropout=0.1):
        super().__init__()
        self.encoderSeq = nn.Sequential([
            encoderBlock(d_model, d_feature, d_ff, d_model//d_feature, dropout) for _ in range(encodeNum)
        ])

    def forward(self,x):
        for encoder in self.encoderSeq:
            x = encoder(x)
        return x 
```
#### decoder
相较于encoder，多了层maskedMultiHeadAttention，为了遮挡当前预测词及后面的词语。计算方式不同的是，encoder并行计算了一整个句子，decoder逐词进行推算，因为第i个词的输入x来自于第i-1个词的decoder输出。其他组件与encoder相同。
```python
class decoderBloack(nn.Mmodule):
    def __init__(self, d_model, d_feature, d_ff, headNum,dropout=0.2):
        super().__init__()
        self.maskedHeadAtten = multiHeadAttention(d_model,d_feature,headNum,dropout=0.1)
        self.headAtten = multiHeadAttention(d_model,d_feature,headNum,dropout=0.1)
        self.position_wise_feed_forward = nn.Sequential{
            nn.Linear(d_model, d_ff),
            nn.Relu(),
            nn.Linear(d_ff, d_model),
        }
        self.normLayer1 = normLayer(d_model)
        self.normLayer2 = normLayer(d_model)
        self.normLayer3 = normLayer(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        att = self.maskedHeadAtten(x,x,x,mask=src_mask)
        x = x + self.dropout(self.normLayer1(att))
        att = self.maskedHeadAtten(x,enc_out,enc_out,mask=tgt_mask)
        x = x + self.dropout(self.normLayer2(att))
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.normLayer3(pos))
        return x

class transformerDecoder(nn.Module):
    def __init__(self, blockNum, d_model, d_feature, d_ff, headNum, dropout=0.1):
        super().__init__()
        self.decoders = nn.ModuleList([
            decoderBloack(d_model, d_feature, d_ff,headNum, dropout=dropout) for _ in range(blockNum)
        ])
    
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x
```

#### teacherForcing
在训练过程中，如果预测错一个，那么下一个decoderBloack的输出就会受到影响，大概率就会越走越歪。为了解决这一问题，人们提出了teacher forcing这一训练trick。每个decoderblock的输出x，由它前一个词的正确预测给出。

