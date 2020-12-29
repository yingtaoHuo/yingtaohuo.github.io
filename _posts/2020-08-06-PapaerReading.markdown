---
layout: post
title:  NLP_2020 Sentiment Analysis
date:   2020-08-07 22:42:20 +0300
description: write some words # Add post description (optional)
img: playground.jpg # Add image post (optional)
tags: [NLP-Sentiment]
author: Richard Huo # Add name author (optional)
comments: true
---
1. 本文分析各个文本预处理过程在情感分析任务中的作用。 *
   *A Comprehensive Analysis of Preprocessing for Word Representation Learning in Affective Tasks*
2. 本文贡献了一个中文多模态数据集，创新点在于数据标注的方式（one mutlimodal annotation and three unimodal annotations for each video clip）。并且进行了一系列的实验，发布了该数据集的baseline。*
   *CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality*
3. 本文提出一种新颖的预处理办法：Sentiment Knowledge Enhanced Pre-training，为了解决目前sentiment words和aspect-sentiment pairs在预处理中被忽视的问题。*
   *SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis*
4. 本文提出cross domain的模型，主要为了解决labeled数据不足的问题。*
   *Adversarial and Domain-Aware BERT for Cross-Domain Sentiment Analysis*
5. 本文提出了一套无监督，跨语种，情感分类模型，命名为multi-view encoder-classifier(MVEC)，解决部分语种缺少标注数据和大型语料库的问题。
   *Cross-Lingual Unsupervised Sentiment Classification with Multi-View Transfer Learning*
6. 关于Aspect sentiment analysis问题，本文提出了过去的研究者们专注于sentence-level sentiment analysis，忽略了document-level的信息，并举出明显案例佐证观点。以下是对于Aspect sentiment analysis的解释。*
   Aspect sentiment analysis: Aspect-based sentiment analysis is a text analysis technique that breaks down text into aspects (attributes or components of a product or service), and then allocates each one a sentiment level (positive, negative or neutral).
   *Aspect Sentiment Classification with Document-level Sentiment Preference Modeling*
7. 本文提出此前的aspect情感研究存在一个问题：人们通过attention机制，强调opinion word与aspect的关系，但是因为语言间复杂的关系，模型往往会confuse这些关系。本文提出通过构建有效的语法树来解决这一问题。*
   *Relational Graph Attention Network for Aspect-based Sentiment Analysis*
8. 本文认为情感分析，情绪分析的结果对于嘲讽检测有正面意义，提出一个多任务框架，使用环境为多模态的场景。这个研究证明，情感检测得到的结果可以提高嘲讽检测模型的效果。
   *Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis*
9.  本文首先将ABSA问题，拆解为三个子问题的集合，提出三个子问题间的交互关系尚未被挖掘这一事实，于是提出想通过构建aspect与opinion的关系对三个子问题的协作信号进行编码（这句话过于拗口，简单点就是，加了点先验知识，比如‘美味’一般情况下不会对应‘地板’，大概率会对应‘食物’类别下的名词）。*
    *Relation-Aware Collaborative Learning for Unified Aspect-Based Sentiment Analysis*
10. 本文尝试拓展bert模型的使用方法，提高类bert模型在SST数据集上，phrase-level sentiment analysis的成绩。通过构建一棵二元树，总结一段phrase的上下文信息，实验证明在phrase-level sentiment classification获得较好效果。后面还有实验可以证明该模型的迁移效果较好。最后设计了可视化方法去展示这一模型的优势。 *
    *SentiBERT: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics*
11. 本文尝试提高domain adversial模型，在cross domain（combat domain gap between different applications）问题上的表现，通过使用图卷积自动编码器获得domain域的信息来达到目的。 *
    *KinGDOM: Knowledge-Guided DOMain adaptation for sentiment analysis*