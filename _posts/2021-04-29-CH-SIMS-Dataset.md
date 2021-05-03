---
layout: post
title:  CH-SIMS-Dataset
date:   2021-04-29 10:56:20 +0300
description: write some words # Add post description (optional)
img: post-2.jpg # Add image post (optional)
tags: [MultiModal]
author: Richard Huo # Add name author (optional)
comments: true
---

## Motivation
1. Unified multimodal annotations can not reflect the independent sentiment of single modalities, and limit the model to capture difference between modalities.
2. Chinese single- and multi-modal sentiment analysis dataset, which contains 2281 refined video segments/clips in wild with both multimodal and independent unimodal annotations.
3. They propose a multi-task learning framework based on late fusion as the baseline.
4. Source code provided by author are available(https://github.com/thuiar/MMSA/)

## Related Works

### Multimodal Datasets
IEMOCAP()
YouTube()
MOUD()
ICT-MMMO()
MOSI()
CMU-MOSEI()

### Multimodal Sentiment Analysis
1. General framework proposed(https://arxiv.org/pdf/1707.09538.pdf) which is composed of representation learning on intra-modality and feature concatenation on inter-modality.
2. (https://arxiv.org/pdf/1806.06176.pdf)Tasi try to factorize representations into two sets of independent factors: multimodal discriminative and modality-specific generative factors. 

### Multi-task Learning
1. The aim of multi-task learning is to improve the generalization performance of multiple related tasks by utlizing useful information contained in these tasks. The classical framework is that different tasks share the first several layers and then have task-specific parameters in the subsequent layers. In this task, multimodal multi-task learning framework is applied for the verification and feasibility of independent unimodal annotations.

#### Problems
Negative segments are more than positive segments. 

## Extracted Features
#### Text
1. add two unique tokens to indicate the beginning and the end for each transcript.
2. pre-trained Chinese BERTbase word embeddings are used to obtain word vectors from transcripts. It is worth noting that they do not use word segmentation tools due to the characteristic of BERT.

#### Audio
1. use LibROSA speech toolkit with default parameters to extract acoustic features at 22050Hz.
2. Totally, 33-dimensional frame-level acoustic features are extracted, including 1-dimensional logarithmic fundamental frequency (log F0), 20-dimensional Melfrequency cepstral coefficients (MFCCs) and 12-dimensional Constant-Q chromatogram (CQT).These features are related to emotions and tone of speech according to ([Li et al., 2018](https://hcsi.cs.tsinghua.edu.cn/Paper/Paper18/MM-LIRUNNAN.pdf)).

#### Vision
1. They employ the MTCNN face detection algorithm ([Zhang et al., 2016a](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)) to extract aligned faces.
2. Then they use MultiComp Openface toolkit to extract the set of 68 facial landmarks, 17 facial action units, head pose, head orientation, and eye gaze. Lastly, 709-dimensional frame-level visual features are extracted in total.

#### For Intra-Modal Representation
1. For the convenience in text, audio and vision, we assume that:
$$L^{u}：Sequence Length$$
$$ D_{i}^{u}：Initial Feature$$
$$ D_{r}^{u}：RepresentationFeatures$$
> where u ∈ {t, a, v}, represent the sequence length,
initial feature is extracted by section 3.3
and representation feature learned by unimodal
feature extractor, respectively. The batch size is B.
2. 计算公式
$$R_{u}=S_{u}\left(I_{u}\right) $$
1. They use a Long Short-Term Memory (LSTM)
network, a deep neural network with three hidden
layers of weights Wa and a deep neural network
with three hidden layers of weights Wv to extract
textual, acoustic and visual embeddings, respectively.


#### For Inter-Modal Representation
They try three fusion methods: LF-DNN, TFN and LMF

#### Multimodal Multitask Learning Framework 
1. Objective Function
$$ \min \frac{1}{N_{t}} \sum_{n=1}^{N_{t}} \sum_{i} \alpha_{i} L\left(y_{i}^{n}, \hat{y}_{i}^{n}\right)+\sum_{j} \beta_{j}\left\|W_{j}\right\|_{2}^{2}$$
> Lastly, we use a three-layer DNN to generate
outputs of different tasks. In this work, we treat
these tasks as regression models.