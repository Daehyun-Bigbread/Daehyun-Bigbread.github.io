---
layout: post
title: Deep learning Studying(12) - Matrix Multiplication
subtitle: Part.1 행렬곱
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 03. 신경망의 기본 구성요소 살펴보기

## Part.1 행렬곱

#### Matrix Multiplication

* 행렬 곱
* Inner Product, Dot Product

![20210710_164254](../../assets/img/20210710_164254.png)



#### Vector Matrix Multiplication 

* 벡터와 행렬의 곱셈

![20210710_163722](../../assets/img/20210710_163722.png)



![20210710_163747](../../assets/img/20210710_163747.png)



#### Batch Matrix Multiplication (BMM)

* 같은 갯수의 행렬 쌍들에 대해서 **병렬**로 행렬 곱 실행
  * torch.bmm: 여기서 torchs는 PyTorch를 의미한다.
  * x * y = z
    * x = (N1 * N2, n, h), y = (N1 * N2, h, m) 

![20210710_163929](../../assets/img/20210710_163929.png)

