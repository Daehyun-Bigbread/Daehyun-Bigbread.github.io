---
layout: post
title: Deep learning Studying(26) - 수식 Linear Regression
subtitle: Part.2 수식 Linear Regression
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---

# Ch 06. 선형희귀(Linear Regression)

## Part.2 수식 Linear Regression

#### Objective

* 데이터셋(![20210717_145059](../../assets/img/20210717_145059.png))이 주어졌을때, loss를 최소로 하는 파라미터(![20210715_233947](../../assets/img/20210715_233947.png))를 찾자.

![20210717_145409](../../assets/img/20210717_145409.png)



#### Loss Minimization using Gradient Descent

* Loss 함수를 파라미터(W, b)로 미분하여, 기울기 값을 활용해보자

![20210717_145545](../../assets/img/20210717_145545.png)



#### Loss Minimization using Gradient Descent - Detail

Loss 함수를 파라미터(![20210715_233947](../../assets/img/20210715_233947.png))로 미분하여, 기울기 값을 활용해보자

![20210717_145805](../../assets/img/20210717_145805.png)
