---
layout: post
title: Deep learning Studying(56) - Softmax & Cross Entropy
subtitle: Part.5 Softmax & Cross Entropy
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---

# Ch 12. 딥러닝 입문 (분류)

### Part.5 Softmax & Cross Entropy

#### Softmax

* 입력 벡터를 discrete(이산) 확률 분포 형태로 봐꿔주는 함수
  * 확률 분포이므로 각 클래스 별 확률 값들의 합은 1이 됨.

![20210801_182307](../../assets/img/20210801_182307.png)



![20210801_182441](../../assets/img/20210801_182441.png)



#### Cross Entropy Loss

* Binary Cross Entropy의 일반화 version

![20210801_182509](../../assets/img/20210801_182509.png)



#### NLL Loss with Log-Softmax

* Log-Softmax

![20210801_184957](../../assets/img/20210801_184957.png)

* Negative Log Likeihood (NLL Loss)

![20210801_185006](../../assets/img/20210801_185006.png)



#### Summary

* Regression task의 MSE loss와 마찬가지로, Classification task에서 Cross Entropy loss를 최소화 하면 분류 문제를 위한 모델을 학습할 수 있다.

![20210801_182730](../../assets/img/20210801_182730.png)



* 이를 위해서 신경망은 softmax 함수를 통해 각 클래스 별 확률 값을 반환한다.

![20210801_182739](../../assets/img/20210801_182739.png)

 

* 우리는 학습이 완료된 신경망을 통해, 입력 x가 주어졌을때, 가장 큰 확률 값을 갖는 ![20210720_144940](../../assets/img/20210720_144940.png)의 index를 알 수 있고, x의 클래스 c를 추측할 수 있다.
