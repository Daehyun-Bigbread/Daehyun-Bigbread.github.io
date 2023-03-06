---
layout: post
title: Deep learning Studying(11) - 텐서 자르고 붙이고~
subtitle: Part.7 실습 텐서 자르기&붙이기
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 02. PyTorch Tutorial

## Part.7 실습 텐서 자르기&붙이기

- PyTorch Tensor Manipulations

- In [1] : 

  ```python
  import torch
  ```



### Slicing and Concatenation

##### Indexing & Slicing



![KakaoTalk_20210709_160437176](../../assets/img/KakaoTalk_20210709_160437176.jpg)





##### split: Tensor를 바람직한 모양으로 분할 한다.

##### chunk: Tensor를 덩어리(chunks)의 개수만큼 나눈다.

##### Index_select: 차원 인덱스를 사용하여 특정 index를 선택한다.



![KakaoTalk_20210709_160512723](../../assets/img/KakaoTalk_20210709_160512723.jpg)



Cat: 리스트 안에 있는 여러개의 Tensor들을 모은다.

Stack: 리스트 안에 있는 여러개의 Tensor들을 쌓는다.

![KakaoTalk_20210709_160656146](../../assets/img/KakaoTalk_20210709_160656146.jpg)



활용 예시. Cat안에 Stack 기능을 구현 한다.

![KakaoTalk_20210709_160739820](../../assets/img/KakaoTalk_20210709_160739820.jpg)

