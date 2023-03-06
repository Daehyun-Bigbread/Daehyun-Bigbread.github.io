---
layout: post
title: Deep learning Studying(10) - 텐서 형태 봐꾸어보기
subtitle: Part.7 실습 텐서 형태 변환
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 02. PyTorch Tutorial

## Part.7 실습 텐서 형태 변환

- PyTorch Tensor Manipulations

- In [1] : 

  ```python
  import torch
  ```



### Tensor Shaping

##### reshape: Change Tensor Shape

![KakaoTalk_20210709_125831616](../../assets/img/KakaoTalk_20210709_125831616.jpg)

#### Squeeze: Remove dimension which has only one element (Dimension 삭제)

* if. 1이 존재하는 Dimension의 element 개수가 1인 경우? 
  * 해당 Dimension은 삭제된다.

#### UnSqueeze: Remove dimension which has only one element (Dimension 삽입)

![KakaoTalk_20210709_141930422](../../assets/img/KakaoTalk_20210709_141930422.jpg)
