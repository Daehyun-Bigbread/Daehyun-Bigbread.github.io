---
layout: post
title: Deep learning Studying(5) - PyTorch 사용법
subtitle: Part.2 Hello PyTorch!
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 02. PyTorch Tutorial

## Part.2 Hello PyTorch!

- import PyTorch

```python
import torch
```



- PyTorch의 float tensor 들을 정의해준다. 

  ```python
  a = torch.FloatTensor([[1, 2],
                         [3, 4]])
  b = torch.FloatTensor([[1, 2],
                         [1, 2]])
  ```

  

- float Tensor의 행렬 곱셈 예제이다.

```python
c = torch.matmul(a, b)
c
```

결과값 : 

```python
tensor([[ 3.,  6.],
        [ 7., 14.]])
```





