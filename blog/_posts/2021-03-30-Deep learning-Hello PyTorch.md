---
layout: post
title: Deep learning 강좌(5) - Hello PyTorch!
subtitle: Fast_Campus_처음부터 시작하는 딥러닝 유치원 Online
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





