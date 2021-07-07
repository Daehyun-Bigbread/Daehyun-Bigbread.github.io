---
layout: post
title: Deep learning 강좌(8) - 실습 텐서 생성하기
subtitle: Fast_Campus_처음부터 시작하는 딥러닝 유치원 Online
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 02. PyTorch Tutorial

## Part.5 실습 텐서 생성하기

- PyTorch Tensor

- In [1] : 

  ```python
  import torch
  ```

#### Tensor Allocation

- In [2] : 

```python
# 2차원 (Matrix), torch.FloatTensor(): float의 값을 담는 tensor
ft = torch.FloatTensor([[1, 2],
                        [3, 4]])
ft
```

- Out [2] : 

  ```python
  tensor([[1., 2.],
          [3., 4.]])
  ```

- In [3] : 

```python
# Long : int
lt = torch.LongTensor([[1, 2],
                       [3, 4]])
lt
```

- Out [3] : 

```python
tensor([[1, 2],
        [3, 4]])
```

- In [4] : 

```python
# Bite : 0 or 1
bt = torch.ByteTensor([[1, 0],
                       [0, 1]])
bt
```

- Out [4] : 

```python
tensor([[1, 0],
        [0, 1]], dtype=torch.uint8)
```

- In [5] : 

  ```python
  x = torch.FloatTensor(3, 2)
  x
  ```

- Out [5] : 

  ```python
  tensor([[0.0000e+00, 4.6566e-10],
          [0.0000e+00, 4.6566e-10],
          [9.8091e-45, 0.0000e+00]])
  ```

  ### NumPy Compatibility

- In [6] : 

```python
import numpy as np

# Define numpy array. -> PyTorch 선언 하는것과 같다.
x = np.array([[1, 2],
              [3, 4]])
print(x, type(x))
```

출력 결과

```python
[[1 2]
 [3 4]] <class 'numpy.ndarray'>
```



- In [7] : 

```python
x = torch.from_numpy(x)
print(x, type(x))
```

출력 결과

```python
tensor([[1, 2],
        [3, 4]]) <class 'torch.Tensor'>
```



- In [8] :

```python
x = x.numpy()
print(x, type(x))
```

출력 결과

```python
[[1 2]
 [3 4]] <class 'numpy.ndarray'>
```



### Tensor Type-casting

#### - Float, Long이 서로 다르면 맞춰 줘야한다.

- In [9] : 

```python
ft.long()
```

* Out [9] : 

```python
tensor([[1, 2],
        [3, 4]])
```



* In [10] : 

```python
lt.float()
```

- Out [10] :  

```python
tensor([[1., 2.],
        [3., 4.]])
```



* In [11] : 

  ```python
  # FloatTensor 끼리 계산해야 한다.
  torch.FloatTensor([1, 0]).byte()
  ```

* Out [11] :

  ```python
  tensor([1, 0], dtype=torch.uint8)
  ```



### Get Shape

* In [12] : 

```python
x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]]])
```

Get Tensor Shape.

* In [13] :

  ```python
  print(x.size())
  print(x.shape)
  ```

  

  ```python
  # |x| = (3,3,2) 
  
  torch.Size([3, 2, 2])
  torch.Size([3, 2, 2])
  ```

  Get number of dimensions in the tensor.

* In [14] : 

```python
# dim: 차원
print(x.dim())
print(len(x.size()))
```



```
3
3
```

Get number of elements in certain dimension of the tensor.

* In [15] : 

```python
print(x.size(1))
print(x.shape[1])
```



```
2
2
```

Get number of elements in the last dimension.

* In [16] : 

```python
# -1의 의미: 알아서 컴퓨터가 채우라는뜻
print(x.size(-1))
print(x.shape[-1])
```



```
2
2
```

