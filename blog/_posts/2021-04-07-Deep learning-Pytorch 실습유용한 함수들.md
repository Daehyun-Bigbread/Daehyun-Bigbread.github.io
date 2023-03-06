---
layout: post
title: Deep learning Studying(12) - 텐서 자르고 붙이고~
subtitle: Part.7 Pytorch 실습 유용한 함수들
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 02. PyTorch Tutorial

## Part.8 Pytorch 실습 유용한 함수들

```python
import torch
```



- #### Expand 함수: 주어진 Tensor를 복사해서 원하는 차원으로 만드는 함수


```python
x = torch.FloatTensor([[[1,2]],
                       [[3,4]]])
print(x.size)
```

torch.size([2, 1, 2])



```python
y = expand(*[2,3,2])

print(y)
print(y.size)
```

```결과
# 출력값
tensor([[[1., 2.],
         [1., 2.],
         [1., 2.]],

        [[3., 4.],
         [3., 4.],
         [3., 4.]]])
torch.Size([2, 3, 2])
```



- cat 함수를 사용하여 expand 함수 구현 
  - 여기서 cat 함수는 tensor를 합쳐주는 함수임

```python
y = torch.cat([x, x, x], dim=1)

print(y)
print(y.size())
```

```
# 출력값
tensor([[[1., 2.],
         [1., 2.],
         [1., 2.]],

        [[3., 4.],
         [3., 4.],
         [3., 4.]]])
torch.Size([2, 3, 2])
```



- #### randperm 함수 (randompermitation 함수)

  - 임의의 어떤 순열 (랜덤) 으로 만들어 내는 함수. like shuffling?

```python
x = torch.randperm(10)

print(x)
print(x.size)
```

```
# 출력값
tensor([8, 4, 0, 6, 3, 2, 1, 9, 5, 7])
torch.Size([10])
```



- #### argmax 함수 (argument max 함수)

  - 값의 최대를 만드는 index를 return 하는 함수 
    * ex) randperm 함수 이용해서 [3,3,3]의 Tensor를 만들어낸다.

```python
x = torch.randperm(3**3).reshape(3,3,-1) # [3,3,3]의 Tensor를 만들어낸다.

print(x)
print(x.size())
```

결과 = 27개의 random 한 permitation을 만들어낸다.

```
# 출력값
tensor([[[ 5, 25,  3],
         [ 6,  7, 15],
         [11,  8,  2]],

        [[10, 16, 12],
         [13,  0, 26],
         [24,  9, 17]],

        [[20,  1, 14],
         [19,  4, 18],
         [22, 23, 21]]])
torch.Size([3, 3, 3])
```



* 이제 argmax 함수를 이용한 Tensor 생성을 해보겠다. - 아직 이해 못함....

```python
y = x.argmax(dim=-1) # -1 dimension (z에서 가장큰 index를 골라야한다.)

print(y)
print(y.size())
```

```
# 출력값
tensor([[1, 2, 0],
        [1, 2, 0],
        [0, 0, 1]])
torch.Size([3, 3])
```

* argmax함수를 사용하면 가장 큰값이 있는 index를 찾을 수 있다. & index만 return
  * 내부적으로 sort가 들어감.



* #### topk 함수: value & index 같이 return 한다.

```python
values, indices = torch.topk(x, k=1, dim=-1)
# x = [3,3,3], k = 제일 높은 index를 뽑아낸다, dim= -1 dimension
# top-k 함수는 k개의 value & index return

print(values.size())
print(indices.size())
```

```
# 출력값
torch.Size([3, 3, 1])
torch.Size([3, 3, 1])
```



- index 값이 맞게 나왔는지 squeeze 함수를 돌려서 알아본다.

```python
print(values.squeeze(-1))
print(indices.squeeze(-1))
```

```
# 출력값
tensor([[25, 15, 11],
        [16, 26, 24],
        [20, 19, 23]])
tensor([[1, 2, 0],
        [1, 2, 0],
        [0, 0, 1]])
```

- 확인

```python
print(x.argmax(dim=-1) == indices.squeeze(-1))
```

```
# 출력값
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
```



* if. topk (top-k) 함수의 k값이 2라면?

```python
_, indices = torch.topk(x, k=2, dim=-1)
print(indices.size())

print(x.argmax(dim=-1) == indices[:, :, 0]) # index 자체를 access 했기에 차원은 날아감 -> [3,3]
```

```
# 출력값
torch.Size([3, 3, 2])
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
```



* top-k 함수로 sort 구현해보기

```python
target_dim = -1
values, indices = torch.topk(x,
                             k=x.size(target_dim), #k 값이 해당 dimesion의 size -> [3,3,3]이므로 3이다. k=3
                             largest=True) #큰값부터

print(values)
```

```
# 출력값 (큰값 순서대로 -> [3,3,k])
tensor([[[25,  5,  3],
         [15,  7,  6],
         [11,  8,  2]],

        [[16, 12, 10],
         [26, 13,  0],
         [24, 17,  9]],

        [[20, 14,  1],
         [19, 18,  4],
         [23, 22, 21]]])
```



* sort 함수로 top-k 함수 구현해보기

```python
k=1
values, indices = torch.sort(x, dim=-1, descending=True) # -1 (z차원), desending - 내림차순
values, indices = values[:, :, :k], indices[:, :, :k]

print(values.squeeze(-1))
print(indices.squeeze(-1))
```

```
# 출력값
tensor([[25, 15, 11],
        [16, 26, 24],
        [20, 19, 23]])
tensor([[1, 2, 0],
        [1, 2, 0],
        [0, 0, 1]])
```

결론 = top-k 함수로 sort 구현가능. 반대의 순서도 가능.



* #### masked_fill 함수 = masking이 된곳에 채워넣어라 ~

```python
x = torch.FloatTensor([i for i in range(3**2)]).reshape(3, -1)

print(x)
print(x.size())
```

```
# 출력값
tensor([[0., 1., 2.],
        [3., 4., 5.],
        [6., 7., 8.]])
torch.Size([3, 3])
```



* 여기서 4보다 큰 애는 True, 아니면 False

```python
mask = x > 4
print(mask)
```

```
# 출력값
tensor([[False, False, False],
        [False, False,  True],
        [ True,  True,  True]])
```

```python
y = x.masked_fill(mask, value=-1)
# -1 = False. 즉, -1인 값에 False를 채워넣어라

print(y)
```

```
# 출력값
tensor([[ 0.,  1.,  2.],
        [ 3.,  4., -1.],
        [-1., -1., -1.]])
```



* #### Ones & Zeros 함수: 0 or 1 Tensor만들때 쓴다.

```python
print(torch.ones(2, 3))
print(torch.zeros(2, 3))
```

```
# 출력값
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

```python
x = torch.FloatTensor([[1, 2, 3],
                       [4, 5, 6]])
print(x.size())
```

```
torch.Size([2, 3])
```



* Ones를 만들때 GPU & 같은 TYPE & DEVICE에 만들고 싶은데, X에 연산을 해야함. 그러면 여기에 맞는 연산을 하고 싶을때 쓴다. 
  * Type & Device 서로 같게 해준다. 만약 once로 돌리면 cpu에서 실행해서 cpu & gpu 같이 돌리면 죽는다... 귀차니즘 들림. 그래서 once_like 돌리면 size와 device & type 서로 같은 torch로 만든다. 

```python
print(torch.ones_like(x))
print(torch.zeros_like(x))
```

```
# 출력값
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```
