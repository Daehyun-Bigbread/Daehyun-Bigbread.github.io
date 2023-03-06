# Ch 02. PyTorch Tutorial

## Part.5 실습 텐서 생성하기

- PyTorch Tensor

  ```python
  import torch
  ```



### Tensor Allocation

- 가장 많이 쓰이는 Tensor는 Float Tensor
  - 32bit의 Float Tensor = 말 그대로 실수(Float) 값을 담는 Tensor이다.

```python
# 2차원 (Matrix), torch.FloatTensor(): float의 값을 담는 tensor
ft = torch.FloatTensor([[1, 2],
                        [3, 4]])
ft
```



```
# 결과 값

tensor([[1., 2.],
        [3., 4.]])
```



- LongTensor = 정수값을 담는 Tensor, 주로 index(정수값) & ByteTensor (0 &1을 담는 Tensor)에 사용

```python
# Long : int (정수값)
lt = torch.LongTensor([[1, 2],
                       [3, 4]])
lt
```



```
# 결과 값

tensor([[1, 2],
        [3, 4]])
```



- ByteTensor

```python
# Bite : 0 or 1
bt = torch.ByteTensor([[1, 0],
                       [0, 1]])
bt
```



```
# 결과 값

tensor([[1, 0],
        [0, 1]], dtype=torch.uint8)
```



- Tensor Size만 원할때는 이렇게 사이즈만 지정해서 만드는 경우도 있음 (쓰레기 값 들어가도 상관 없을시)

  ```python
  x = torch.FloatTensor(3, 2)
  x
  ```

  

  ```
  # 결과 값
  tensor([[0.0000e+00, 4.6566e-10],
          [0.0000e+00, 4.6566e-10],
          [9.8091e-45, 0.0000e+00]])
  ```

  



### NumPy Compatibility

* NumPy =  Pytorch랑 호환이 잘되는게 장점이다.

```python
import numpy as np

# Define numpy array. -> PyTorch 선언 하는것과 같다.
# np -> numpy import 해서 torch 선언 하는것과 같음
x = np.array([[1, 2],
              [3, 4]])
print(x, type(x))
```



```
# 결과 값
[[1 2]
 [3 4]] <class 'numpy.ndarray'>
```



- Numpy를 torch로 가져오고 싶으면? 
  - torch.from_numpy(x) 함수를 이용하면 된다. 그러면 torch의 형태로 나온다.

```python
x = torch.from_numpy(x)
print(x, type(x))
```



```
# 결과 값
tensor([[1, 2],
        [3, 4]]) <class 'torch.Tensor'>
```



- torch.tensor를 numpy의 ndarray로 보내고 싶으면 x.numpy로 실행하면 return이 된다.

```python
x = x.numpy()
print(x, type(x))
```



```
# 결과 값
[[1 2]
 [3 4]] <class 'numpy.ndarray'>
```



### Tensor Type-casting

- ####  Float, Long이 서로 다르면 맞춰 줘야한다.

  - 무슨말이냐? Float Tensor는 Float Tensor 끼리, Long Tensor는 Long Tensor 끼리 값을 맞춰줘서 더해줘야 한다. 안맞추면? 오류나요 ㅎㅎ

​	

- Float Tensor를 Long Tensor로 봐꾸려면?

```python
ft.long()
```



```
# 결과 값. Long Tensor로 return 된다.
tensor([[1, 2],
        [3, 4]])
```



* Long Tensor를 FloatTensor로 봐꾸려면?
  * Float Tensor로 봐꾸듯이 반대로만 적어주면 된다.

```python
lt.float()
```



```
# 결과 값. Float Tensor로 return 된다.
tensor([[1., 2.],
        [3., 4.]])
```



* If. Byte Tensor로 봐꾸려면 Tensor 뒤에 'byte()' 만 적어주면 된다.

  ```python
  # FloatTensor 끼리 계산해야 한다.
  torch.FloatTensor([1, 0]).byte()
  ```

  

  ```
  # 결과 값
  tensor([1, 0], dtype=torch.uint8)
  ```



### Get Shape

* Tensor의 size를 알고 싶을때 쓰는 함수이다. 
  * Ex) |x| = (3,2,2) Tensor

```python
x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]]])
```



* x.size() 함수를 쓰면 튜플처럼 size를 return 해준다.

  * 또한 x.shape 함수를 써줘도 상관은 없음. 
    * 단, size처럼 뒤에 '( )' 붙이면 오류난다야... 함수가 안되기 때문. 제대로된 return 값이 안나온다.

  ```python
  print(x.size())
  print(x.shape)
  ```

  

  ```
  # 결과 값
  # |x| = (3,2,2) 
  
  torch.Size([3, 2, 2])
  torch.Size([3, 2, 2])
  
  ```

  

* 'x.dim()' 함수를 쓴다면 Tensor의 차원개수를 알수 있다. 

  * 'x.size()' 를 쓰면 list or tuple의 형식으로 나온다. 거기앞에 len을 쓰면 차원의 개수(Tensor의 모양)를 알수 있다.

```python
# dim: 차원
print(x.dim())
print(len(x.size()))
```



```
# 결과 값
3
3
```



* 특정 차원의 숫자를 알고 싶으면 'x.size()' 이렇게 함수를 써주면 된다.
  * x Tensor는 |x| = (3,2,2) 였으므로 (0,1,2) 순서. 그러면 1의 순서의 값인 2를 출력하는 것이다.
    * 근데 x.shape는 함수가 아니므로 뒤에 꼭 index를 적어줘야 한다.

```python
print(x.size(1))
print(x.shape[1])
```



```
# 결과 값
2
2
```



* -1를 쓰게 되면 마지막 순서 or 알아서 채우라는 뜻.
  * -1은 마지막 순서를 의미 함으로 2가 된다.

```python
# -1의 의미: 알아서 컴퓨터가 채우라는뜻
print(x.size(-1))
print(x.shape[-1])
```



```
# 결과 값
2
2
```

