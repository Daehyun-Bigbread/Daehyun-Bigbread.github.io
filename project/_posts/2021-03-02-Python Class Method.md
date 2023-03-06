---
layout: post
title: Data Structure (1-2) - Point Class Method
subtitle: 연산자 오버로딩
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
tags: [Data_Structure, Python, Point_Class]
comments: true
---

# 02. Point Class Method - 연산자 오버로딩

### Point Class 메쏘드 (연산자 오버로딩 포함)

- 메쏘드는 magic method:  __init__, __str__, __len__ 이런 magic method와 일반 method로 구분할수 있다.
  - Magic 메쏘드의 이름은 두 개의 underscore로 메쏘드 이름을 감싼 형태.
- Magic method의 몇 가지 예를 들어보도록 하겠다.
  - point 객체는 점을 나타내지만 벡터로 보일수도 있기 때문에 두점을 더하는 연산 진행시 각 좌표에 새로운 점을 생성해서 return.

```python
def add(self, other):
    return Point(self.x + other.x, self.y + other.y)
```

```python
p = Point(1, 2)
q = Point(3, 4)
r = p.add(q)
print(r)    # r = (4, 6)
```

근데, 일반적인 덧셈처럼 a = b + c 의 형식으로 + 연산자를 사용할수 있다면. 이해하기 쉽다. 그래서 파이썬에선 이 + 를 위한 magic Method를 제공한다. 정,실수의 덧셈 + 연산자에 Point 클래스의 덧셈 연산을 덧 입혔다는 의미로 이 기능을 "연산자 overloading" 이라고 한다.

```python
def __add__(self, other):
    return Point(self.x + other.x, self.y + other.y)
```

```python
a = b + c
print(a)     # a = (4, 6)
```

- 특별 메쏘드의 이름은 __add__ 이다. a = b + c를 하면, 실제로는 a = p.__add__(a)가 호출되어, 두 벡터의 합 벡터가 리턴되어 a 에 저장된다.

```python
class Point:
  def __init__(self, x=0, y=0):
    self.x = x
    self.y = y

  def __str__(self):
    return f"({self.x}, {self.y})"

  def __add__(self, other):
    return Point(self.x + other.x, self.y + other.y)

b = Point(1, 2)
c = Point(3, 4)
a = b + c
d = p.__add__(b)
print(a, d)
```



#### 산술연산자의 종류

- __add__, __sub__, __mul__, __truediv__, __floordiv__, __mod__ 등이 있고 각각 **+, -, \*, /, //, %**에 대응된다. 

- __iadd__, __isub__, __imul__, __itruediv__, __ifloordiv__, __imod__ 등은 각각 **+=, -=, \*=, /=, //=, %=**에 대응된다.

- 다른 산술 연산자 오버로딩를 위한 매직 메쏘드도 있다

- 비교 연산자 오버로딩도 가능하다

- - __lt__, __le__, __gt__, __ge__, __eq__, __ne__ 등은 각각 **<, <=, >, >=, ==, !=**에 대응된다

- 연산자 오버로딩 기능을 Point 클래스에 뺄셈 연산자 오버로딩을 위해 __sub__ 메쏘드를 작성하면

- - r = p - q 형식으로 사용하면 되고, 벡터의 뺄셈처럼 대응되는 좌표 값을 빼주면 된다.

```python
def __sub__(self, other):
    return Point(self.x - other.x, self.y - other.y)
```

- 이제, 곱셈 연산을 생각해보자. 두 벡터의 곱은 덧셈이나 뺄셈처럼 대응되는 좌표 값을 더하거나 빼는 식으로 정의되지 않는다. 벡테의 곱셈 연산은 r = 3 * p의 형태처럼 p의 좌표 값에 모두 상수 3을 곱하는 식으로 사용된다. 즉, **scalar \* vector** 형식으로 사용된다

- scalar 값은 Point 객체가 아니기 때문에 연산에 참여하는 두 객체의 타입이 같지 않다는 문제가 발생한다. 파이썬에서는 이러한 경우에도 연산자 오버로딩 기능을 지원한다. 단, 오른쪽 객체를 기준으로 오버로딩을 해야 한다

- __rmul__ (right multiplication) magic 메쏘드는 `*` 연산자의 오른쪽에 등장하는 객체가 `self`가 되고 왼쪽 객체가 `other`가 되는 형식이다. 그래서 이름에 r이 붙었다. 이 경우에 `self`와 `other`의 타입이 달라도 된다

- - r = 3 * p   # r = p.__rmul__(3)의 형식으로 호출됨 (반대가 아님에 주의!)

```python
def __rmul__(self, other):
    return Point(self.x * other, self.y * other)
```

```python
class Point:
  def __init__(self, x=0, y=0):
    self.x = x
    self.y = y
  
  def __str__(self):
    return f"({self.x}, {self.y})"
  
  def __add__(self, other):
    return Point(self.x + other.x, self.y + other.y)
  
  def __sub__(self, other):
    return Point(self.x - other.x, self.y - other.y)
  
  def __rmul__(self, other):
    return Point(self.x * other, self.y * other)

p = Point(1, 2)
q = Point(3, 4)
r = p - q
print(r)
r = 3 * p
print(r)
r = p * 3   # 이 문장은 에러를 발생시킨다!
print(r)
```

