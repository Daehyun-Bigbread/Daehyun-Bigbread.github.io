---
layout: post
title: Data Structure (1-3) - Point Class Method 일반
subtitle: 일반
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
tags: [Data_Structure, Python, Point_Class]
comments: true
---

# 03. Point Class Method - 일반

### Point Class 메쏘드 - 일반

- 연산자 오버로딩을 통한 산술 연산을 구현하는 것 이외에 필요한 연산은 뭐가 있을까?
  
- 점 길이 (벡터의 길이), 내적, 외적, 두 점 사이의 거리, 점의 이동, 점의 회전 등 다양하다. 필요한 내용을 메쏘드로 구현하면 된다

- - 길이: length(p) = 점 (벡터) `p`의 길이 = ![img](https://hufs.goorm.io/texconverter?eq=%5Csqrt%7Bp.x%5E2%20%2B%20p.y%5E2%7D)
  - 내적: dot(p, q) = 두 벡터 p와 `q`의 내적 = ![img](https://hufs.goorm.io/texconverter?eq=p.x%20%5Ccdot%20q.x%20%2B%20p.y%5Ccdot%20q.y)
  - 거리: dist(p, q) = 두 점 `p`와 `q`의 길이 = ![img](https://hufs.goorm.io/texconverter?eq=%5Cmathrm%7Blength%7D(p-q))
  - 이동: move(p, dx, dy) = 점 p를 x-축으로 `dx` 만큼, y-축으로 `dy` 만큼 더해 이동

```python
import math

class Point:
  def __init__(self, x=0, y=0):
    self.x = x
    self.y = y
  
  def __str__(self):
    return f"({self.x}, {self.y})"

  def __sub__(self, other):
    return Point(self.x - other.x, self.y - other.y)

  def length(self):
    return math.sqrt(self.x**2 + self.y**2)
  
  def dot(self, q):
    return self.x*q.x + self.y*q.y

  def dist(self, q):
    return (self-q).length()

  def move(self, dx=0, dy=0):
    self.x += dx
    self.y += dy
​
p = Point(1, 2)
q = Point(2, 3)
print(f"p = {p}, q = {q}")
print("length of p =", p.length())
print("dot of p and q =", p.dot(q))
print("dist of p and q =", p.dist(q))
p.move(3, 5)
print("move p by (3, 5) =", p)
```



- 만약, 점의 좌표 값을 읽거나 변경하고 싶다면 어떻게 해야 할까?

- 가장 간단한 방법은 p.x, p.y 처럼 멤버 값을 **직접 참조**하는 것이지만, 객체지향언어의 원칙에 **위배**된다. 최대한 클래스 내부의 멤버 값을 직접 참조하지 않고 **정해진 메쏘드로만 참조**하도록 하는 게 좋다

- - `p.getX()`: p의 x-좌표 값을 리턴
  - `p.getY()`: p의 y-좌표 값을 리턴
  - `p.get()`:  p의 x-좌표 값과 y-좌표 값을 동시에 리턴
  - `p.setX(x)`: p.x = val
  - `p.setY(y)`: p.y = val
  - `p.set(x, y)`: p.x, p.y = x, y