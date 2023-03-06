---
layout: post
title: Data Structure (1-1) - Point Class 선언
subtitle: 파이썬 클래스 단계별 복습
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
tags: [Data_Structure, Python, Point_Class]
comments: true
---



# 01. Point Class 선언

### Point Class 선언

- Point 클래스는 2차원 평면의 점 (또는 2차원 벡터)을 나타내는 클래스이다
  - 필요한 멤버는 점의 x-좌표와 y-좌표이다
- 다음과 같이 선언할 수 있다

```python
class Point:
    def __init__(self, x=0, y=0):
        self.x = x    # x-좌표를 위한 멤버 self.x
        self.y = y    # y-좌표를 위한 멤버 self.y
```

- 생성 함수 (magic method 중 하나) __init__의 매개변수로 두 좌표 값을 받는다. default 좌표 값은 0으로 정했다. (다른 default 값으로 지정해도 상관없다)
- 또한 클래스의 모든 메쏘드의 첫 번째 매개변수는 이 메쏘드를 호출하는 객체를 나타내는 self 이어야 한다.

```python
  class Point:
  	def __init__(self, x=0, y=0):
  		self.x = x
  		self.y = y
  		
  p = Point(1, 2)
  print(p)
```

실행결과: <__main__.Point object at 0x7f652e778438>

왜 이런결과가 뜬것일까?

- print(p)를 수행하면 객체 p를 프린트해야 하는데, 어떤걸 출력해야 하는지 print는 알지도 못한다.
- 그래서 print에 무엇을 출력해야 할지 알려주는 함수가 ___str__ 함수이다

- __str__ 함수는 Point 객체의 출력하고 싶은 내용을 문자열로 만들어 리턴하기만 하면 된다

- - f"({self.x}, {self.y})" 형식의 f-문자열은 `{ }`안에 오는 변수를 변수의 값으로 교체해 문자열을 만드는 방법이다

```python
class Point:
  	def __init__(self, x=0, y=0):
  		self.x = x
  		self.y = y
        
  	def __str__(self):
   		return f"({self.x}, {self.y})"
    
  p = Point(1, 2)	# x = 1, y = 2인 객체 생성
  print(p)			# p.__str__()이 호출되고, 리턴된 "(1, 2)"를 출력함
```

실행결과: (1, 2)