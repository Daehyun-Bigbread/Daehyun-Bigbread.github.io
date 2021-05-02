---
layout: post
title: Data Structure (1-4) - Point 클래스를 Vector 클래스로 확장
subtitle: Vector 클래스 설계
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
tags: [Data_Structure, Python, Point_Class]
comments: true
---

# 04. Point 클래스를 Vector 클래스로 확장

### Vector 클래스 설계

- Point 클래스는 2차원의 점 또는 2차원 벡터만을 위한 클래스이므로, 3차원, 4차원 등의 여러 차원 벡터를 자유롭게 표현할 수 있는 Vector 클래스가 필요하다
  
- - __init__에 전달하는 초기 좌표 값의 개수는 벡터의 차원에 따라 결정되므로 매개 변수의 개수를 마음대로 지정할 수 있는 기능을 사용한다

  - - def function(*args): 라고 함수를 정의하면, 임의의 개수의 매개 변수들을 tuple args를 통해 전달할 수 있다

```
def __init__(self, *args):
    self._coords = list(args) # 좌표 값을 리스트 _coords에 저장
```

- - _coords 처럼 _ 하나로 시작하는 멤버 변수는 nonpublic 변수로 만약 이 클래스를 다른 곳에서 import해 사용하는 경우에는 nonpublic 멤버는 import 되지 않고 감춰진다. 좌표 값을 참조하기 위해선, magic 메쏘드 __getitem__과 __setitem__을 사용한다 (뒤에서 설명)

- 벡터 출력이 필요하므로 __str__ 메쏘드를 정의한다

```
def __str__(self):
    return str(tuple(self._coords))    # tuple로 바꿔주는 이유는 단순히 (x, y, z) 형식으로 출력하기 위해서 
```

- __len__ magic 메쏘드를 정의한다. 이 메쏘드는 벡터의 차원을 리턴하는 것으로 len(v)의 형태로 사용할 수 있게 하는 메쏘드이다

```
def __len__(self):
    return len(self._coords)
```

- __getitem__(self, k) 는 self의 k번째 값을 리턴하는 magic 메쏘드이다. 여기서는 k번째 좌표 값 _coords[k]를 리턴한다
- __setitem__(self, k, val) 는 self의 k번째 값에 val 값을 대입하는 magic 메쏘드이다. 즉, _coords[k] = val이 된다

```
def __getitem__(self, k):
    return self._coords[k]
def __setitem__(self, k, val):
    self._coords[k] = val
```

- - 이 두 메쏘드가 정의되어 있다면, 벡터 v에 대해서, v[1] = v[0] + 3 처럼 **인덱스**를 통해 좌표 값을 읽거나 변경할 수 있게 된다

- __len__ 과 __getitem__ 메쏘드가 정의되면 해당 클래스에 대한 iterator가 자동으로 정의된다. iterator는 for 루프에서 각 원소를 차례대로 접근하는 데 사용한다. 다음의 코드의 출력 결과는 `1 2 3`이다

```
v = Vector(1, 2, 3)  # (1, 2, 3) 3차원 벡터 v 생성
print(v)        # (1, 2, 3)
for c in v:    # for 루프를 돌면서 c = v[0], v[1], v[2]이 됨
    print(c, end=" ")
print()        # 1 2 3
```

- - 클래스의 iterator를 지정하는 다른 방법 (__iter__를 정의하거나 yield를 사용한 generator를 정의하는 방법 등)도 있다 

- Point 클래스처럼 백터 덧셈, 뺄셈, 곱셈, 길이, 내적, 외적 등 필요한 연산을 구현할 수 있다