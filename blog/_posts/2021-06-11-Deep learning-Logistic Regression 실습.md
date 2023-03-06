---
layout: post
title: Deep learning Studying(32) - Logistic Regression 실습
subtitle: Part.6 Logistic Regression 실습
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---

# Ch 07. 로지스틱 희귀(Logistic Regression)

## Part.6 Logistic Regression 실습

#### Logistic Regression

#### Load Dataset from sklearn

In [1] :

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

In [2] :

```python
from sklearn.datasets import load_breast_cancer
cancer = load.breast_cancer()

print(cancer.DESCR)
```



![20210721_163800](../../assets/img/20210721_163800.png)



![20210721_160823](../../assets/img/20210721_160823.png)



![KakaoTalk_20210721_165007283](../../assets/img/KakaoTalk_20210721_165007283.jpg)



In [3] : 

```python
df = pd.DataFrame(cancer.data, colums=cancer.feature_names)
df['class'] = cancer.target

df.tail()
```



Out [3] :

![20210721_165422](../../assets/img/20210721_165422.png)





#### Pair Plot with mean features 

In [4] :

```python
sns.pairplot(df[['class'] + list(df.columns[:10])])
plt.show()
```

![20210721_172142](../../assets/img/20210721_172142.png)



#### Pair Plot with std features 

In [5] :

```python
# 아래로 내려 갈수록 값은 0에 가까워지고, 위로 올라갈수록 1에 가까워진다.
sns.pairplot(df[['class'] + list(df.columns[10:20])])
plt.show()
```

![20210721_165819](../../assets/img/20210721_165819.png)



#### Pair plot with worst features

In [6] :

```python
sns.pairplot(df[['class'] + list(df.columns[20:30])])
plt.show()
```

![20210721_170616](../../assets/img/20210721_170616.png)



#### Select features

In [7] :

```python
cols = ["mean radius", "mean texture",
        "mean smoothness", "mean compactness", "mean concave points",
        "worst radius", "worst texture",
        "worst smoothness", "worst compactness", "worst concave points",
        "class"]
```



In [8] :

```python
for c in cols[:-1]:
    sns.histplot(df, x=c, hue=cols[-1], bins=50, stat='probability')
    plt.show()
```



![20210721_172618](../../assets/img/20210721_172618.png)



#### Train Model with PyTorch

#### 

In [9] : 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```



![20210721_173548](../../assets/img/20210721_173548.png)



![20210721_173616](../../assets/img/20210721_173616.png)

