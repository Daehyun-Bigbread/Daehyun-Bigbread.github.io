---
layout: post
title: Deep learning Studying(2) - 좋은 인공지능이란?
subtitle: Part.2 Generalization ~ 3. Workflow
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 01. Deep Learning Overview

## Part.2 Generalization (좋은 인공지능이란?)

* x가 주어졌을 때, y를 반환하는 함수
  * y = f (x)

* 파라미터(weight parameter)란
  * f 가 동작하는 방식 (x가 들어왔을 때, 어떤 y를 뱉어낼 것인가?)을 결정

- 학습이란?
  - x와 y의 쌍으로 이루어진 데이터가 주어졌을 때, x로부터 y로 가는 관계를 배우는 것
  -  x와 y를 통해 적절한 파라미터를 찾아내는 것
- 모델이란?
  - 상황에 따라 알고리즘 자체를 이르거나, 파라미터를 이룬다.



### 좋은 인공지능 모델은? 

* 일반화(Generalization)을 잘하는 모델
* 보지 못한 (unseen) 데이터에 대해서 좋은 예측 (prediction)을 하는 모델
  * 우리는 모든 경우의 수에 대해서 데이터를 모을 수 없기 때문
  * 보지 못한 경우에 대해서, 모델은 좋은 판단을 내릴 수 있어야 함
  
  

### 기존 머신러닝의 한계

- 기존 머신러닝은 주로 선형 또는 낮은 차원의 데이터를 다루기 위해 설계되었음.
-  Kernel 등을 사용하여 비선형 데이터를 다룰 수 있지만, 한계가 명확함
- 이미지, 텍스트, 음성과 같은 훨씬 더 높은 차원의 데이터들에 대해 낮은 성능을 보임



## Part.3 Workflow

### Our Objective

* 주어진 데이터에 대해서 결과를 내는 가상의 함수를 모사하는 함수를 만드는것
  * ex) 주어진 숫자 그림을 보고 숫자 맞추기

![](C:\Users\bigda\Documents\GitHub\Daehyun-Bigbread.github.io\assets\img\20210519_145853.png)

## Working Process

![](C:\Users\bigda\Documents\GitHub\Daehyun-Bigbread.github.io\assets\img\20210519_150422.png)

### 문제 정의

* 풀고자 하는 문제를 단계별로 나누고 simplify#하여야 한다.

  * ex) 김치찌개 끓이기, 라면 끓이기

* 신경망이라는 함수에 넣기 위한 x와 결과값 y가 정의 되어야 한다.

  ### 													y = f (x)

  

### 데이터 수집

* 문제 정의에 따라 정해진 x와 y
* 풀고자 하는 문제의 영역에 따라 수집 방법이 다름
  * 자연어처리, 컴퓨터비전: Crawling
  * 자연어분석: 실제로 수집한 데이터
* 필요에 따라서 레이블링(Labeling) 작업을 수행한다.
  * 자동적으로 레이블(Label)이 y로 주어질 수도 있음
  * 하지만 대부분 경우는 레이블이 따로 필요함
  * 비지도학습 (Unsupervised learning)을 기대 X



### 데이터 전처리 및 분석

* 수집된 데이터를 신경망에 넣어주기 위한 형태로 가공하는 과정
  * 입출력 값을 정제(Cleaning & Normalization)
* 이 과정에서 탐험적분석(Exploratory Data Analaysis, EDA)이 필요
  * 데이터에 알맞는 알고리즘을 찾기 위함
  * 자연어처리, 컴퓨터비전의 경우에는 생략되기도 함
*  영상처리(Computer Vision)의 경우, 데이터 증강(Augmentation)이 수행됨
  * ex) rotation, flipping, shifting 등의 간단한 연산



### 알고리즘 적용

* 데이터에 대해 가설을 세우고, 해당 가설을 위한 알고리즘(모델)을 적용



### 평가

• 문제 정의에 따른 공정하고 올바른 평가 방법 필요
• 가설을 검증하기 위한 실험 설계
• 테스트 셋(Test set) 구성
• 너무 쉽거나 어렵다면 판별력이 떨어짐
• 실제 데이터와 가장 가깝게 구성되어야 함
• 정량적 평가(Extrinsic Evaluation)와 정성적 평가(Intrinsic Evaluation)로 나뉨



### 배포

* 학습과 평가가 완료된 모델 weights 파일을 배포함
*  RESTful API 등을 통해 wrapping 후 배포
*  데이터 분포의 변화에 따른 모델 업데이트 및 유지/보수가 필요할 수 있음