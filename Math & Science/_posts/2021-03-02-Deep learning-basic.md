---
layout: post
title: Deep learning Studying(1) - Deep learning 기본 개념
subtitle: Part.1 딥러닝이란?
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---



# Ch 01. Deep Learning Overview

### Part.1 딥러닝이란?

##### Deep Neural Networks (DNN)을 학습시켜 문제를 해결하는 것

인공신경망(Artificial Neural Networks)의 적통을 이어받음
Neuron들로 구성된 신경망을 학습하여 문제를 해결하도록 동작하는 함수

- 기존 신경망에 비하여 더 깊은 구조를 갖는 것이 특징
  - 과거 학습시킬 수 없었던 깊은 신경망을 잘 학습시킬 수 있게 됨
  - 인터넷의 발달로 빅데이터가 널리 활용되고, 이를 통해 깊은 신경망을 학습시킬 수 있게 됨
  - GPU를 활용한 병렬연산에 대한 방법이 대중화되며, 신경망의 학습/추론 속도가 비약적으로 증가



### 왜 딥러닝인가? 

비선형 함수로, 기존 머신러닝에 비해 패턴 인식 능력이 월등함, 

- 이미지나 텍스트, 음성과 같은 분야들에서 비약적인 성능 개선을 만듦
- 기존 머신러닝과 달리 hand-craftedfeature가 필요 없음
- 단순히 raw값을 넣는 것으로, 자동으로 특징(feature)을 학습함



### 딥러닝의 역사와 패러다임의 변화

- 1980년대, 역전파(Back-propagation) 알고리즘의 개발로 인한 중흥기
- 2000년대, 근근히 이어나가던 명맥
- 2010년대 초, ImageNet 우승과 음성인식(Speech#Recognition)의 상용화

![](C:\Users\bigda\Documents\GitHub\Daehyun-Bigbread.github.io\assets\img\image-20210407182158452.png)

- 2015년, 기계번역(Machine#Translation)의 상용화
- 2017년, 알파고(AlphaGo)의 승리
- 2018년, GAN을 통한 이미지 합성의 발전



### 딥러닝의 역사와 패러다임의 변화

- 기존 패러다임
  - Hand-crafted feature를 추출하여 머신러닝 모델에 넣고 학습
  -  여러 단계의 sub-module로 이루어져 있었음
    - ex) 음성인식, 기계번역 등
- 새로운 패러다임
  - Raw 값을 신경망에 넣으면, 자동으로 특징(feature)을 학습
  - 하나의 task에 대해서, 하나의 신경망 모델이 존재하는 end-to-end방식



### 딥러닝의 활용 사례

- 음성인식(Speech Recognition),+사용자 의도 파악(Intend Classification)
  -  ex) Apple Siri, Google Assistant, Samsung Bixby
- 기계번역(Machine+Translation)
- 자율주행(Autonomous+Driving)
- 객체 인식(Object Detection), 이미지 분류(Image+Classification)
  -  ex) 이미지 검색
- 사진 합성(Image+Generation), 사진 보정, Super Resolution
  - ex) 흑백 사진 -> 컬러 사진, 저해상도 사진 -> 고해상도 사진
- 데이터 분석(Data+Science)
  - Tabular 데이터 분석
  - 시계열(time-series) 데이터 분석