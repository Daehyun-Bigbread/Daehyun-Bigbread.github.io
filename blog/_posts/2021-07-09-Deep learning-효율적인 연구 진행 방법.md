---
layout: post
title: Deep learning 강좌(44) - 효율적인 연구 진행 방법
subtitle: Part.2 Hyper-Parameter 별 결과물 관리
gh-repo: Daehyun-Bigbread/daehyun-bigbread.github.io
gh-badge: [star, fork, follow]
tags: [Deeplearning, pytorch]
comments: true
---

# Ch 10. 딥러닝 학습을 쉽게 하는 방법

### Part.2 Hyper-Parameter 별 결과물 관리

* 수많은 튜닝 결과는 어떻게 관리할 것인가?
  * 각 hyper-parameter별 성능 (accuracy, loss 등), 실험 마다 나오는 모델 (weight) 파일



* 가장 간단한 방법 : 모델 파일 이름에 저장
  * model.n_layer-10. n_epochs-100. act-leaky_relu. loss-xxx. accuracy-xx.pth
  * 하지만 결국 Table로 정리가 필요하다. (엑셀)



* 실험 관리를 도와주는 프레임워크
  * MLFLOW: https://mlflow.org/
  * WanDB: https://www.wandb.com/ (부분 유료)
