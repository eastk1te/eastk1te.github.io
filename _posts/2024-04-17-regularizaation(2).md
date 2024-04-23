---
title: '[Paper]Regularization(2)'
author: east
date: 2024-04-24 04:00:00 +09:00
categories: [Paper, Regularization]
tags: [Paper, Regularization, Data Augmentation, BatchNormalization, Dropout, EarlyStopping]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
---




이전 포스팅에서 explicit regluarization을 소개했습니다. 다음으로 Implicit regularization으로 흔히 사용되는 Data augmentation, EarlyStopping, Batch Normalization, Dropout 등을 소개하겠습니다.
Label Smoothing(라벨 스무딩은 모델이 너무 확신하기 쉬운 예측을 방지하기 위해 정답 라벨에 대한 신뢰도를 조금 낮추는 기법입니다. 이를 통해 모델이 일반적인 특징에만 의존하여 좀 더 일반화된 결정을 내릴 수 있습니다.)

![2](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/478a343f-59fe-4e4f-99d4-4bb89f2b92c8){: w="500"}
_Figure 2 : paperswithcode_

> ## Ⅰ. Implicit Regularization

Explicit가 아닌 기술들은 자동적으로 Implicit 카테고리로 들어가게 됩니다. 즉, 손실함수에 직접적으로 명시하지 않고도 과적합을 방지하는 방법을 의미합니다.

> ## Ⅱ. Data augmentation

과적합은 학습 데이터에 너무 맞춰져 있어 새로운 데이터에 대한 일반화 성능이 저하되는 현상입니다. 따라서 이를 해결하기 위해 학습 데이터를 늘려서  다양성을 증가시켜 모델이 일반적인 패턴을 파악하도록 하는 방법입니다. 이러한 방법은 다른 방버들과 다르게 일반화 성능의 일관된 향상을 제공해 파라미터를 재조정하는 등의 작업이 필요하지 않습니다.

![3](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/e4df4896-e81b-4904-9a12-3c23afad25fd){: w="500"}
_Figure 3 : 일반화 성능 비교, (라이트 증강) 수평, 수직 이동 등과 같은 간단한 변환 (헤비 증강) 보다 넓은 범위의 아핀 변환 등을 수행_

보통 이미지 데이터에는 미러링, 크롭핑, 필터 적용 등과 같은 변환 기법을 적용합니다. 또한 라이트닝, 블러링, 쉬프팅 등과 같은 효과를 추가하여 다양성을 높입니다.

또한, 텍스트 데이터들에는 번역, 동의어 및 유의어 교체, 문장 순서의 재구성 등으로 증강이 가능합니다.





> ## Ⅲ. Early stopping

EarlyStopping은 학습 중 가장 좋은 가중치를 저장하고 업데이트하며 더 이상 성능의 개선이 이루어지지 않을때 학습을 중지하는 방법으로 최적화 과정을 매개변수 공간으로 제한하는 정규화 기법으로 작동합니다.

신경망의 지도 학습 중에 과적합이 시작되는 시점을 감지하는데 검증 데이터 셋을 사용할 수 있으며 과적합을 피하기 위해 수렴하기 전에 아래와 같이 중지합니다. 그러나 이러한 방법은 조기 종료의 일반적인 기준을 선택하기가 어렵고, 학습이 갑자기 중단되는 경우가 많습니다.

![1](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/9f918452-47a2-4dac-b48b-4c569eb67199){: w="500"}
_Figure 1 : 이상적인 학습 및 검증 오차 곡선_

일반적인 지도 학습에서 Figure 1과 같은 이상적인 다이어그램을 확인할 수 있습니다. 그러나 실제로는 위처럼 이상적으로 나오지않고 일반화 오류가 증가해도 유효성 검사 오류가 더 낮아질 수 있어 이러한 여러 지역 최소값을 갖게됩니다. 따라서 훈련 시간과 일반화 오류 사이의 최적의 결과값을 찾기란 쉽지 않습니다.



> ### ⅰ. On EarlyStopping in Gradient desent learning

해당 논문을 통해 조기 종료 규칙(epoch 수에 관한 함수)으로 일반화 오류의 확률적 상한선을 제안하여 EarlyStopping을 통해 일반화 오류를 줄이는 원리를 설명하였습니다.

EarlyStopping은 ML에서 비모수 회귀 문제를 정규화하는데 사용될 수 있는데, 주어진 입력 공간 X, 출력 공간 Y에서 주어진 확률분포 ρ에서 특정 입력 x를 기반으로 하는 회귀함수 $$f_{ ρ}$$를 찾는 것입니다. 즉, 해당 회귀함수와 가까워지는 과정에서 EarlyStopping을 통해 모델이 학습 데이터에만 너무 많이 적합되는 것을 방지할 수 있습니다.

$$ f_{ρ}(x) = \int_{Y} y \, d ρ(y \mid x), \, x \in X $$

회귀 함수($$f_{ ρ}$$)를 근사화하는 데 일반적으로 선택되는 한 가지 방법은 재생 커널 힐베르트 공간에서 함수를 사용하는 것입니다. 이러한 공간은 무한 차원이 될 수 있으며, 이는 임의 크기의 훈련 세트에 과적합되는 해결책을 제공할 수 있습니다. 

`Reproducing Kernel Hilbert Space`
: 재생커널힐베르트 공간은 함수 공간의 특별한 유형으로, 함수를 포함하고 있는 유클리드 공간의 개념을 무한 차원으로 일반화한 Hilbert 공간입니다. 이 공간은 임의의 입력값에 대한 함수의 값들을 내적으로 계산할 수 있도록 하는 커널의 일종으로 재생 성질($$f(y) = \langle f, K(\cdot, y) \rangle, \; y \in \forall $$)을 만족하는 재생 커널(reproducing kernel)을 지니고 있습니다. $$K(x_i, x_j) = \langle \Phi(x_i), \Phi(x_j) \rangle$$ 에서  $$K(\cdot)$$ 는 커널 함수를 나타내며,  $$\Phi(\cdot)$$ 는 입력 데이터  $$x_i$$를 고차원 특징 공간으로 매핑하는 함수입니다. 즉, 커널 함수를 사용하여 데이터를 고차원 특징 공간으로 매핑하는 함수 공간입니다.

이러한 비모수 회귀 문제를 정규화하는 한 가지 방법은 경사 하강법과 같은 반복적인 절차에 조기 중단 규칙을 적용하는 것입니다. 제안된 조기 중단 규칙은 반복 횟수의 함수로 일반화 오류에 대한 상한에 대한 분석을 기반으로 합니다. 

즉, 반복횟수가 증가함에 따라 일반화 오류가 어떻게 변화하는지를 고려한 상한을 분석하여 모델의 학습을 언제 중단해야할지 결정합니다.


> ### ⅱ. Example of MSE

MSE를 예시로 $$X \subseteq \mathbb{R}^n$$이고 $$Y = \mathbb{R}$$일 때, 독립적으로 추출된 샘플 집합 $$\mathbf{z} = {(x_i, y_i) \in X \times Y : i = 1, \dots, m} \in Z^m $$ 에 대해, 다음과 같은 기능을 최소화합니다.

$${\mathcal{E}}(f) = \int_{X \times Y} (f(x) - y)^{2} \, d ρ$$

여기서 f는 재생성 커널 힐베르트 공간 $$\mathcal{H}$$의 구성원이고, 우리는 최소 제곱 손실 함수에 대한 기대 오차를 최소화해야합니다. $$\mathcal{E}$$가 알려지지 않은 확률 측도  ρ에 의존하기 때문에 계산에 사용할 수 없습니다. 따라서 아래와 같이 실제 데이터 집합에 대한 경험적인 손실함수의 기댓값을 사용하여 모델을 조정하고 최적화합니다.

$$\mathcal{E}_{\mathbf{z}}(f) = \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i)^{2} \tag{1}$$

따라서, 우리는 아래와 같이 $$f_z$$와 $$f_ρ$$의 기대 위험의 차이를 제어하고 싶습니다. 

$$\mathcal{E}(f_{t}^{\mathbf{z} })-{\mathcal{E}}(f_ρ)$$

이러한 차이는 두 가지 항의 합으로 다시 쓸 수 있습니다.

$${\mathcal {E}}(f_{t}^{\mathbf {z} })-{\mathcal {E}}(f_{ ρ }) = \left[{\mathcal {E}}(f_{t}^{\mathbf {z} })-{\mathcal {E}}(f_{t})\right]+\left[{\mathcal {E}}(f_{t})-{\mathcal {E}}(f_{ ρ })\right]$$

위 방정식은 편향-분산의 균형을 제시하며 이를 해결하는 것으로 최적의 조기 종료 규칙을 얻을 수 있습니다. 이 규칙은 일반화 오류에 대한 확률적 한계와 연관되어 교차 검증과 같은 데이터 기반 방법을 사용한 적응형 조기 종료 규칙을 얻을 수 있습니다.

위의 나온 (1)을 최적화 하기위해 아래와 같은 경사하강법으로 현재 모델과 기준 모델 사이의 차이에 대한 커널 손실이 적용됩니다.

$$\begin{align}
f_{t+1} &= f_t - \gamma_t L_K(f_t - f_{\rho}), \quad f_0 = 0 \\
\gamma_t &= \frac{1}{\kappa^2(t + 1)^\theta} \;, \theta \in [0, 1] \\
(L_K f)(x') &= \int_{X} K(x', x) f(x) \, d\rho_X \\
∇E(f) &= L_Kf - L_Kf_{\rho}\\
∇E_z(f) &= \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i)K(x_i) \\
\end{align}$$ 

위와 같은 방법을 통해 함수 f의 (1) RKHS에 대한 목적 함수와 (2) 일반화 오차를 최소화하는 값을 보여줍니다.


![4](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/64b52494-175c-48b8-8e6e-4fc4c4798b07){: w="500"}
_Figure 4 : 조기중단 규칙 하 모델의 일반화 오류에 대한 관계_

Figure 4를 통해 모델의 근사 오차를 최소화하는 학습 데이터(m)과 관련된 최적의 조기 중단 규칙을 설정할 수 있었습니다. 이것은 해당 규칙 아래 모델의 일반화 오류가 데이터 크기 m에 따라 어떻게 감소하는지에 대한 내용도 나타냅니다.


![5](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/77fc91b3-1292-4964-9ae8-8afb5b1d1611){: w="500"}
_Figure 5 : 일반화 오류에 대한 두번째 상한선._

Figure 5에서 일반화의 오류를 분산(Sample Error)과 편향(Approximation Error)에 대한 식으로 표현한 후 이 상한값을 최소화하는 t값을 찾기위해 최소 정수로 설정하여 계산합니다. 이를 통해 최적의 조기 종료 시점을 사용했을때의 상한선을 추정합니다. 

즉, Figure 5 마지막에사 최적의 조기 종료 규칙($$t^*(m)$$)에 따른 `상한을 추정하여 학습을 조기에 중단하여 해당 상한 이상으로 학습이 진행되지 않도록 제한함으로써 과적합을 방지`하게 되었습니다. 따라서, 이러한 EarlyStopping을 통해 통해 학습 프로세스가 이 상한값을 넘어가지 않도록 설정하는 것이 중요합니다.

이렇게 이론적으로 도출된 최적의 조기 종료 시간은 근사값일 뿐이라 실제 적용에서는 해당 값에 근거하여 초기 가이드라인을 제안하되 조정할 필요가 있습니다.

> ### ⅲ. Code

Keras에서는 Loss가 몇번의 epoch 동안 개선되지 않으면 학습을 조기에 중단하도록 하거나 가장 좋았던 지점의 가중치를 복원하는 옵션등을 사용합니다.

```python
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
...
history = model.fit(..., callbacks=[callback])
```





> ## Ⅳ. Dropout

Dropout은 신경망 연구에 저명하신 Geoffrey E. Hinton님이 2014년에 발표한 "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" 논문을 통해 현재까지 5만회에 가까운 인용수를 지닐 만큼 유명한 정규화 방법을 소개하였습니다.

해당 방법을 요약하자면 학습 중 신경망의 연결 일부를 무작위로 끊어 각 부분이 서로 의존적으로 발달하는 것을 방지합니다. 

이 방법은 진화론적 관점에서 부모로 부터 유전자의 절반을 각각 받아 소량의 무작위 돌연변이를 추가한 후 결합하여 자손을 생성하는 과정에서 유래하였다고 합니다. 이러한 방법은 서로 다른 유전자들이 결합하여 잘 작동하는 능력이 더욱 강하게 만든다고 합니다. 해당 이론과 유사하게 드랍아웃을 통해 잠재 유닛이 무작위로 선택된 다른 유닛과 함께 작업하는 방법을 배워 강건하게 만들고, 다른 유닛에 대한 의존 없이도 독립적인 기능을 생성하게 될 것입니다.


![6](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/1f45dd88-a595-4349-83bb-636ce27fa612){: w="500"}
_Figure 6 : (a) 일반적인 신경망 (b) 드랍아웃을 적용한 신경망_

고정된 크기의 모델을 정규화하는 가장 좋은 방법은 파라미터 설정에 대한 모델의 예측 결과들을 평균내어 일반적으로 만드는 것이지만 큰 신경망의 경우 결과를 평균화하는 과정에 비용이 많이 들어가게 됩니다. 따라서, 이러한 문제를 다루기 위해 Dropout은 과적합을 방지하고 많고 다양한 신경망 아키텍처의 결합으로 효과적으로 근사하는 방법을 제공합니다.

학습 시 각 유닛은 독립적인 일정 확률(보통 p \in (0.5, 1.0), 0.6이 베스트라고 논문 결과를 보여줌)을 통해 활성화 여부를 정하게 됩니다. 이는 n개의 유닛이 지수적으로 $$2^n$$개의 가능한 "thinned" 네트워크를 샘플링하여 학습하는 것으로 볼 수 있습니다. 이러한 방식으로 학습된 네트워크는 공유된 가중치를 가지고, 이는 각 얇은 네트워크가 독립적으로 학습된 것과 유사한 효과를 내면서도 실제로는 많은 네트워크를 별도로 학습시키지 않아도 됩니다. 

그리고 테스트시 드랍아웃 없이 가중치를 공유하는 단일 "unthinned" 신경망을 사용하게 되는데, 해당 네트워크의 가중치는 학습된 가중치의 scaled-down된 버전으로 출력 가중치가 p로 조절되어 "thinned" 네트워크의 결과와 유사하도록 만듭니다. 이러한 스케일링을 통해 가중치를 공유하는 $$2^n$$ 네트워크를 단일 신경망으로 결합되어 여러 "thinned"의 네트워크의 예측을 평균화하는 효과에 근사하게 됩니다. 즉, 근사 평균 방법을 사용하여 상당한 일반화 에러를 낮추게 됩니다.

간단하게 말해 여러 다른 신경망을 학습시켜 단 하나의 신경망을 사용하는 것으로 평균 예측값을 내는 것과 유사한 결과를 내는 것으로 학습된 모델이 잘 일반화되어 새로운 데이터에 대해 더 좋은 성능을 낼 수 있게 됩니다.단순화해서 해결할 수 있습니다.

이러한 드랍아웃의 아이디어는 FFN 뿐만 아니라 일반적으로 볼츠만 머신과 같은 그래픽 모델 등 에서도 적용이 가능합니다.

> ### ⅰ. Model Architecture

![7](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/c6273bef-d698-49f3-a315-85aa19cf08c5){: w="500"}
_Figure 7 : (a) 일반적인 신경망 (b) 드랍아웃을 적용한 신경망_

$$\begin{align}
f(\cdot) &= \text{activation func} \\
\star &= \text{element-wise product}
r^{(l)}_j &~ \text(Bernoulli)(p) \\
\tilde{y}^{(l)} &= r^{(l)} \star y^{(l)} \\
z_i^{(l+1)} &= w_i^{(l+1)}\tilde{y}^{(l)} + b_i^{(l+1)} \\
y_i^{(l+1)} &= f(z_i^{(l+1)})\\
\end{align}$$ 

p값을 가지는 독립적인 베르누이 함수로 활성화 유무로 thinned outputs($$\tilde{y}^{(l)}$$)을 생성하여 다음 층의 입력으로 사용합니다. 이는 더 큰 네트워크에서 하위 네트워크를 샘플링하는 것과 같습니다. 테스트 시 가중치는 p를 통해 스케일이 조정($$W_{test}^{(l)} = p W^{(l)}$$)됩니다.

> ### ⅱ. Result

![8](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/d3260d9a-cea5-4a20-990d-ad9ae7d2b94a){: w="500"}
_Figure 8 : 드랍아웃 유무에 따른 다양한 아키텍처에서의 성능 비교 그래프_

Figure 8 을통해 Dropout을 적용한 아키텍처가 일반화 성능을 더 잘 표현하는 것을 확인할 수 있습니다.

![9](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/aaaa7da3-ffcb-46ef-b45e-2e2ceca4c973){: w="500"}
_Figure 9 : MNIST 데이터에서 AutoEncoder로 학습된 특징_

또한, Figure 9 에서 확인 가능하듯이 왼쪽의 경우 잠재 유닛이 혼자서는 의미 있는 특징을 감지하지 못했지만 오른쪽의 경우 각각의 잠재 유닛이 독립적으로 이미지의 특징을 감지한 것을 확인할 수 있습니다. 이는 드랍아웃이 co-adapting을 분리하는 것을 보여주어 일반화 오류를 줄이는 주된 이유라고 할 수 있습니다.

> ### ⅲ. Code

```python

class Dropout:
  ...
  def forward(self,input_data, is_train=True):
    # 학습 시
    if is_train: 
      # 각 유닛의 활성화 유무를 결정함
      self.mask = np.random.rand(*input_data.shape) > self.dropout_ratio
      return input_data * self.mask

    # 테스트 시
    else:
      # 학습된 가중치를 스케일링함.
      return input_data * (1.0 - self.dropout_ratio)
  ...
```











> ## Ⅴ. Batch Normalization

배치 정규화는 2015년에 Google에서 발표한 "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" 논문으로 5만회가 넘어가는 인용수를 지닌 만큼 많이 사용되는 정규화 기법 중 하나입니다.

$$ \Theta = \text{argmin}_\Theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{ℓ}(x_i, \Theta) $$

확률적 경사 하강법(SGD)은 위와 같이 전체 데이터 세트를 사용하는 대신 미니 배치를 사용하여 계산량을 줄이고 loss를 최소화하는 파라미터를 최적화하는 알고리즘입니다. 

`미니배치(Mini Batch)`
: 미니배치에서의 기울기는 학습 데이터에 대한 기울기의 추정치이고, 배치를 통한 계산은 병렬 처리로 효율적으로 계산될 수 있습니다. 또는, 메모리의 부족현상을 막기 위해 여러번 학습하는 방식으로의 사용이 가능합니다.

그러나 신경망은 이전 층의 출력을 입력으로 받아들이는 연쇄 과정에서 매개변수를 통해 손실함수를 계산함으로 매개변수에 대한 초기값이나 학습률 등의 하이퍼파라미터의 변화에 대해 입력 데이터가 받는 영향이 커져 민감해지므로 신중하게 조정해야합니다.

여기서 각 레이어의 입력 분포의 변화는 레이어가 계속 새로운 분포에 적응해야 한다는 문제로 학습 시 입력되는 분포가 변경되는 현상을 공변량 이동(covariate shift)이라 하고, 일반적으로는 도메인 적응(domain adaptation)을 통해 처리됩니다.

$$ℓ = F2(F1(u, Θ1), Θ2)$$

여기서 F1과 F2는 임의의 변환들이고, Θ1, Θ2는 손실 ℓ을 최소화하기 위해 학습되어야 하는 파라미터들입니다. 이 과정에서 서브네트워크의 입력 분포가 고정되어있지 않은 상태에서 비선형 활성화 함수(sigmoid 등)를 사용하면 그래디언트 폭발이나 소실의 문제를 발생시키기 떄문에 서브네트워크로의 입력 분포가 고정되면 서브네트워크 외부의 레이어에도 긍정적인 영향을 미칠 것입니다. 

즉, 학습시 신경망 내부 노드의 분포 변화를 내부 공변량 이동(Internal Covariate Shift)이라 일컫고, 해당 논문에서 이를 줄이는 새로운 매커니즘인 Batch Normalization을 제안하였습니다. 

> ### ⅰ. Towards Reducing Internal Covariate Shift

입력 데이터의 백색화를 통해 모델이 최적의 파라미터로 더 빠르게 수렴하는 것처럼 입력이 통계적 특성을 가지도록 선형 변환하여 평균을 0으로 만들고 단위 분산을 갖도록 변환하는 것은 네트워크 학습이 더 빨리 수렴하도록 도와줍니다.

따라서, 각 레이어의 입력의 통계적 특성을 동일하게 하는 것이 유리합니다. 이를 위해 입력을 백색화한다면 입력 분포를 고정해 내부 공변량 이동의 부정적인 영향을 제거할 수 있게됩니다. 매 학습 단계마다 또는 일정한 간격으로 네트워크를 백색화한다면 최적화 알고리즘 매개변수가 activation에 의존하도록 변경함으로써 매개변수 크기나 초기값에 대한 기울기의 의존성을 줄이게 됩니다.

`백색화`
: 평균이 0이며 공분산이 단위 행렬인 정규 분포 형태의 데이터로 변환하는 기법으로 데이터의 상관 관계를 제거하고 독립적인 특징을 갖도록 만드는 과정입니다. 특히, 공분산 행렬을 다루어 데이터의 상관관계를 조정합니다.

그러나 최적화 단계에서 매개변수를 업데이트하는데 정규화를 고려해야하는데 백색화의 적용은 레이어의 입력 데이터 뿐만아니라, 간접적으로 전체 데이터셋의 통계적 속성(전체 데이터의 평균)을 반영하기 때문에 각 영향($$frac{∂Norm(x, X)}{∂x}와 frac{∂Norm(x, X)}{∂X}$$)을 계산해야 합니다. 이 과정은 어렵고 미분 가능하지도 않기 때문에 다른 대안을 찾아야합니다.

> ###  ⅱ. Normalization via Mini-Batch Statistics

![12](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/75849016-bd86-49a2-a393-6ee7f80d764d){: w="500"}
_Figure 12 : Normalization_

레이어의 각 입력을 간단하게 정규화($$\hat{x}^{(k)} = \frac {x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x_{(k)}]}}$$)하는 것은 출력 분포를 변화 시킬 수 있어 원래의 입력 분포를 변경하지 않도록 항등 변환이 되도록 해야합니다. 

$$\begin{align}
y^{(k)} &= γ^{(k)}\hat{x}^{(k)} + β^{(k)}\\
γ^{(k)} &= \sqrt{Var[x(k)]}\\
β^{(k)} = E[x(k)]\\
\end{align}$$ 

위와 같이 각 $$x^{(k)}$$에 대해 아래와 같이 정규화된 값을 학습되는 두 개의 스케일링($$γ^{(k)}$$)과 이동 인자($$β^{(k)}$$) 매개변수를 통한 선형변환으로 데이터의 분포를 조절할 수 있게 해 다른 네트워크 레이어로 전달되는 하위 네트워크의 입력으로 네트워크의 표현 능력을 복원해 사용합니다.(ex. x:입력,z:은닉 $$\rightarrow$$ z = W*BN(x)+b)

스케일링과 이동인자를 사용하는 이유는 데이터를 계속 정규화하게 되면 활성화 함수의 비선형 성질을 잃게 되는 문제를 다루기 위해서 입니다.(ex. sigmoid의 경우 입력값이 (0,1) 사이의 값은 선형부분으로 꼬리부분의 비선형 성질을 잃게 됨.)

전체 학습 데이터셋의 통계적 특성을 활용하여 각 학습 단계의 입력을 정규화하는 것은 확률적 최적화 방법에서 현실적이지 않습니다. 따라서, 확률적 경사 하강법 훈련에서 미니배치를 사용해 평균과 분산의 추정치를 생성하면 정규화에 사용되는 통계량이 그라디언트 역전파에 완전히 참여할 수 있습니다. 

$$\begin{align}
\frac{\partial \mathcal{L}}{\partial x_{bi}} \\
\frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma \\
\sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial x_{bi}} \cdot (x_i - \mu_B) \cdot \frac{{-1}}{{2}} \cdot (σ^2_B + \epsilon)^{-\frac{3}{2}} \\
\sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial x_{bi}} \cdot \sqrt{\frac{-1}{σ^2_B + \epsilon}} \\
\sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot x_{bi} \\
\sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \\
\end{align}$$ 

위는 역전파를 위해 체인룰을 이용하여 매개변수의 기울기를 계산과정입니다. 

$$BN_{γ,β} : x_{1...m} \rightarrow y_{1...m} $$

위와 같은 변환을 Batch Normalizing Transform으로 제시하고 지금까지의 과정을 아래 처럼 요약합니다.

![10](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/81a1bb30-edce-4e9d-96fd-402f10120234){: w="500"}
_Figure 10 : 미니배치에서 활성화 x에 적용된 배치 정규화 변환_


따라서, BN 변환은 미분 가능한 변환으로 내부 공변량 변화가 적은 입력 분포에서 계속 학습할 수 있게하고 학습 속도를 가속화합니다. 더욱이 정규화된 activation에 학습된 선형 변환을 적용하여 항등 변환으로 표현하고 네트워크 용량을 보존합니다.









> #### 1. Training and Inference with BatchNormalized Networks

![11](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/069ffeb1-efde-472f-a7fb-19eaaf86a600){: w="500"}
_Figure 11 : Batch-Nomalized Network의 학습 단계의 의사코드_

미니 배치에 의존하는 정규화 activation은 효율적인 학습을 가능케 하지만, 추론 시에는 필요하지도 바람직하지도 않습니다. 결국 추론시 출력이 입력에만 결정론적으로 의존하기를 원해 미니 배치 대신 모집단 통계를 사용하여 아래와 같은 정규화를 사용해 학습 모드와 추론 모드를 구분합니다.

$$\hat{x} = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}}$$

위 식에서 ε은 분산이 0이 되는 것을 막기위한 아주 작은 값으로 이를 무시하면 학습 시와 같은 평균이 0이고 분산이 1인 분포를 가집니다. 여기서 비편형 분산 추정치인 $$Var[x] = \frac{m}{m-1} \cdot E_B[\sigma^2_B]$$를 사용하는데 기대치는 크기 m의 학습 미니 배치에 대해 계산되고 $$\sigma^2_B$$는 표본 분산이 됩니다. 그리고 미니 배치 대신에 이동 평균을 사용하여 학습 시 모델의 정확도를 추적할 수 있습니다.


> #### 2. Batch Normalization enables higher learning rates

심층 신경망에서 높은 학습률을 사용하면 기울기의 폭발, 소실, 지역 최솟값 등의 결과가 나올 수 있습니다. 배치 정규화는 이러한 문제들을 해결하는 방법으로 매개변수에 대한 작은 변경이 그래디언트에 큰 영향을 미치는 것을 방지합니다. 

아래와 같이 배치정규화를 사용하면 레이어를 통한 역전파가 매개변수의 스케일에 영향을 받지않습니다.

$$
BN(Wu) = BN((aW)u)
\frace{\partial BN((aW)u)}{\partial u} = \frace{\partial BN(Wu)}{\partial u}
\frace{\partial BN((aW)u)}{\partial aW} = \frace{1}{a} \cdot \frac{\partial BN(Wu)}{\partial W}
$$

위에서 더 큰 가중치가 더 작은 그래디언트로 유도하며 매개변수의 성장을 안정화시킵니다. 


> #### 3. Batch Normalization regularizes themodel

해당 논문에서 다른 기법들과 함께 사용할때 배치정규화가 일반적으로 과적합을 줄이는 드랍아웃을 제거하거나 강도를 줄이는 현상을 발견했으며 이는 배치정규화가 충분한 일반화효과를 제공했다는 것을 나타냈습니다.

> ### ⅲ. Code

정리하자면 배치정규화는 레이어 입력의 평균과 분산을 고정하는 정규화 단계를 통해 매개변수 크기나 초기값에 대한 기울기의 의존성을 줄여 안정적인 학습을 가능하게 하고 비선형성을 사용하며 일반화 성능을 향상시키는 방법으로 `학습 속도의 증가`, `가중치 초기화에 대한 민감도의 감소` 그리고 `일반화 능력`을 얻을 수 있습니다.

```python

  def call(self, inputs, training=None):
    ...
    if training_value == False: g-explicit-bool-comparison
      # 학습이 아니면 이동 평균, 이동 분산 사용.
      mean, variance = self.moving_mean, self.moving_variance
    # 학습 과정
    else:
      ...
      # 입력에 대한 평균과 분산 계산
      mean, variance = self._moments(
          tf.cast(inputs, self._param_dtype),
          reduction_axes,
          keep_dims=keep_dims)
      
    def _compose_transforms(scale, offset, then_scale, then_offset):
      # 스케일링 인자 포함 적용
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      # 이동 인자만 적용
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    ...

    if training_value == False:  
      mean, variance = self.moving_mean, self.moving_variance
    # 학습 과정
    else:
      # 입력에 대한 추가적인 선형 변환
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(tf.shape(inputs))
        adj_scale = control_flow_util.smart_cond(
            training, lambda: adj_scale, lambda: tf.ones_like(adj_scale))
        adj_bias = control_flow_util.smart_cond(
            training, lambda: adj_bias, lambda: tf.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)
      
    ...

    if self.renorm:
        true_branch = true_branch_renorm
      else:
        true_branch = lambda: _do_update(self.moving_variance, new_variance)
    ...

    return outputs


```

> ## Ⅵ. REFERENCES

1. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
2. ["Data augmentation instead of explicit regularization", 2020](https://arxiv.org/pdf/1806.03852.pdf)
3. [Introduction to RKHS, and some simple kernel algorithms](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf)
4. [Y. Yao at el, "ON EARLY STOPPING IN GRADIENT DESCENT LEARNING",  2007](https://yao-lab.github.io/publications/earlystop.pdf)
5. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate  Shift](https://arxiv.org/pdf/1502.03167.pdf)
6. [LeCun, Y at el, "Efficient backprop", 1998](https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf)
7. [tensorflow-batchnormalization](https://github.com/keras-team/keras/blob/v2.6.0/keras/layers/normalization/batch_normalization.py)