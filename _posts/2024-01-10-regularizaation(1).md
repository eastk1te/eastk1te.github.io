---
title: '[Paper]Regularization(1)'
author: east
date: 2024-01-10 12:38:00 +09:00
categories: [Paper, Regularization]
tags: [Paper, Regularization, Lidge, Lasso, ElasticNet]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
---

정규화$$_{Regularization}$$이라고 불리는 해당 방법은 모델이 학습 데이터에 대해 과하게 적합되는 현상을 방지해 일반화 성능을 높이기 위한 방법이라고 합니다. 학습 데이터를 외우는 것을 방지하고 학습이 보다 안정적이고 빠르게 만들어 줍니다. 

이번 포스팅을 통해 이전부터 궁금했던 정규화 기법에 대해 공부한 내용을 정리하도록 하겠습니다.

> ## Ⅰ. Bias-Variance Decomposition

모델이 학습 데이터에 과적합 되는 현상은 모델의 높은 분산과 연관이 있고 이는 모델이 학습 데이터의 특정 샘플에 민감하다는 얘기입니다. 머신러닝 모델의 예측 오류를 이해하기 위해서 모델의 총 오류를 세 가지 주요요소로 분해하여 설명합니다.

$$\begin{align}
\text{Total Error} &= \text{Variance} + (\text{Bias}^2) + \text{Irreducible Error} \\
\text{Bias} &= \mathbb{E}[\hat{f}(x)] - f(x) \\
\text{Variance} &= \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2] \\
\end{align}$$ 

![4](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/8bfc4d55-ae36-449d-984c-c03679fab279){: w="500"}
_출처 : 위키피디아_

분산은 예측값이 평균을 중심으로 얼마나 퍼져있는지를 나타내는 예측값의 변동성을 나타내고 편향은 모델이 맞추지 못하는 부분을 나타냅니다. 따라서, 분산이 높으면 중심에 대해 퍼져있고 편향이 높으면 목표와 동떨어져있습니다. 위 그림에서 좌측하단의 그림처럼 분산과 편향이 둘다 낮으면 베스트이겠지만 모델의 복잡도 관점에서 총오차는 동일해 분산과 편향이 trade-off 관계를 가집니다. 즉, 모델이 학습 데이터의 디테일한 부분까지 학습을 하게 된다면 모델의 분산이 높아지기에 이러한 분산을 낮추기위해 정규화 기법을 적용합니다.

회귀모델에서 MSE를 예시로 들면 MSE는 아래와같이 구성됩니다.

$$\begin{align}
MSE(\hat{θ}) &= E[\Vert \hat{θ} - θ_0 \Vert^2] \\
&= E[(\hat{θ} - θ_0)'(\hat{θ} - θ_0)] \\
&= E[(\underbrace{\hat{θ} - E(\hat{θ})}_{α} + \underbrace{E(\hat{θ} - θ_0)}_{β})'(\hat{θ} - E(\hat{θ}) + E(\hat{θ} - θ_0))] \\
&= E[α'α] + E[α'β] + E[β'α] + E[β'β] \\
&= E[α'α] + \underbrace{E[(\hat{θ} - E(\hat{θ}))]}_{=0}(E(\hat{θ} - θ_0)) + (E(\hat{θ} - θ_0))\underbrace{E[(\hat{θ} - E(\hat{θ}))]}_{=0} + E[β'β] \\
&= E[α'α] + E[β'β] \\
&= E[(\hat{θ} - E(\hat{θ}))'(\hat{θ} - E(\hat{θ}))] + E[(E(\hat{θ} - θ_0))'(E(\hat{θ} - θ_0))] \\
&= E[\sum(\hat{θ} - E(\hat{θ}))^2] + E[(E(\hat{θ} - θ_0))'(E(\hat{θ} - θ_0))] \\
&= \sum(E[(\hat{θ} - E(\hat{θ}))^2]) + E[(E(\hat{θ} - θ_0))'(E(\hat{θ} - θ_0))] \\
&= \sum Var(\hat{θ}) + bias(\hat{β})bias(\hat{β}) \\
\end{align}$$ 

그러나 보통의 OLS 추정에서는 표본평균에 대한 비편향 추정량이기 때문에 bias 항이 0이 되어 MSE는 분산과 노이즈 항으로 구성됩니다.

간단하게 말해서 정규화 기법은 모델의 분산을 낮추는 방법으로 "outcome의 simplifying"이라고 이야기할 수 있습니다. 이는 크게 두가지 카테고리인 explicit와 implicit로 분류되고 같은 문제에 결합하여 적용할 수 있습니다.


> ## Ⅱ. Explicit

목적 함수(loss)에 직접적으로 관여하는 방법으로 해당 카테고리로 떨어지는 모든 기술들은 문제에 명시적인 항$$_{term}$$을 추가하는 형태로 가중치 감쇠$$_{Weight-Decay}$$라고도 불리웁니다.

아래와 같이 목적함수가 선언되었을때 추가적인 제약 항을 추가하는 형태로 regularization합니다.

$$J(w_1,w_2,b) = \frac{1}{n}\sum^{n}(y_i-\hat{y}_i)^2$$

앞으로 해당 카테고리의 L1(Lasso), L2(Ridge), ElasticNet 등의 흔항 타입을 소개하도록 하겠습니다.

선형대수에서는 벡터의 크기를 계산하기 위해 아래와 같이 p-norm이라는 방법을 사용합니다.

$$\Vert x \Vert_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{\frac{1}{p}}$$

노름(norm) 공간은 노름이 정의된 벡터 공간으로 노름은 벡터들의 크기를 측정하는 방법입니다. 즉, 노름 공간은 벡터들의 크기를 측정할 수 있는 공간을 의미합니다.

L1-norm은 맨해튼 노름이라고 알려져 있으며 격자 공간 사이의 거리를 측정하는 방법이고, L2-norm은 유클리드 노름으로 직선 거리를 측정합니다. 이러한 특성 때문에 p-norm의 정규화항을 추가하는 Lasso 정규화 기법이 L1, Ridge 정규화 기법이 L2라고도 불립니다.

이외에 행렬 노름으로 행렬의 크기를 측정하는 Frobenius-norm이나 Matrix-norm 등이 존재합니다.

> ## Ⅲ. Ridge

$$J(w_1,w_2,b) = \frac{1}{n}\sum^{n}(y_i-\hat{y}_i)^2 + \lambda * (w_1^2 + w_2^2)$$

Ridege는 L2로도 불리며 목적함수에 가중치 제곱항을 더합니다. 이 가중치 제곱항은 모든 coefficients에 shrink하는 경향이 있지만 L1 규제와 다르게 어느 가중치의 집합이 0으로 가지 않습니다.

해당 논문은 다항회귀에서 최소 잔차 제곱 합(MSE)에 기반한 파라미터 추정이 다중공선성 문제로 인해 예측 벡터들이 직교하지 않아 독립변수들 간에 강한 선형 의존성이 존재할 수 있다는 단점을 보완하고자 해당 논문에서 $$X'X$$ 행렬의 대각선에 작은 양수를 더하는 추정 방법을 제안하였습니다,

다항 회귀 $$Y = Xβ + ε$$에서 β를 추정하기 위해 보통 아래와 같이 최소제곱추정법(OLS, Ordinary Least Squares)을 사용합니다.

$$\begin{align}
SSR &= ε'ε \\
&= (y - Xβ)'(y - Xβ) \\
&= y'y \underbrace{- y'Xβ - β'X'y}_{scalar} + β'X'Xβ \\
&= y'y -2β'X'y + β'X'Xβ \\
\end{align}$$ 

해당 SSR이 최소가 되게하기 위해 β로 미분하면 아래와 같이 됩니다.

$$\begin{align}
\frac{∂}{∂β}SSR &= -2X'y + 2X'Xβ \\
\hat{β} &= (X'X)^{-1}X'y
\end{align}$$ 

이 과정에서 $$X'X$$가 상관관계행렬과 같은 unit matrix라면 좋지만 그렇지 않다면 모델의 성능이 떨어져 "error"에 민감하게 반응합니다. 이러한 현상은 예측값과 실제값의 차이가 클수록 신뢰도가 더 떨어지도록 됩니다. 따라서, 최소제곱추정법은 데이터를 생성하는 과정의 물리, 화학, 엔지니어링 같은 분야에서 적합하지 않습니다.

이러한 경우 블랙박스나 변수 제거를 통해 대안적인 모델링을 사용할 수 있는데 강제로 사용하면 $$X'X$$형태의 $$X_i$$ 사이의 상관관계의 연결을 파괴하게 됩니다. 

> ### ⅰ. Properties of best linear unbiased estimation

위의 과정을 통해 $$\hat{β}$$의 성질을 알아보았으며 $X'X$가 단위 행렬(상관 행렬) 형태가 아닌 경우  β의 추정 조건의 효과를 설명하기 위해서 $$\hat{β}$$의 공분산 행렬($$Var(\hat{β})$$)과 거리($$E[L_1^2]$$) 두 가지 성질을 다루었습니다.

$$\hat{β}$$의 분산은 $$\hat{β}$$ 간의 공분산행렬로 해석될 수 있습니다.

$$\begin{align}
Var(\hat{β}) &= Var((X'X)^{-1}X'y) \tag{i}\\
&= \underbrace{(X'X)^{-1}X'}_{scalar}Var(y)X(X'X)^{-1} \\
&= \sigma^2(X'X)^{-1}
\end{align}$$ 

(1)에서 $$X'X$$는 회귀 계수의 공분산 행렬의 추정치로 사용됩니다.

$$L_1 = Distance \; \hat{β} \; from \; β \;$$

$$\hat{β}$$와 β의 거리인 $$L_1$$은 아래와 같이 표현할 수 있습니다.

$$\begin{align}
\hat{β} &= (X'X)^{-1}X'y \\
&= (X'X)^{-1}X'(Xβ + ε) \\
&= β + (X'X)^{-1}X'ε \\
\hat{β} - β &= (X'X)^{-1}X'ε \\
\end{align}$$ 

오차가 서로 독립이고 등분산성을 가정하면 오차의 분산은 대각원소가 $$\sigma^2$$이고, 비대각원소는 0인 잔차 벡터 ε의 공분산 행렬($$E[ε'ε] = \sigma^2I$$)가 성립해 아래 처럼 전개됩니다. 

$$\begin{align}
E[L_1^2] &= E[(\hat{β} - β)'(\hat{β} - β)] \tag{ii}\\
&= E[((X'X)^{-1}X'ε)'((X'X)^{-1}X'ε)] \\
&= E[tr(((X'X)^{-1}X'ε)'((X'X)^{-1}X'ε))] \\
&= E[tr(ε'X((X'X)^{-1})((X'X)^{-1}X'ε))] \\
&= E[tr(ε'εX'X(X'X)^{-1}((X'X)^{-1}))] \\
&= E[tr(ε'ε(X'X)^{-1})] \\
&= tr(E[ε'ε(X'X)^{-1}]) \\
&= tr(E[ε'ε](X'X)^{-1}) \\
&= tr((X'X)^{-1}\sigma^2) \\
&= \sigma^2tr((X'X)^{-1}) \\
\end{align}$$ 

이 과정에는 Trace의 성질(tr(AB) = tr(BA), A'A = tr(A'A), E[tr(A)] = tr(E[A])) 들이 사용되었습니다. 또한, 변수들의 제곱합으로 공분산행렬인 X'X의 대각합 tr(X'X)는 아래의 성질을 만족합니다.(P : A의 고유벡터, D : 대각행렬)

$$\begin{align}
tr(A) &= tr(PDP^{-1})\\
&= tr(DPP^{-1}) \\
&= tr(D)\\
\end{align}$$ 

위의 내용을 다시 요약하면, 아래와 같이 나타낼 수 있습니다.

$$E[L_1^2] = \sigma^2\sum\frac{1}{\lambda_i}\text{, }\lambda_i\text{ : i의 분산정보}$$

따라서, 위의 평균과 분산의 하한은 각각 $$\frac{\sigma^2}{\lambda_{min}}, \frac{\sigma^4}{\lambda_{min}^2}$$이 됩니다. 따라서, 요인 factor 공간의 shape는 $$X'X$$에서 하나 이상의 작은 eigenvalue 값이 작을수록 거리가 멀어지게 됩니다. 즉, 고유값이 클수록 비직교 데이터에서 존재하는 문제가 발생합니다. 

직관적으로 설명하면 변수들의 분산정보($$\lambda_i$$)가 작아지면서 다중공선성 발생시 각 독립변수의 영향을 미쳐 각 변수들의 고유 정보량이 작아지게 됩니다.


> ### ⅱ. Ridge Regression

$$X'X$$의 대각행렬이 단위 행렬에서 벗어나는 큰 값을 가질때 변수들의 분산이 크다는 것을 의미해 다중공선성이 증가합니다. 따라서 $$X'X$$에 kI를 더하여 고유정보가 작아지는 것을 방지합니다.

아래와 같이 Ridge 추정량을 W로 치환하여 고유값을 계산합니다.

$$\begin{align}
\hat{β}^* &= (X'X + kI)^{-1}X'y \\
 &= WX'y \\
\end{align}$$ 

$$A^{-1}x = \lambda^{-1}x$$를 활용하여 kI를 더한 W의 고유값이 $$\frac{1}{\lambda + k}$$가 되어 $$X'X$$행렬의 고유값 분모에 k를 더하는 방법을 취해 다중공선성을 작게 만듭니다.

이를 $$y_* = \begin{pmatrix} y \\ 0 \end{pmatrix}, X_* = \begin{pmatrix} X \\ kI \end{pmatrix}$$로 치환하고, 목적함수로 표현하면 아래와 같습니다.

$$\begin{align}
SSR &= ε'ε \\
&= (y_* - X_*β)'(y_* - X_*β) \\
&= \begin{pmatrix} y-Xβ \\ - kβ \end{pmatrix}'\begin{pmatrix} y-Xβ \\ - kβ \end{pmatrix} \\
&= (y-Xβ)'(y-Xβ) + k^2β^2 \\
&= (y-Xβ)'(y-Xβ) + \lambda β^2 \\
\end{align}$$ 



![1](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/ad44f1f7-4b2d-4460-9132-f42112a55647){: w="500"}
_Figure 1 : 분산, 제곱편향 및 매개변수 k 간의 그래프_

Figure 1을 통해 모델의 복잡성과 정규화 강도를 조절하는 매개변수 k에 따른 효과를 확인할 수 있습니다. 이러한 그래프는 편향-분산의 trade-off 관계를 시각적으로 이해하는데 도움이 되고, 파라미터가 0에 가까울수록 Ridge모델이 선형회귀모델과 유사해집니다. 즉, k에 따라 모델이 실제 데이터의 패턴을 포착하는 능력이 감소해 덜민감해집니다.

쉽게 말해 k가 증가하면 공분산 행렬에서 변수의 분산이 작아져 다중공선성으로 인한 불안정성을 줄여 모델의 예측 분산을 감소시킵니다. 이는 안정적인 모델 추정을 가능하게 해줍니다.


<!-- 
> #### Update. $V[L_1^2]$의 증명 부분 해결(24.04.22)
{: .prompt-info }

증명 아직 못함. 
-->







> ## Ⅳ. Lasso

$$J(w_1,w_2,b) = \frac{1}{n}\sum^{n}(y_i-\hat{y}_i)^2 + \lambda * (\vert w_1 \vert + \vert w_2 \vert)$$

L1-norm을 사용해 L1이라 불리는 Lasso는 가중치 계수가 0으로 수렴하는 성질을 포함하는 가중치 계수의 절대값 항을 더하는 형식입니다.

앞서 설명한 MSE 방법은 불편 추정량이지만 분산이 클 때가 종종있고, 예측 오차를 크게하여 회귀 모형의 예측 성능을 안 좋게도 합니다. 이러한 문제를 해결하기위해 두 가지 방법이 존재합니다. 

처음으로는 Ridge 처럼 추정량의 편의가 발생하는 대신 적절하게 회귀계수를 축소시켜 분산을 획기적으로 줄여 전체 오차를 줄이는 방법이  방법이나 모형의 해석 방법이 있습니다.

두번째로는 모형의 해석에서 변수 선택을 통해 필요한 변수만을 모형에 추가하여 해석력을 높이는 방법으로 해당 논문이 초점을 맞춘 방법입니다.


> ### ⅰ. Lasso Regression


우선 MSE를 아래와 같이 표현이 가능합니다.

$$argmin{\sum(y_i - α - \sum β_jx_{ij})^2}$$

α는 편차로 $$\bar{y}$$로 설정할 수 있고, α에 대한 제약이 없으면 최적화 과정에서 $$\bar{y}$$가 목표의 평균(0)으로 설정되게 됩니다. 즉, 회귀 모델에서는 일반성을 잃지 않고 $$\bar{y}=0$$ 설정합니다. 

따라서, 모델이 잔차를 최소화하기위해 평균을 지나는 Centering 변환($$y^* = y_i - \bar{y}$$, $$α^* = α - \bar{y}$$)을 통해 아래와 같이 치환하여 절편을 없앨 수 있습니다.

$$argmin{\sum(y_i^* - α^* - \sum β_jx_{ij})^2}$$


Lasso는 아래와 같은 NG(Non-negative Garrote) 추정량을 모티브로 제약 조건을 추가합니다.

$$\hat{β}_{NG} = c \cdot \hat{β}_{OLS}$$

해당 추정량은 회귀계수에 양수의 상수 c를 곱하여 제약 조건을 추가하는 방식으로 모델링 됩니다. 이는 OLS 추정치를 기반으로 하지만, 양수의 가중치를 적용하여 회귀 계수를 제한합니다.

위와 유사한 제약 조건을 추가한 Lasso의 MSE는 아래와 같이 표기할 수 있습니다.

$$argmin{\sum(y_i - \sum c_jβ_jx_{ij})^2} \; subject \; to \; c_j \geq 0, \sum c_j \leq t$$

동일하게 양수의 상수 c를 곱하지만 상수의 총합이 t보다 작게 설정하여 변수 선택이 가능하게 됩니다.

이렇게 제약조건을 포함하는 함수는 아래와 같은 라그랑주 승수$$_{Lagrange-multipliers}$$를 사용하여 변환됩니다. f(x)는 최소화할 함수이고, g(x)는 제약조건입니다.

$$\begin{align}
L(x, \lambda) &= f(X) + \lambda g(x) \\
\frac{∂}{∂x}L &= 0 \\
\frac{∂}{∂\lambda}L &= 0 \\
\end{align}$$ 

이러한 라그랑주 승수를 통해 제약조건이 있는 최적화문제를 해결하게 됩니다.



결론적으로 Lasso는 OLS의 MSE에 L1-norm 패널티항을 더하는 형식으로 아래와 같이 표현됩니다.

$$\begin{align}
argmin{\sum(y_i - \sum β_jx_{ij})^2} ;\ subject ;\ to ;\ \sum |β| \leq t\\
\Downarrow \\
MSE_{Lasso} = MSE + \alpha \vert w \vert\\
\end{align}$$ 


동일하게 Ridge에서도 아래와 같이 표현됩니다.

$$\begin{align}
argmin{\sum(y_i - \sum β_jx_{ij})^2} ;\ subject ;\ to ;\ \sum  β^2 \leq t\\
\Downarrow \\
MSE_{Lasso} = MSE + \alpha w^2 \\
\end{align}$$ 

> ### ⅱ. Geometry of Lasso

![2](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/364e3d01-c35e-47e7-a8e6-cca196090c0b){: w="500"}
_Figure 2 : 등고선은 손실함수를 뜻하고, 음영 처리된 영역은 제약 영역을 뜻함_

기하학적 표현으로 Figure 2 에서 Ridge는 $$\sum β^2 \leq t$$ 제약조건으로 반지름이 t인 반원 모양의 제약 영역을 보이고, Lasso는 $$\sum \vert β \vert \leq t$$로 마름모 꼴의 모양의 제약공간을 보입니다.

특히, Lasso 에서는 해당 등고선이 마름모의 꼭짓점과 마주쳐 특정 변수가 0으로 수렴하는 성질을 시각적으로 보여줍니다. Ridge는 0에 가까울 수는 있지만 완전히 0으로 수렴하지 않습니다.












> ## Ⅴ. Elastic Net

$$J(w_1,w_2,b) = \frac{1}{n}\sum^{n}(y_i-\hat{y}_i)^2 + \lambda * \alpha * (\vert w_1 \vert + \vert w_2 \vert) + \lambda * (1- \alpha) *  (w_1^2 + w_2^2)$$

ElasticNet은 L1과 L2의 결합으로 위의 두항이 cost function에 더해지고, 새로운 파라미터는 두 사이의 밸런스를 조정합니다.

![3](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/ff997900-bbb1-4289-adac-0055a200ab56){: .w="400"}
_Figure 3 : 2차원의 contour plot_

두 항의 결합을 통해 Figure 3와 같이 제약공간의 변형이 진행됩니다.

Ridge는 중요도가 낮은 변수를 선택적으로 제거하기 어렵고 Lasso는 변수의 수가 표본 수 보다 많거나 상관관계가 높은 변수들이 많은 경우에는 성능이 저하되는 단점을을 가집니다. 이러한 단점들을 극복하기위해 두 항을 결합하여 변수 선택과 분산을 줄이는 방법을 제안하였습니다.

> ## Ⅵ. When to use

그렇다면 언제 이러한 정규화 기법들을 사용해야 할까요?

대부분의 Scikit-learn 모델은 L2를 디폴트로 가지고 있습니다. L1은 몇 가중치가 0으로 수렴하기에 해당 특성이 제거되므로 과적합을 방지하기 위해서는 보통 L2가 더 나은 방법이라고 합니다. 또한, ElasticNet은 해당 데이터와 모델에 L1과 L2 둘 중 어느 것이 더 적합할지 확실하지 않을 때 유용할 수 있습니다.

일반적으로 L2는 변수가 많고 모델에 대부분 사용하고 싶을때 사용하고, L1은 상관관계가 많은 고차원의 데이터셋일 때 부분 집합만을 사용하기 원할떄 사용한다고 합니다.


> ## Ⅶ. Code

모델의 목적함수에 직접적으로 정규화 항을 추가하는 방법도 있지만 tensorflow의 Regularizer는 레이어의 가중치에 정규화 패널티를 직접 적용할 수 있습니다. 이는 적용 위치의 차이를 통해 적용되는 수준과 강도를 조절합니다.

```python
class L1(Regularizer):
    def __call__(self, x):
        # L1 Loss = self.l2 * sum(|w|)
        return self.l1 * ops.sum(ops.absolute(x))

class L2(Regularizer):
    def __call__(self, x):
        # L2 Loss = self.l2 * sum(w ** 2)
        return self.l2 * ops.sum(ops.square(x))

class L1L2(Regularizer):
    def __call__(self, x):
        regularization = ops.convert_to_tensor(0.0, dtype=x.dtype)
        if self.l1:
            regularization += self.l1 * ops.sum(ops.absolute(x))
        if self.l2:
            regularization += self.l2 * ops.sum(ops.square(x))
        return regularization

# 해당 Loss들은 Layer.add_weight로 layer.losses에 추거됩니다.
```


1. [Everything You Need To Know About Regularization](https://towardsdatascience.com/everything-you-need-to-know-about-regularization-64734f240622)
2. [Hoerl, A.E. and Kennard, R.W., 1970. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), pp.55-67.](http://homepages.math.uic.edu/~lreyzin/papers/ridge.pdf)
3. [Tibshirani, R., 1996. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), pp.267-288.](http://www-personal.umich.edu/~jizhu/jizhu/wuke/Tibs-JRSSB96.pdf)
4. [H. Zou et al, "Regularization and variable selection via the elastic net", 2005](https://hastie.su.domains/Papers/elasticnet.pdf)
5. [Expectation of dot product of Distance from β^ to β](https://math.stackexchange.com/questions/4358222/expectation-of-dot-product-of-distance-from-hat-beta-to-beta)
6. [Ridge Regression Proof and Implementation](https://www.kaggle.com/code/residentmario/ridge-regression-proof-and-implementation)
7. [bias-variance decomposition](https://norman3.github.io/prml/docs/chapter03/2.html)
8. [Hastie, T., Tibshirani, R., and Friedman, J., 2009. "The Elements of Statistical Learning: Data Mining, Inference, and Prediction."](https://hastie.su.domains/Papers/ESLII.pdf)