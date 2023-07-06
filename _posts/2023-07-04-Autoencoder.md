---
title: 'AutoEncoder'
author: east
date: 2023-07-04 00:00:00 +09:00
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAGS]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
---

우리는 [이전 포스트]()를 통해 DBN의 학습에 대해 배웠으며 DBN의 pre-trained 부분은 AutoEncoder 형식의 구조로 구성되었다는 것을 알게 되었습니다. 그런데 여기서 AutoEncoder 방식을 정확히 이해하고자 해당 포스트를 작성하게 되었습니다.

AutoEncoder는 비지도 학습 방법인 신경망 기반의 기계학습 모델로 차원축소와 최대우도밀도추정기법을 활용하여, 고차원 데이터를 저차원에 효과적으로 매핑하고, 데이터의 확률 밀도 함수를 추정하며, 핵심적인 특징을 추출해낼 수 있는 강력한 비지도 학습 도구입니다. 따라서 AutoEncoder를 작성하기 이전에 다른 내용들을 먼저 알아보겠습니다.

> ## Ⅰ. 차원축소의 개요

차원 축소$$_{Dimensionality-Reduction}$$는 고차원의 데이터에서 중요한 정보를 최대한 보존하면서 데이터의 차원(특징)을 줄이는 과정입니다. 이 방법은 데이터 저장 공간을 절약하고, 처리 속도를 높이며, 오버피팅(overfitting) 문제를 완화하고, 데이터 분석 및 시각화를 용이하게 하는 등 다양한 목적에 사용됩니다. 또한, 차원 축소는 머신러닝 모델의 학습 효율과 성능을 향상시키는 데에도 큰 도움을 줍니다.

예를들면, `(3.5, -1.7, 2.8,-3.5,-1.4)`{: .filepath} &rarr; `(0.32, -1.3, 1.2)`{: .filepath}같은 형태입니다.

차원 축소 기법은 크게 두가지 유형으로 분류 됩니다.

- ### ⅰ. Feature Selection

  특징 선택은 원본 데이터셋에서 중요한 특징들만 선택하여 차원을 줄이는 방법입니다. 이를 통해 불필요한 특징이 제거되어 데이터의 복잡성이 줄어들고, 특징 간의 상관 관계가 줄어 학습의 속도나 과적합을 방지할 수 있습니다. 
  
  아래는 해당 과정의 주요 특징 선택방식들입니다.

  1. Filter Methods(필터 방법)
    
    종속 변수와 독립 변수 사이의 통계적 관계를 기반으로 중요한 변수를 선택합니다. 이 방법은 학습 알고리즘과는 독립적으로 작동하며, 상관 계수(correlation coefficient), 카이제곱 테스트(Chi-squared test), 정보 획득(information gain) 등을 사용할 수 있습니다.
  
  2. Wrapper Methods(래퍼 방법)
    
    독립 변수의 부분 집합을 선택하고, 학습 알고리즘을 사용하여 모델을 학습시킨 후 그 성능을 평가합니다. 그 중 가장 좋은 성능을 보이는 변수들을 선택합니다. 이 방법은 높은 계산 비용을 갖지만, 선택된 변수들이 학습 알고리즘과의 호환성이 높다는 장점이 있습니다. 대표적인 래퍼 방법에는 순차 특징 선택(Sequential Feature Selection)과 재귀 특징 제거(Recursive Feature Elimination)가 있습니다.
   
  3. Embedded Methods(임베디드 방법)
    
    이 방법은 학습 알고리즘의 학습 과정 중에 변수의 중요도를 판단하여 선택합니다. 필터 방법과 래퍼 방법의 장점을 결합한 접근법으로, 계산 비용이 낮으면서도 학습 알고리즘과의 호환성이 높은 변수를 선택할 수 있습니다. 대표적인 임베디드 방법으로는 라쏘 회귀(Lasso Regression)와 랜덤 포레스트(Random Forest)의 변수 중요도(feature importance) 평가가 있습니다.

- ### ⅱ. Feature Extraction

  데이터의 중요한 속성을 추출하는 과정으로, 원본 데이터의 차원을 축소하면서, 데이터의 구조와 패턴을 가장 잘 나타내는 특징을 발견하는 것을 목표로 합니다. 차원 축소 기법들(예: PCA, t-SNE)도 여기에 포함됩니다.

  해당 과정은 크게 선형과 비선형인 두가지 기법으로 나뉘며 아래와 같습니다.
  
  - Linear Dimensionality Reduction
      - Principal Component Analysis(PCA, 주성분 분석)
        - 데이터 분산을 최대한 보존하는 새로운 축을 찾아 데이터를 재구성하는 기법으로 공분산 행렬을 기반으로 하는 고윳값 분해$$_{eigenvalue-decomposition}$$ 및 특이값 분해$$_{singularvalue-decomposition}$$를 사용함.
      - Linear Discriminant Analysis(LDA, 선형 판별 분석)
        - 레이블이 있는 데이터 클래스 간 분산을 최대화하고 클래스 내 분산을 최소화하여 최적의 선형 조합을 찾는 기법
      - etc...
  - Non-Linear Dimensionality Reduction
      - [Autoencoders(오토인코더)](./#ⅴ-autoencoder)
      - t-distributed Stochastic Neighbor Embedding(t-SNE)
        - 고차원 공간에서 데이터 간 지역 구조를 보존하며, 유사한 객체를 저차원 공간으로 사상하는 기법
      - lsomap
        - 모든 점 사이의 두 측점사이의 타원체면을 따라 이루어진 거리인 측지선 거리를 유지하는 더 낮은 차원의 임베딩을 하는 기법
      - locally-linear embedding(LLE)
        - 서로 인접한 데이터들을 데이터의 지역적 선형 구조를 보존(neighborhood-preserving)하면서 고차원인 데이터셋을 저차원으로 매핑하는 기법.
      - etc...

> ### ⅲ. Manifold learning

비선형 차원 축소 기법 중 하나로 선형 차원축소기법의 PCA에서 사용하는 projection(투영)관점에서 설명하자면, 고차원 데이터가 저차원 곡면(manifold)으로 투영되어 더 간단한 구조를 갖는 것을 의미합니다. 해당 저차원 곡면은 원래 고차원 데이터 공간에서의 패턴, 구조 및 중요한 관계를 보존하려고 시도합니다. 

Manifold(다양체)란?
: 수학과 기하학에서 사용되는 용어로, 저차원 공간에서의 지역적 구조를 가지는 고차원 공간의 일부를 의미함. 즉, 고차원 공간의 subspace로 차원 축소를 가능케 함. 예시로 고차원에 공간에 한 점으로 이미지를 매핑시키면 유사한 이미지들이 모여 전체 공간의 부분집합을 이루는데 그것을 매니폴드(Manifold)라고 부른다.(ex. [Cloud Vision API Demo](http://vision-explorer.reactive.ai/#/galaxy?_k=n2cees), t-SNE )


> #### Manifold Hypothesis(Assumption)

- 고차원의 데이터는 밀도는 낮지만, 이들의 집합을 포함하는 저차원의 매니폴드가 있다.
- 이 저차원의 매니폴드를 벗어나는 순간 급격히 밀도는 낮아진다.

해당 가설은 데이터의 차원이 증가할수록 해당 공간의 크기(부피)가 기하급수적으로 증가하기 때문에 데이터의 밀도는 급속도로 희박해진다는것을 반영합니다. 즉, 차원이 증가할수록 데이터의 분포 분석 또는 모델 추정에 필요한 샘플데이터의 개수가 기하급수적으로 증가하게 됩니다. 

따라서, 데이터가 희소해지고 데이터의 분포와 거리 개념이 복잡하게 변동하기 떄문에 고차원 데이터 간의 이웃기반 학습$$_{neighborhood-based-training}$$인 유클리디안 거리는 유의미한 거리 개념이 아닐 가능성이 높습니다.

결과적으로는 Manifold 학습을 통해 sparse한 고차원 데이터에서 발생하는 차원의 저주$$_{Curse-of-Dimensionality}$$를 피할 수 있고, 주요한 특징들로만 구성된 저차원 공간의 데이터를 추출할 수 있습니다.

> ## Ⅱ. Representation Learning

기계 학습에서 말하는 특성 학습의 하위 분야입니다. 
이는 입력 데이터의 특징을 자동으로 학습하여 더 의미 있는 표현으로 변환하는 것을 목표로 합니다. 

Representation Learning은 데이터로부터 내재된 구조와 유용한 특징을 추출하는 과정입니다. 이는 기계 학습 모델이 입력 데이터의 유용한 특징을 더 쉽게 학습하고 이해할 수 있도록 돕는 것입니다. 기존의 기계 학습 방법에서는 사람이 이미 알고 있는 특징을 수동으로 추출하여 사용하는 경우가 많았습니다. 하지만 이러한 방식은 도메인 지식이 요구되고, 문제에 따라 다양한 특징 추출 방법을 개발해야 하는 번거로움에 반해 자동으로 특징을 추출하므로 문제 도메인 지식이 크게 요구되지 않고, 다양한 데이터에서도 적용할 수 있습니다.

이로 인해 표현 학습은 딥러닝 모델의 층 구조와 밀접한 연관이 있습니다. 예를 들어, 사전 학습된 모델에서 각 층은 위쪽(input data와 멀어질수록)으로 갈수록 점차적으로 더 높은 수준의 추상화를 생성합니다. 맨 아래 층은 이미지에서 에지를 찾는 것과 같은 낮은 수준의 특징을 학습하는 반면, 더 높은 층은 이 특징들을 조합하여 고수준의 이미지 개체를 인식하고 구별할 수 있도록 발전합니다. 

표현 학습이 적용되는 다양한 기법으로 앞으로 배울 AutoEncoder나, 단어를 고차원 벡터 공간에서 저차원 벡터 공간으로 표현하는 Word embedding 등이 있습니다.

결론적으로, 표현 학습은 머신러닝 알고리즘이 원본 데이터 공간에서 더 낮은 차원의 잠재 공간(latent space)으로 데이터를 매핑하거나 추상화함으로써 중요한 정보와 패턴인 특징(feature)을 보존하고 불필요한 정보를 제거하여 고수준의 표현으로 변환하는데 중점을 둡니다. 이를 활용하므로서 특징 공학에 대한 의존도가 낮아지고 데이터의 내재된 패턴을 더 잘 이해하여 관련 문제에 대해서도 적용하는 일반화 성능의 향상으로 이어질 수 있습니다.

> ## Ⅲ. Efficient Coding Learning

인지 과학과 기계 학습에서 주로 연구되는 개념 중 하나로 주어진 데이터와 자원(ex. 뉴런의 개수, 데이터 용량 등)을 효율적으로 나타내기 위해 코드로 압축하는 방법을 극대화하는 방법을 연구하는 분야입니다. 이를 통해 중복을 최소화하고 데이터의 관련 정보만을 유지하는데 집중합니다. 이 과정에서 발생하는 희소성 및 압축 방법론에서 인공신경망 및 효율적인 표현을 발견하게 됩니다. 

이 개념은 1961년 Horace Barlow에 의해 처음 소개되었으며, 생물학적인 시스템에서 영감을 받은 개념입니다. 생물학적 뇌의 인코딩 프로세스와 통계 학습이 어떻게 상호 작용하는지를 이해하는 데 도움이 됩니다. 

예를 들어, 인간의 시각 시스템은 시각 자극을 효율적으로 표현하기 위해 중요한 정보를 추출하고 불필요한 세부 정보를 제거하여 처리합니다. 이러한 생물학적 시스템의 원리와 비슷하게, Efficient Coding Learning은 입력 데이터의 특징을 학습하여 중요한 정보를 효율적으로 표현하는 방법을 찾습니다. 

이 개념을 통해 뇌의 인코딩 프로세스에 대한 이해를 돕고, 여러 가지 에너지 효과적인 표현 학습 기술을 적용하여 입력 데이터의 중요한 특징만을 학습하게 되며 잡음과 상관 없는 불필요한 정보는 제외됩니다. 이러한 접근 방식은 신경망 모델의 일반화 성능을 향상시키고, 고차원 데이터를 처리하는 데 있어 더 빠른 학습과 더 낮은 에너지 소모를 가능하게 합니다. 

결론적으로, efficient coding learning은 신경망의 구조와 학습 접근법을 최적화하여 처리 성능과 에너지 소모에서 균형을 이루기 위한 머신 러닝에 대한 학습 이론입니다. 이를 통해 더 효과적인 머신 러닝 모델 및 뇌의 인코딩 프로세스에 대한 이해를 도모합니다.

> ## Ⅳ. 최대우도밀도추정(MLE)

최대 우도 밀도 추정$$_{Maximum-Likelihood-Density-Estimation}$$은 주어진 데이터를 사용하여 확률밀도함수(PDF$$_{probability-density-function}$$), 즉 확률 분포의 모수$$_{parameters}$$를 추정하는 통계적 방법입니다. 이 방법은 관측된 데이터가 확률 분포의 특정 모수를 따른다고 가정하고, 확률 분포의 모수를 추정하여 최대한 관측 데이터와 일치하는 확률 밀도 함수를 찾는 것을 목표로 합니다. 

![density estimation](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/44bb59fc-b3af-4029-b1d3-9c0d3c3fd7a6)
_출처 : https://www.slideshare.net/NaverEngineering/ss-96581209_ 

최대 우도 밀도 추정의 기본 개념은 우도(likelihood) 함수 설정을 통해 관측 데이터에 대한 확률 밀도 함수의 모수를 추정하는 것(미분 및 최적화 기법 사용)입니다. 우도는 관측된 데이터가 주어졌을 때, 이 데이터가 특정 모수를 가지는 확률 분포에서 생성될 확률을 나타냅니다. 따라서 최대 우도 추정법$$_{MLE}$$은 이 우도를 최대화하는 모수를 찾는 것을 목표로 합니다. 

다시말하자면, MLE을 통해 관측된 데이터에 대한 확률밀도함수의 모수를 도출하여 데이터 생성에 가장 적합한 확률분포를 찾는 것이 최대 우도 밀도 추정의 핵심입니다. 

> ## Ⅴ. AutoEncoder

오토인코더(Autoencoder)는 비지도 학습 방식의 인공 신경망으로 우리가 앞서 배운 Restricted Boltzmann Machine$$_{RBM}$$은 AutoEncoder$$_{AE}$$와 유사한 목표를 갖고 있습니다. 히든 레이어에서 데이터에 관한 latent factor들을 얻어내는 것이 목표로 데이터를 압축하고, 다시 압축 해제하는 과정을 거치게 만드는 신경망입니다. 이 구조는 입력 데이터의 차원 축소(representation learning)와 노이즈 제거를 위한 효율적인 데이터 인코딩을 수행할 수 있습니다. 즉, 오토인코더는 데이터의 은닉적 표현(hidden representation)을 학습하고 복원된 출력을 생성합니다. 

따라서, 오토인코더는 input 데이터의 feature를 추출하는 차원 축소 기법이나 network parameter 초기화, 사전학습 등에 많이 사용됩니다. 이때는 batch-norm, xavier initialization과 같은 기법이 없었다고 합니다.

오토인코더는 두가지 유형의 구조가 존재합니다.

- #### Undercomplete AutoEncoder
  
  은닉층의 뉴런 수가 입력층 및 출력층 보다 작은 구조로 압축된 표현을 학습하여 원본의 특성을 보존하면서 데이터의 차원을 축소에 유용합니다.

- #### Overcomplete AutoEncoder
  
  은닉층의 뉴런 수가 입력층 및 출력층 보다 큰 구조로 인공신경망이 데이터의 압축된 표현을 찾지 못하고 단순히 입력을 복사하는 일명 identity function을 학습할 수 있습니다. 이는 여러 변수들로 구성된 지수

오토인코더 구조는 아래와 같은 세 부분으로 구성됩니다.

![img](https://github.com/eastk1te/P.T/assets/77319450/06fddeb7-8ffe-44d9-8896-6fa046a761f6)
_Figure 1 : input data x를 encoder network에 통과시켜 압축된 latent z를 얻고, 압축된 z vector로부터 x와 같은 크기의 output data y를 얻습니다._  

> ### ⅰ. Encoder

$$z = h(x), h(X) = W_ex + b_e$$

입력 데이터를 받아들이고, 은닉 레이어를 통해 저차원의 은닉 표현으로 변환합니다. 적어도 input data에 관해서는 잘 복원하고, 최소한의 성능을 보장합니다.

인코더의 역할은 주어진 고차원의 데이터를 낮은 차원의 벡터$$_{representation vector}$$로 압축시켜 변환합니다. 데이터가 갖는 특성들을 작은 공간안에 우겨넣으려다보니 상징적인 특성들로 구성된 벡터공간인 latent space를 생성해야하는것입니다. "차원 감소"와 비슷할 수 있으나 "투영"이 아닌 비선형적인 방식의 차원감소가 이루어지기에 압축이라는 말을 사용합니다.

> ### ⅱ. Latent Code
  
이렇게 입력값 x로부터 추출된 특징을 latent code 또는 은닉 표현$$_{Hidden-representation}$$이라고 합니다. 이는 인코더를 통해 얻어진 입력 데이터의 압축된 표현으로 중요한 특성을 포착하며, 원래 데이터의 차원보다 낮은 차원의 공간에 위치합니다.

> ### ⅲ. Decoder

$$y = g(h(x)), g(x)=W_dz+B_d$$

디코더$$_{Decoder}$$는 은닉 표현을 입력으로 받아, 원본 입력 데이터와 유사한 높은 차원의 복원된 출력 데이터를 생성합니다. latent space안에 있는 representation vector를 임의로 주었을떄, 그것을 갖고있는 원래의 의미를 "압축해제"하여 원본 사이즈의 데이터 형태로 복원시키는 역할입니다. 이는 최소한 학습 데이터를 만들어 줄 수 있다는 의미를 가집니다.

> ### ⅳ. Loss
  
$$Loss = y - \hat{y}$$

해당 과정에서는 오토인코더의 입력과 출력의 크기가 같아하는데, 그 이유는 원본 입력 데이터와 복원된 출력 데이터 간의 차이를 최소화할 목적을 가지고 있기 때문입니다.

Cross-Entropy가 MSE보다 더 나은 결과를 제공하는 이유
: CE는 출력과 실제 값의 차이에 따라 기울기가 확장되거나 축소되지 않습니다. 따라서 CE가 MSE보다 기울기 소실 문제에 대해 더 자유롭습니다. 추가적으로 ReLU$$_{Rectified-Linear-Unit}$$은 미분값이 0 또는 1로 기울기 소실 문제와 활성화 값의 빠른 수렴을 돕는 훌륭한 활성화 함수입니다.


즉, 오토인코더의 학습 방법은 `비지도 학습`$$_{Unsupervised-learning}$$을 따르며, loss는 `negative ML`$$_{ML-density-estimation}$$로 해석된다. 이렇게 구성된 오토인코더의 인코더는 `차원 축소`$$_{Manifold-learning}$$ 역할을 수행하며, 디코더는 `생성 모델`$$_{Generative-model-learning}$$의 역할을 한다.


> ### ⅴ. 왜 굳이 압축하고 풀어낼까?

오토인코더의 목표는 원본 입력 데이터를 압축된 은닉 표현 형태로 인코딩하고, 이것을 다시 복원하여 원본 입력 데이터와 유사한 데이터 출력값을 복구하는 것입니다. 이 과정에서 손실(minimization loss)는 일반적으로 입력 데이터와 복원된 출력 데이터 간의 차이를 나타내는 값입니다. 따라서 출력 데이터 크기가 입력 데이터 크기와 동일해야 이러한 재구성 손실을 계산할 수 있으며, 모델 학습을 통해 원본 데이터를 잘 인코딩하고 복원할 수 있습니다.


1. 뉴럴넷이 압축하는 방식을 배우면 비선형적인 차원감소를 수행하고, 이를 통해 더 낮은 차원으로 표현
2. 새로운 데이터를 생성할떄 힘을 발휘할 수 있기 때문


인코더를 통해 더 낮은 차원의 압축된 데이터를 얻는다고 하자. 디코더는 이것을 거꾸로 압축된 낮은 차원의 벡터 공간에서 임의의 점을 선정해 다시 압축해제를 시키면 지금껏 보지 못한 데이터가 생성될 수 있따.

딥러닝의 핵심적인 내용은 단순히 "NN의 층이 깊다" 정도에서 끝나는 것이 아니라, "층이 깊어질때 무슨 효과가있는가"이다. 층이 더 깊어질수록(즉, input layer에서 멀어질수록) 추상적인 feature를 추출할 수 있다고 알려짐.

> ### ⅵ. Stacking AutoEncoder for pre-training

우리는 이제 어떻게 하면 좀 더 잘 학습시키는가? 라는 질문에 도달 한다. 오토인코더는 적어도 입력값에 대해서는 복원을 잘한다는 특징을 활용해 학습 데이터 셋에 있는 입력 데이터를 잘 표현하는 가중치를 학습시킬 수 있습니다. 따라서, 오토인코더를 Stack의 형태로 쌓아 올려 더 깊은 층을 만드는 것을 Stacked AutoEncoder라고 하며, 이는 우리가 앞서 배운 [DBN](../RBM/#ⅱ-dbndeep-belief-net)의 형태와 유사합니다. 이러한 작업이 끝난 후 추상적인 특성들을 fine-tuning하면 깊은 뉴럴 네트워크를 훈련시킬 수 있을 것이라는 생각이 오늘날의 딥러닝 알고리즘을 있게 했습니다.

> ### ⅶ. DAE(Denosing AE)

DAE는 복원 능력을 더 강화하기 위해 기본적인 AE의 학습 방법을 조금 변형한 것입니다.

핵심 논문은 "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion"입니다. 이 논문은 2010년 JMLR(Journal of Machine Learning Research)에 발표되었습니다. 

![1](https://github.com/eastk1te/P.T/assets/77319450/be4956c4-c034-459e-babf-a84ddbba2937)
_Figure 1 : x = input data,  $\tilde{x}$ = noise가 추가된 데이터, $h(\cdot)$ = Encoder, z = latent vector ,$g(\cdot)$ = Decoder, Loss = $L(x,y)$로 노이즈가 포함된 데이터가 아닌 원본 데이터와 출력 값으로 계산._

잡음이 없는 원 데이터 x에 잡을을 가하여 잡음이 있는 데이터 $\tilde{x}$를 만들어 냅니다. 그 후 잡음이 있는 데이터를 가하여 출력 y가 잡음이 있는 영상$\tilde{x}$가 아닌 원 영상 x에 가까워지도록 학습을 시킵니다.

즉, DAE는 노이즈가 있는 입력 데이터를 받아서 원래의 깨끗한 입력 데이터를 복원하는 능력을 활용합니다. 이렇게 DAE는 지역적인 잡음 제거 기준을 사용하여 노이즈가 있는 입력의 특성 표현을 효과적으로 압축하고 복원하는 강인한 표현력을 갖게 됩니다.

![manifold관점](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/12056786-1b8e-45b9-a0a9-eabd13484073)
_Figure 2 : Manifold learning의 관점에서 원 데이터는 저차원 공간인 실선에 가깝지만 노이즈를 추가한 데이터는 실선에 멀어집니다. 따라서 모델은 노이즈를 추가한 데이터를 실선에 "Project"하는 방향으로 학습됩니다._


논문에서는 잡음이 없는 경우보다 잡음의 비율이 높아질 수록 필터는 local한 특징보다 global한 특징을 추출하게 된다는 것이 나왔는데, 잡음이 많아지면서 DAE의 학습과정이 결과적으로 분명한 특징을 학습하도록 만들어주는 것으로 해석된다고 합니다.


> ### ⅷ. Sparse Autoencoder

Sparse Autoencoder (SAE)에 대한 개념과 아이디어는 여러 연구자들에 의해 개발되었지만, Andrew Ng의 "Sparse Autoencoder" 논문에서 상세하게 설명되었습니다. SAE는 입력 데이터의 희소 표현을 학습하기 위해 훈련하는 인공신경망입니다. 이를 위해 희소성 패널티를 추가하여 학습 중에 많은 히든 유닛이 활성화되지 않고 소수의 유닛만 활성화되도록 유도합니다. 이러한 과정을 통해, 중요한 정보가 더 간결하게 압축된 표현으로 학습됩니다.

$$L(\theta) = MSE(x,y) + \lambda * \sum_{j=1}^{D_y}KL(\rho || \hat{\rho}_j )$$

- $x$ : 입력 데이터
- $y$ : 재구성된 출력
- $MSE(x,y)$ : 재구성 손실함수로 목적에 따라 CE등으로 변경될 수 있지만 기본적으로 Mean Squared Error(MSE)를 사용합니다. 
- $\lambda$ : 희소성 규제 항 가중치
- $\rho$ : 희소성 목표 파라미터로 뉴런이 활성화되는 정도를 제어하는 역할을 합니다.(일반적으로 0과 1 사이의 작은 값으로 설정)
- $\hat{\rho}_j$ : 중간층 unit의 평균 활성도의 추정치이다.
- $\sum_{j=1}^{D_y}KL(\rho\|\|\hat{\rho}_j)$ : 
  
  희소성 규제 항$$_{Sparsity-Regularization-Term}$$으로서, L1 규제 또는 KLD$$_{Kullback-Leibler-divergence}$$ 등을 사용하여 정의됩니다. 이 항은 재구성 손실에 추가되어 희소성을 강조하고, 희소성 목표($\rho$)에 비해 실제 활성화 정도와의 차이를 계산합니다. 만약 둘의 분포가 같다면 KL값은 0이 되고, 다르면 0보다 큰 값을 갖게 된다. 
  
  이를 통해 재구성 손실과 희소성 규제 항을 최적화함으로써, 입력 데이터를 효과적으로 재구성하는 모델을 학습할 수 있습니다.

Kullback-Leibler-divergence(KLD)이란?
: 두 확률분포 간의 차이를 측정하는 지표입니다. 흔히 정보 이론에서 사용되며, 상대 엔트로피(relative entropy)라고도 부릅니다. KLD는 두 확률분포 사이의 정보 손실을 나타내는 값으로, 한 확률분포가 다른 확률분포를 얼마나 잘 설명하는지를 계산합니다. 수학적으로, 두 확률분포 P와 Q에 대해 P(x)와 Q(x)는 각각 x에 상응하는 확률을 나타내고 다음과 같이 정의됩니다. $KLD(P \|\| Q) = ∑ P(x) * log(P(x) / Q(x))$. KLD는 양의 값이며, 두 확률분포가 동일할 경우 0이 됩니다. KLD는 비대칭$$_{asymmetric}$$인지라, $KLD(P \|\| Q)$와 $KLD(Q \|\| P)$의 값이 다릅니다.

> #### 학습하기도 바쁜데, 왜 이러한 행동을 해야 할까요?!

학습 과정에서 뉴런들에게 특정한 작업을 수행하도록 지시하는 것은 매우 중요한데, 그 이유는 뉴런의 활성화가 주요한 목표 예측에 도움이 되는 중요한 정보를 제공하기 때문입니다. 기본적으로, 우리는 뉴런들에게 특별한 역할을 부여함으로써, 그들이 학습 과정에서 효과적으로 참여하고 도움이 되도록 유도하고 있습니다. 이렇게 함으로써, 학습하는 동안 모델이 지식을 보다 정교하게 표현할 수 있게 됩니다.

이를 확인하는 한 가지 방법은 ρ로 정의된 Sparsity 파라미터를 사용하는 것이다.

우리는 ρ이 0이 되기를 원합니다. 따라서, $\rho_I$값이 $\rho$와 비슷해야 하므로 희소성 규제 항을 KLD를 활용하여 계산합니다.

뉴런이 몇 번이나 활성화된 지를 어떻게 측정할 수 있는가? 다음 방정식을 통해 계산할 수 있다.

$$\sum_{j=1}^{D_y}KL(\rho \|\| \hat{\rho}_j ) = \sum_{j=1}^{D_y}\rho log(\frac{\rho}{\rho_j}) + (1-\rho)log(\frac{(1-\rho)}{(1-\rho_I)})$$

$$where, \rho_I = \sum(\sigma(X_i) / m)$$

여기서 $\rho_I$는 모든 뉴런들을 합쳐 평균을 얻은 것이라고 하고 $\sigma(\cdot)$는 activation function을 나태내고, $\sigma(\cdot)$가 양수라면 활성화 된 것 입니다. 우리는 이 항이 $\rho$에 의해 주어진 sparsity 매개변수와 최대한 같기를 원합니다.

이렇게 Sparse AutoEncoder를 통해 sparse 한 노드(0 이 많은)들을 만들고, 그 중에서 0과 가까운 값들은 전부 0으로 보내버리고 0 이 아닌값들만 사용하여 네트워크 학습을 진행합니다. 조금 더 직관적으로는 대부분의 시간 동안 뉴런들이 활동하지 못하게 하는 것입니다. 이는 규제 기법을 통해 coding층의 훈련시마다 뉴런 개수를 제한함으로서 각 뉴런들이 더 유용한 특성을 학습하여 coding을 만들어 낼 수 있도록 하는 기법입니다. 예를들어 dropout에서 일부 뉴런을 의도적으로 훈련에서 누락시켜 나머지 뉴런들이 더 유용한 특성을 학습하도록 하게 만드는 것과 비슷합니다.



> ### ⅸ. CAE(Contractive AE)

주요 논문은 "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction" 입니다. 이 논문은 2011년 International Conference on Machine learning (ICML)에 발표되었습니다. CAE의 목적 함수는 정규화 항을 추가함으로써 모델이 작은 변화를 무시하는 표현을 학습하는 데 초점을 맞춥니다. 또한, 본 논문은 수많은 실험을 통해 CAE가 입력 공간에서의 전이에 불변하는 표현을 학습한다는 것을 입증합니다. 따라서 원본 데이터에 대해 민감하지 않은 견고한 피쳐를 추출하는 데 어떻게 사용할 수 있는지에 대해 설명하였습니다. 


![2](https://github.com/eastk1te/P.T/assets/77319450/f53f71de-e48f-492f-b884-affd00b911e1)


CAE는 DAE와 같이 작은 변화에 강건한 모델을 학습하는 것이 목적으로 Encoder가 입력 데이터의 작은 변화에 저항하도록 하는데 중점을 두고 있습니다. 즉, 인코더가 디코더에서 재구성할때 자코비안 행렬을 손실함수에 추가하여 중요하지 않은 입력의 변화를 무시하도록 하여 특징을 추출할때 작은변화에 덜민감하도록 중점을 둔다는 의미입니다.

$$argmin_{enc,dec}E[loss(x,dec\cdot enc(x)] + \lambda||\nabla_xenc(x)||_2^2$$

$$L(\theta) = R(x, g(f(x))) + \lambda \cdot C(x, h, J)$$
- $L(\theta)$: 전체 손실 함수
- $x$: 입력 데이터
- $g(h(x))$ : 재구성 에러로 인코딩 후 디코딩된 재구성 데이터 
- $R(x, g(f(x)))$ : 재구성 손실 (예: 평균 제곱 오차(MSE) 또는 교차 엔트로피)
- $\lambda$: 하이퍼 파라미터로 입력과 재구성의 민감도를 조절하는 가중치를 설정합니다.
- $C(x, h, J)$ : 
  
  목적 함수의 최적화 과정 중 입력 데이터와 재구성에 관한 민감도를 제한하는 컨트랙티브 손실 항$$_{Contractive-Regularization}$$입니다. $J$는 입력 데이터에 따라 인코더 출력의 변화를 설명하는 야코비안 행렬이며 $h$는 인코딩 된 표현입니다. 해당 항은 feature space가 훈련 데이터의 이웃으로 수렴하도록 매핑을 격려한다.
  
  ![3](https://github.com/eastk1te/P.T/assets/77319450/9033acf2-8c77-4a7e-8265-5b4ddbcc9586)
  _출처 : https://www.slideshare.net/NaverEngineering/ss-96581209_

  종종 이 항을 $||J||^2_F$으로 표현하며, 여기서 $||\cdot||_F$는 프로베니우스 노름(Frobenius norm)입니다.

  ![CAE](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/2483671d-6c4b-46a9-b74e-fa0c71b98e0d)
  _출처 : http://dmqm.korea.ac.kr/uploads/seminar/DMQAseminar_210813.pdf_

  대부분의 원본 데이터 입력 변화에 따른 방향에 대하여 표현이 지역적으로 불변하는 것을 장점으로 가지고 있다.

> ## Ⅵ. VAE(Variational-AutoEncoder)

우선 VAE는 2013년에 작성된 "Auto-Encoding Variational Bayes"에서 이 개념이 소개되었으며 Generative modeling으로써 오토인코더 정의와 다르게 생성모델을 학습하는 과정에서 모델의 구조가 오토인코더와 유사합니다. 따라서 VAE는 AE와는 기원이 다르며 이름과 구조가 비슷하여 혼당하기 쉬움을 인지해야합니다.

VAE는 Input X를 잘 설명하는 feature를 추출하여 Latent vector z에 담고, 이 Latent vector z를 통해 X와 유사하지만 완전히 새로운 데이터를 생성하는 것을 목표로 합니다. 이때 각 feature가 가우시안 분포를 따른다고 가정하고 latent z는 각 feature의 평균과 분산값을 나타냅니다. 즉, 하나의 숫자로 나타내는 것이 아니라 가우시안 확률분포에 기반한 확률값으로 나타낸다.

$$p_{\theta}(x|z) \space where, z \sim (\mu, \sigma^2)$$

![VAE](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/aa70628f-6043-48ad-8d85-0b09ed846b02)
_Figure 1 : VAE와 AE의 latent space를 시각화해보면 차이를 발견할 수있다._

> ### ⅰ.학습


VAE의 학습 목표는 원본 데이터 x와 생성된 데이터 $g(h(x))$간의 차이를 최소화하는 것입니다. 즉, 변분 추론(Variational Inference)을 사용하여 모델 파라미터를 최적화합니다. 

$$\underset{\theta}{argmax} \space p_{\theta}(x) = \int p_\theta(z)p_\theta (x|z)dz$$

손실함수는 아래와 같이 두가지 항목으로 구성됩니다. 이 두 손실함수 항목을 최소화하여 VAE를 업데이트하고 최적화합니다.

$$L(x^i, \theta , \phi) = - E_z[logp_\theta (x^i|z)] + D_{KL}(q_\phi(z|x^i)||p_\theta (z))$$

$E_z[logp_\theta (x^i\|z)]$ 는 복원 손실((reconstruct error))로 입력값과 출력값의 차이이고, $D_{KL}(q_\phi(z|x^i)||p_\theta (z))$는 KLD 손실로 인코더의 출력 중 평균 및 분산과 사전 정의된 확률분포(ex. 가우시안 분포) 간의 차이를 측정하는 항목으로 잠재 공간에서의 생성된 사후확률분포에 근접하게 만들어야합니다.


> ### ⅱ. CVAE(Conditional VAE)

이름에서 알 수 있듯이 조건을 부여합니다.

앞서 나온 VAE에서는 latent space가 임의로 sampling되면서 어떤 숫자가 샘플링될지 제어할수없었습니다. 하지만 CVAE의 핵심 아이디어로 인코더와 디코더에 조건 정보(일반적으로 원핫 벡터나 해당하는 특성 벡터)를 포함시키면 인코더와 디코더 모두 원하는 특성에 따라 데이터를 생성할 수 있습니다.

![4](https://github.com/eastk1te/P.T/assets/77319450/39aa257b-263a-4d15-a73c-c30c5f145aa6)
_Figure 2 : CVAE에서 label 정보를 알고있으면 Encoder와 Decoder에 적용하는 구조(M2)_

![5](https://github.com/eastk1te/P.T/assets/77319450/7059e85a-1208-4482-aabd-e21747b17326)
_Figure 3 : Figure 2에서 식을 풀어보면 ELBO$_{Evidence Lower Bound}식이 똑같이 유지됨을 보여$_

![6](https://github.com/eastk1te/P.T/assets/77319450/dce942ce-238f-46c6-896f-773df2c0a1aa)
_Figure 4 : Label을 모르는 Unsupervise learning에 대한 구조들로 좌 :  y를 추정하는 맨 좌측의 네트워크를 구성하는 M2 구조, 우 : 인코더로 학습 한 이후 y를 추정하는 구조를 따로 만드는 M3구_

> ### ⅲ. AAE(Adversarial AE)

오토인코더와 생성적 적대 신경망(GAN)의 개념을 결합한 생성 모델입니다. AAE는 효율적인 특성 추출과 샘플 생성을 목표로 하는 비지도 학습 방법입니다. 

AAE의 핵심 아이디어는 잠재 변수의 사전 확률 분포와 근사 사후 확률 분포 간의 약점을 극복하고, 적대적 학습을 사용하여 잠재 공간의 더 정교한 사후 분포를 학습하는 것입니다. 

![7](https://github.com/eastk1te/P.T/assets/77319450/36058c52-02c0-40a3-9706-caf2f9c1be1b)
_Figure 5 : KLD부분이 sample들의 분포가 target의 분포와 같아지게 만드는 GAN과 유사하다. 따라서, KLD대신 GAN Loss를 사요하면 임의의 함수에 대해서도 가능하다는 것을 나타냄._

![8](https://github.com/eastk1te/P.T/assets/77319450/961e8dc2-060d-4ffb-bc78-847f6ac9621f)
_Figure 6 : Autoencoder에서 Encoder 부분이 Generator 역할로 가짜 샘플을 만들어내고, Target 분포에서 진짜 샘플을 넣고 판별을 하면서 학습을 진행함._

오토인코더의 잠재 변수 z에 생성자 함수를 적용하여, 원하는 특성을 포함하는 새로운 데이터를 생성합니다. 판별자는 생성된 데이터와 실제 데이터를 구별하려고 시도하며, 이 과정에서 오토인코더의 성능이 개선됩니다. 

![9](https://github.com/eastk1te/P.T/assets/77319450/21c4790a-ba48-4493-b91d-1d6039fee9af)
_Figure 7 : VAE에서의 KLD 항을 사용하지않고, GAN Loss를 대신 사용하여 나타낸 식._

![10](https://github.com/eastk1te/P.T/assets/77319450/0519357d-4072-4298-8ae6-ebf9a6ee5ec3)
_Figure 8 : GAN은 생성자와 판별자의 목적이 달라 따로 학습을 시킨다. 따라서 AAE에서도 위 세개의 부분을 번갈아가면서 학습시킨다._

AAE는 다양한 종류의 데이터(이미지, 텍스트, 음성 등)에 사용할 수 있으며, 특성 학습과 데이터 생성 작업에 특히 효과적입니다. 또한, 풍부한 생성 콘텐트를 만드는 데 사용되거나 GAN과 함께 사용되어 더 정밀한 생성 결과를 도출할 수 있습니다.


> ## Ⅶ. REFERENCES

1. [AutoEncoder의 모든것](https://deepinsight.tistory.com/126)
2. [VAE vs AE](https://bcho.tistory.com/1326)
3. [오토인코더 - 공돌이의 수학정리노트](https://angeloyeo.github.io/2020/10/10/autoencoder.html)
4. [오토인코더 자료 모음](https://subinium.github.io/VAE-AE/#1-%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94%EC%9D%98-%EB%AA%A8%EB%93%A0-%EA%B2%83)
5. [오토인코더의 모든것 - Naver Tech](https://d2.naver.com/news/0956269)
6. ["Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion"](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)
7. [CAE - DMQA open seminar](http://dmqm.korea.ac.kr/activity/seminar/330)
8. ["Contractive Auto-Encoders: Explicit Invariance During Feature Extraction"](https://icml.cc/2011/papers/455_icmlpaper.pdf)
9.  [VAE - wikidocs](https://wikidocs.net/152474)
10. ["Sparse Autoencoder", Andrew Ng](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
11. [Sparse Autoencoder - 대소기의 블로구](https://soki.tistory.com/64)
12. [여러가지 구조의 Autoencoders](https://data-newbie.tistory.com/180)
13. [CVAE](https://chickencat-jjanga.tistory.com/4)

<br><br>
---