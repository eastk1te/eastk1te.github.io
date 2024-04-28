---
title: '[Paper]Transformer(1)'
author: east
date: 2023-07-05 00:00:00 +09:00
categories: [Paper, NLP]
tags: [Paper, NLP, Seq2Seq, Bahdanau Att., Luong Att.]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
---

트랜스포머(Transformer)는 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로 기존의 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 어텐션(Attention)만으로 구현한 모델입니다. 이 모델은 RNN을 사용하지 않고, 인코더-디코더 구조를 설계하였음에도 번역 성능에서도 RNN보다 우수한 성능을 보여주었습니다. 해당 논문, 우리는 고정된 길이의 fixed-length vector가 기본적인 인코더-디코더 구조의 성능 향상에서 병목현상을 일으킨다고 추측했고, 해당 context vector c의 확장을 제안했습니다.

따라서, 트랜스포머를 배우기 이전에 Sequence-to-Sequence 부터 시작하겠습니다.

> ## Ⅰ. Seq2Seq

Sequence-to-Sequence 모델은 2014년에 발표된 ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) 논문에서 처음으로 이 개념이 소개되었습니다. 

논문에서는 입력과 출력 사이의 복잡한 구문분석이나 개체명인식과 같은 중간단계를 최소화하여 입력 시퀀스를 출력 시퀀스로 바로 출력하는 end-to-end의 접근법을 사용했습니다. 기존의 RNN에서 sequence 입력 데이터를 입력과 출력 시퀀스의 다른 크기를 적용하기 복잡한 문제와 고질적인 기울기 소실로 인한 장기 의존성 문제로 학습하기 어려운 문제에서 벗어난 LSTM을 활용하여 아래와 같이 두개의 딥 러닝 모델인 인코더와 디코더로 구성됩니다. 해당 모델은 Many-to-Many 형태의 RNN 기반의 구조로 Many-to-one은 Encoder, one-to-Many를 Decoder이고 그 사이의 있는 vector를 Context Vector라고 부릅니다.

![image](https://wikidocs.net/images/page/24996/%EB%8B%A8%EC%96%B4%ED%86%A0%ED%81%B0%EB%93%A4%EC%9D%B4.PNG)
_Figure 1 : encoder-decoder_


> ### ⅰ. Encoder

우선 Figure 1에서 인코더에 들어가는 입력시퀀스인 "I am a student"라는 문장을 embedding을 통해 (시퀀스 길이, 임베딩 차원)의 vector로 변환이 이루어집니다.

여기서 `인코더`$$_{Encoder}$$는 LSTM cell이라는 unit들로 구성된 네트워크로 구성이 되어있으며, 해당 unit들은 각각의 단어들과 이전의 은닉 상태들을 입력으로 고정된 크기의 은닉상태$$_{hidden-state}$$를 출력합니다. 즉, "I", "am", "a", "student"라는 단어들은 임베딩을 통해 각각 (1, embedding)의 크기가 되고 LSTM cell을 통해 (hidden_dim, 1)의 크기로 나오게 됩니다.

> ### ⅱ. Context Vector

이렇게 인코더에서 모든 단어들을 순차적으로 입력받은 뒤에 마지막 unit에서 모든 단어 정보들을 압축해서 하나의 고정된 크기$$_{fixed-length}$$의 `은닉상태 벡터인 Context Vector`로 만들어냅니다. 해당 Vector는 입력 문장의 모든 단어 토큰들의 정보를 요약해서 담고있는 압축된 정보라고 할 수 있습니다.

> ### ⅲ. Decoder

이제 `디코더`$$_{Decoder}$$에서 압축된 정보를 출력 sequence로 만들어 내는 역할을 수행해야 합니다. 디코더는 인코더와 같이 LSTM cell인 RNN기반의 unit들로 구성된 네트워크이며 인코더로 부터 전달받은 Context vector를 초기 은닉 상태로 설정하고 출력 시퀀스 생성을 알리는 시작 토큰 \<sos\>$$_{start-of-sequence}$$를 입력값으로 초기 unit이 수행됩니다. 이렇게 수행된 LSTM 계층 unit을 통과하여 출력 어휘의 크기와 동일한 차원으로 변환하는 아핀 변환$$_{Affine-transformation}$$ 계층을 지나 Softmax함수를 활용하여 확률분포의 형태로 변환하는 softmax계층을 통해 모든 단어들의 확률 분포를 얻게 됩니다.

이렇게 나온 확률 분포를 기반으로 선택 될 수 있는 모든 단어들로부터 하나의 단어를 출력값으로 정해야 하는데, 확률이 가장 높은 단어를 사용하는 결정적인$$_{Deterministic}$$한 방법이나 확률 분포에 따라 생성하는 확률적인$$_{Stochastic}$$한 방법으로 샘플링하는 방법들을 사용합니다. 

아핀 변환이란?
: Ax + b의 형태로 기하학적 변환 중 하나로 같은 공간 내에서 점, 직선, 평면들을 이동시키는 변환으로 선형변환과 편향벡터의 덧셈 등과 같은 계산이다.


이렇게 나온 출력 단어와 은닉 상태를 다음 디코더의 unit으로 보내는 방식으로 디코더의 unit들이 수행됩니다. 해당 부분은 inference 파트에서 진행이 되는 내용이고, 학습 파트에서는 이전 단계에서 나온 출력값이 아닌 target sequence의 해당하는 값을 넣는 교사 강요$$_{teacher-forcing}$$ 전략을 사용하여 학습 속도를 높이고 빠른 수렴을 도모합니다. 추가적으로 디코더가 출력 시퀀스를 무한대로 생성하면 안되기에 종결하는 \<eos\>$$_{end-of-sequence}$$ 토큰을 학습데이터에 적용하여 문장 생성을 종결하도록 학습해야합니다. 

> ### ⅳ. 구현

```python
seq2seq 모델 구현

내가 구현한건 LSTM cell이고,
LSTM 네트워크는 내가 말한거처럼 들어감.

LSTM cell.
LSTM network
encoder
decoder

input
encoder
hidden
context
decoder shape 그리


https://blog.naver.com/sooftware/221784419691

```

> ### ⅴ. Seq2Seq의 한계

인코더는 입력 시퀀스를 하나의 벡터표현으로 압축하고, 디코더는 이 벡터 표현을 통해서 출력 시퀀스를 만들어냈습니다. 하지만 이러한 구조는 인코더가 입력 시퀀스를 하나의 벡터로 압축하는 과정에서 `입력 시퀀스의 정보가 일부 손실된다는 단점`{: .filepath}이 있었고, 이를 해결하기 위해 어텐션 매커니즘이 나오게 되었습니다. 

`어텐션 매커니즘`$$_{attention-mechanism}$$은 위에서 나온 Sequence-to-Sequence 모델의 성능을 향상시키기 위해 고안된 기술로 `디코더가 예측을 생성할 떄, 인코더의 모든 입력 벡터에 가중치를 부여함으로써 문맥 정보를 더 잘 반영하도록 돕는 방식`{: .filepath}입니다. 해당 가중치를 구하는 스코어 계산 방식의 차이에 따라 다양한 종류가 있지만, 일반적으로 주로 사용되는 몇 가지 기본적인 유형에 대해 다루어 보겠습니다.


> ## Ⅱ. Bahdanau Attention

2014년에 나온 
["NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE"](https://arxiv.org/pdf/1409.0473.pdf) 논문의 공동저자의 이름에서 비롯되어 Bahdanau Attention이라고 불려지는 모델은 어텐션 매커니즘이 처음 적용된 논문으로 현재의 어텐션 매커니즘들의 시초라고 할 수 있습니다.

> ### ⅰ. Abstract

전통적인 통계기반의 기계번역과 다르게 신경망 기계번역은 기계번역의 성능을 최대화하게 결합하는 단순신경망을 구축하는것입니다. 해당 논문에서는 기존의 encoder-decoder 구조에서 고정된 길이의 vector를 사용하게되면 병목 현상$$_{bottlenect}$$이 발생하는 문제를 풀고자 명시적인 엄격한 세그먼트(즉, 고정된 크기의 vector)없이 자동적으로 예측하려는 목표 단어와 연관된 원문의 부분을 찾는 방법을 제안했습니다. 


병목현상(bottleneck)이란?
: 일련의 과정에서 한 가지 요소가 전체 성능이나 효율성을 저하시키는 현상을 의미합니다. 이는 해당 요소가 프로세스나 시스템의 성능에 제한을 가한다고 판단되는 경우 사용되는 용어입니다. encoder-decoder 구조에서 이러한 문제가 발생하는 것은 문장이 길어질수록 문장 전체의 정보를 짧은 고정 길이의 벡터로 나타내기 어렵기 떄문입니다.

> ### ⅱ. Introduction

인코더-디코더의 접근방법은 원문 sentence에서 고정된 길이의 벡터로 압축하면서 잠재적인 이슈가 발생합니다. 이것은 특히 훈련 corpus보다 큰 긴 sentences에서 어려워집니다. 이러한 문제는 ["(2014). On the properties of neural ¨ machine translation: Encoder–Decoder approaches"](https://arxiv.org/abs/1409.1259) 논문에서 input sentence가 증가함에따라 인코더-디코더 구조의 성능이 약화되는 것을 보여줍니다.

이러한 문제를 해결하기위해, Alignment 및 Translation을 동시에 학습하는 인코더-디코더 모델의 확장을 제안합니다. 제안된 모델은 번역을 할때 매 반복마다 단어를 생성하고 원문장에서 가장 집중적으로 관련된 정보들의 위치를 찾습니다. 그리고 이러한 위치(postion)와 이전에 생성된 목표 단어들과 관련된 context vector를 기반으로 목표 단어를 예측합니다.

기존의 인코더-디코더와 구분되는 부분은 고정된 길이의 vector로 만들 필요가 없는 것입니다. 대신에 입력 문장을 vector들의 시퀀스로 만들고 번역시 디코딩할때 해당 벡터들의 부분집합을 적용하여 선택합니다. 원문장의 정보가 길이에 관하여 고정된 길이의 벡터로 변환될때 뭉개는 문제에 대해서 자유롭고, 긴 문장에 대해서 더 잘 대응하는 것을 보여줍니다.

> ### ⅲ. Background : NEURAL MACHINE TRANSLATION

신경망 기계변역(NMP$$_{Neural-machine-translation}$$)를 확률적인 관점에서보면 번역은 원문장 x가 주어졌을때의 t를 찾는 조건부확률을 최대화하는 target 문장 y를 찾는 방법입니다. 최근에는 신경망으로 조건부확률분포를 직접적으로 학습하는 수많은 논문들이 게제되었습니다. 두개의 컴포넌트로 구성된 x를 인코딩하고 y를 디코딩하는 방법의 접근방법인 encoder-decoder 모델을 예시로 이러한 새로운 접근법은 검증된 결과를 보여줬습니다. 


> ### ⅳ. LEARNING TO ALIGN AND TRANSLATE(Attention기법)

해당 논문에서 양방향 RNN으로 구성된 인코더와 번역을 디코딩하는 동안 원문장을 검색하는 것을 모방하는 디코더로 구성된 새로운 아키텍처를 제안했습니다.

![fit1](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/57e2c907-109a-4440-99cb-a81862ca33a6)
_Figure 2 : 원문 sentence가 주어 졌을떼,  t 번쨰 목표 단어(target word) $y_t$를 만드는 과정_


새로운 모델에서 아래와 같이 각각의 조건부 분포를 정의합니다. 

$$p(y_i|y_1, ..., y_{i-1}, x) = g(y_{i-1}, s_i, c_i)$$

$$where, \space s_i = f(s_{i-1}, y_{i-1}, c_i)$$

$s_i$는 time i에 대한 decoder의 은닉 상태로 이전의 출력 값 $y_{i-1}$과 각각의 목표 단어 $y_i$에 대한 각각의 context vector $c_i$로 생성됩니다. 여기에서 $f(\cdot), g(\cdot)$는 non-linear function을 의미한다.

$$c_i = \sum^{T_x}_{j=1}\alpha_{ij}h_{j \cdot}$$

time $j$에서 입력 $x_j$에 대해서 forward hidden state와 backward hidden state를 연결하여 j 번쨰 hidden state $h_j$를 생성합니다. 여기서 대괄호 $$[\cdot]$$은 두 벡터를 연결한다는 의미로 단순히 두 벡터를 나란히 배치하여 한 벡터로 만드는 방법입니다. 

$$h_j = [\vec{h}_j, \overset{\leftarrow}{h}_j]^\top$$

컨텍스트 벡터 $c_i$는 입력 시퀀스를 매핑하는 인코더의 주석$$_{annotations}$$ 시퀀스($h_1, ... , h_{T_x}$)인 은닉 상태들에 의존합니다. 각 주석 $h_i$는 입력 시퀀스의 i번쨰 단어 주변 부분에 대한 강한 초점을 가지면서 전체 입력시퀀스에 대한 정보를 포함합니다. 즉, context vector $c_i$는 encoder의 hidden state들의 weighted sum으로 표시되고 각각의 가중치는 $\alpha_{ij}$로 표시됩니다.

$$\alpha_{ij} = \frac{exp(e_{ij})}{\sum^{T_x}_{k=1}exp(e_{ik})}$$

$$where, \space e_{ik}=a(s_{i-1},h_j)$$ 

여기서 인코더의 은닉 상태들의 가중치인 $$\alpha _{ij}$$는 $e_{ij}$라 표기되는 alignment model의 Softmax함수로 표현이 됩니다. $e_{ij}$는 encoder의 j 위치의 은닉 상태 정보 $h_j$와 decoder의 이전 은닉 상태 정보 $s_{i-1}$ 사이에 얼마나 관련이 있는지 score를 매깁니다. 이를 통해 디코더가 입력 ㅣ퀀스의 관련 부분에 집중하면서 출력 시퀀스를 생성할 수 있습니다. 다시말해 `alignment model은 원문과 번역문 사이에서 단어들의 대응 관계를 찾는 역할`로 해당 연관성을 잠재 변수를 통해 고려하지 않고
Soft-Alignment라는 어텐션 스코어를 통해 직접 계산합니다. 이렇게 $$a(\cdot)$$는 전통적인 기계 번역과 달리제안된 시스템의 다른 모든 구성 요소와 함께 순전파 신경망으로 매개변수화 하고, 역전파를 통한 목적함수의 기울기를 계산하고 업데이트하는 최적화를 통해 효과적인 학습을 합니다. 그래서 해당 모델이 전체 번역 모델과 통합되어 잘 작동될 수 있습니다. 아래는 $$a(\cdot)$$을 하이퍼볼릭-탄젠트 함수를 통하여 수식으로 표현한 내용으로 $W$는 각각 학습 가능한 가중치들입니다.

$$a(s_{i-1},h_j) = W_a^Ttanh(W_bs_{i-1} + W_ch_j)$$

즉, 해당 weight $$\alpha _{ij}$$는 target word $y_i$와 source word $x_j, \space j=1,...,T$들과 어느정도 얼마나 연관성이 있는지를 나타내는 연관성에 대한 가중치 확률분포들을 얻을 수 있다. 그러면 i번째 context vector $c_i$는 모든 확률 $a_{ij}$의 모든 주석들에 대한 기대$$_{Expectation}$$ 주석입니다. 다시말해, 각 주석에 대한 확률인 weight를 기반으로 전체 주석들의 기대값이 source sentence에서 어떤 위치의 단어에 더 attention을 줄지 판단하는 i번쨰 컨텍스트 벡터가 된다는 뜻으로 생각할 수 있습니다. 이러한 방식으로 인해, 기존에 모든 문장을 하나의 고정 길이 벡터로 변환하는 작업을 수행하고도 더 좋은 성능을 제공할 수 있다고 합니다.

직관적으로 디코더의 주의 메커니즘을 구현하면 디코더는 원문장에 집중해야 할 부분을 결정합니다. 디코더에게 집중 기능을 부여함으로써 원문장의 모든 정보를 고정 길이 벡터로 인코딩할 필요가 없어집니다. 이 새로운 접근법을 통해 정보가 주석 시퀀스 전체에 퍼질 수 있으며 디코더에 따라 선택적으로 검색됩니다.

> 어텐션의 컨셉은 모델이 Alignment를 다른 Modalityes(Encoder, Decoder, etc..)들 사이에서 배우는 것입니다.
{: .prompt-tip }

이러한 방식으로 만들어진 Banadau Attention은 Additive Attention으로도 불리는데, Query(Q)와 Key(K)의 유사성을 계산하기 위해 순전파 신경망을 통과시켜 중간 히든 벡터를 생성한 후, 이를 사용하여 유사도를 계산하고 Value(V)를 가중합하는 유형을 뜻 합니다.


> ### ⅴ. 구현

```python
바나다우 어텐션
임베딩을 진행하여 단어들 벡터공간에 표시함으로써 유사도 계산이가능
https://paul-hyun.github.io/transformer-01/
https://snoop2head.github.io/Configuring/Professor-Lim's-Master-Class/
```






<!-- Luong Attention -->

> ## Ⅲ. Luong Attention

원 논문은 ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025)으로 Luong 어텐션은 Bahdanau 어텐션을 개선한 버전이고, 단순화된 점수 함수와 전역 및 지역 어텐션 방법을 제안합니다. 

어텐션 매커니즘은 번역과정에서 입력 문장의 특정 부분에 집중하는 방식의 NMT$$_{Neural machine translation}$$으로 최근에 크게 발전했습니다. 그에비해 효과적인 아키텍처를 분석하는 일에는 지지부진했습니다. 따라서 해당 논문에서는 간단하고 효과적인 어텐션 매커니즘 두가지를 시험합니다. `전체 문장의 단어를 참조하는 global 접근법`과 `문장의 부분단어들만 바라보는 local 접근법`입니다. 

> ### ⅰ. Introduction


효율성과 단순함을 중요시하며, 두 가지 새로운 타입의 어텐션 기반 모델을 구상하였습니다. Global Approach는 전체 문장을 참조하는 반면, Local Approach는 한번에 원문의 일부분만을 고려합니다.

이전의 바다나우(Bahdanau) 접근법과 비슷하지만, 조금 더 단순한 구조를 가지고 있습니다. 더 나아가, 하드(Hard)와 소프트(Soft) 어텐션 모델과 유사할 수 있는데, 글로벌 모델이나 소프트 어텐션보다 계산량이 적으면서 동시에 하드 어텐션과는 다르게 로컬 어텐션은 거의 모든 면에서 구별이 가능하여 구현이나 학습이 용이합니다.

> ### ⅱ. Neural Machine Translation

원문의 표현인 s를 디코더의 은닉 상태를 초기화할 때 단 한 번만 사용하지만, 바다나우(Bahdanau)와 해당 논문의 경우, 원문$$_{source}$$의 은닉 상태 s를 전체 번역 과정에서 참조합니다. 이러한 접근법은 어텐션 메커니즘이고, 다음에 자세히 다룰 것입니다. 우리의 목적 함수는 아래와 같이 정의됩니다. 

$$J_t = \sum_{(x,y) \in \mathbb{D}}-logp(y|x)$$

여기서 $\mathbb{D}$는 훈련 말뭉치들, 즉 병렬 훈련 코퍼스를 의미합니다.

> ### ⅲ. Attention-based Models


다양한 어텐션 기반 모델들은 크게 글로벌(Global)과 로컬(Local) 어텐션 두 가지 카테고리로 구분됩니다. 이는 전체 원문에 주목할 것인지, 아니면 부분적으로만 주목할 것인지의 차이를 가집니다. 

디코딩 단계의 매 t 스텝마다 두 가지 접근법은 여러 LSTM으로 구성된 Modality의 최상위 레이어에서 나온 은닉 상태(hidden state)를 받습니다. 

목표는 목표 단어(target word) $y_t$를 예측하는 데 도움이 되는 관련 원문 정보인 문맥 벡터(context vector) $c_t$를 전달하는 것입니다. 다만, 두 모델 간의 차이점은 문맥 벡터(context vector)를 어떻게 전달하는지에 있으며, 이 방식은 아래와 같은 단계를 공유합니다. 

목표 은닉 상태(target hidden state) $h_t$와 문맥 벡터(context vector) $c_t$를 받았을 때, 이 두 벡터에서 얻은 정보를 결합하기 위해 우리는 단순히 연속된 레이어를 사용할 수 있고, 아래와 같은 추가 은닉 상태를 생성할 수 있습니다. 

$$\tilde{h_t} = tanh(W_c[c_t:h_t])$$ 

이 추가 벡터 $\tilde{h_t}$는 소프트맥스 레이어(softmax layer)를 거쳐 아래와 같은 예측 확률 분포를 얻을 수 있습니다. 

$$p(y_t|y_{<t}, x) = softmax(W_s\tilde{h_t})$$ 

이제 각 모델에서 원문 측의 문맥 벡터(context vector) $c_t$를 계산하는 세부 사항을 설명하겠습니다.


> ### ⅳ. Global attention

해당 글로벌 어텐션 모델의 핵심 개념은 context vector를 전달할떄 인코더의 모든 은닉상태를 고려하는 것입니다. 

이 타입의 모델에서는 다양한 길이의 정렬 벡터$$_{alignment vector}$$ $a_t$를 원문 쪽의 모든 스텝에서 동일한 크기로 맞추고, 아래와 같이 현재 은닉 상태 $h_t$와 각 원문의 은닉 상태인 $\bar{h_s}$를 비교하여 전달합니다.

$$a_t(s)=align(h_t, \bar{h_s})  \qquad\qquad\quad\qquad$$

$$= \frac{exp(h_t, \bar{h_s})}{\sum_{s^\prime}exp(score(h_t, \bar{h_s}))}$$


![f2](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/fc0c6595-886d-4835-9ffc-e33b09e20ba5)
_Figure 2 : Gloval attentional model, 매 t 스텝마다 현재 목표 단어의 은닉 상태 $h_t$와 원문의 은닉 상태 $\bar{h}_t$를 가지고 가변 길이의 정렬 가중 벡터 $a_t$를 추론합니다. 이렇게 나온 정렬 가중 벡터를 전체 원문 은닉 상태에 가중 평균으로 Global context를 계산합니다._

아래의 score는 content 기반의 함수로써 세가지의 다양한 대안이 존재합니다.

$$score(h_t, \bar{h_s})=
\begin{cases}
h_t^\top \bar{h_s} & dot\\
h_t^\top W_a\bar{h_s}& general\\
v_a^\top tanh(W_a[h_t;\bar{h_s}]) & concat
\end{cases}$$

또한, 이전에 시도한 어텐션 기반 모델은 위치 기반 함수(location-based function)를 활용하여 아래와 같은 정렬 점수를 계산하였습니다.

$$a_t = softmax(W_ah_t) \quad location$$

$a_t$를 가중치로 사용하여, 문맥 벡터 $c_t$는 전체 문장의 잠재 변수 가중 평균으로 계산했습니다. 

$$c_t = \sum_{s} a_t(s) \bar{h_s}$$ 

Bahdanau 모델과 비교해서 글로벌 어텐션 접근법은 유사하지만 어떻게 간소화하고 일반화된지 알 수 있는 몇가지 구분되는 큰 특징이 있습니다. 처음으로, 최상위 LSTM 레이어의 은닉 상태가 인코더와 디코더에서 Bahdanau 모델에서 사용되었다면, 이 모델은 양방향 인코더의 순방향과 역방향을 결합한 원문의 잠재 변수를 활용합니다. 두 번째로, 계산 과정이 좀 더 간소하며, 마지막으로 이 모델은 다른 모델들과 비교하여 더 나은 대안이 됩니다.


> ### ⅴ. local attention

![f3](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/995618ea-6d0a-45de-abb3-c7a53c2343db)
_Figure 3 : Local attention model, 우선 현재 타겟 단어에서 단일 Aligned position $p_t$를 예측합니다. 다음으로 $p_t$를 중심으로 한 Window가 context vector $c_t$를 계산하는데 사용되고, Window 안에 있는 원문 은닉 상태를 가중평균하여 계산합니다. 여기서 가중치 $a_t$는 현재 은닉 상태 $h_t$와 윈도우 안에 있는 은닉 상태 $\bar{h}_t$들을 사용하여 계산합니다._


앞서 설명한 전역 어텐션의 약점은 각 목표 단어를 원문의 모든 단어 측으로 처리한다는 것입니다. 이는 계산 비용이 많이 들고 터무니 없이 긴 문장을 생성할 가능성이 있습니다. 이러한 약점을 해결하기 위해 타깃 단어별로 원문의 일부 위치에만 집중하는 로컬 어텐션 메커니즘을 제안했습니다.

이 모델은 소프트 어텐션과 하드 어텐션 사이의 트레이드오프 관에서 영감을 받았습니다. 소프트 어텐션은 글로벌 어텐션 방식에서 원 이미지의 모든 패치에 "부드럽게" 가중치를 두는 것입니다. 반면, 하드 어텐션은 이미지의 패치를 한 번에 하나씩만 보는 방법입니다. 이는 추론 시간이 계산 비용이 덜 비싸지만 non-differentiable(미분이 불가능?)하며 분산 감소나 강화 학습과 같은 계산 기법이 필요합니다. 

이 로컬 어텐션 메커니즘은 문맥의 작은 윈도우에만 집중하며 미분 가능합니다. 이 접근법은 소프트 어텐션에서 발생하는 비싼 계산을 피할 수 있으며, 하드 어텐션보다 학습이 쉽습니다. 상세하게 설명하면, 모델은 처음으로 각 타깃 단어에 대해 정렬된 위치 $p_t$를 매 t 시간마다 할당합니다. 문맥 벡터 $c_t$는 원문 은닉 상태의 가중 평균으로 [$p_t-D, p_t+D$] 크기의 윈도우에 전달됩니다. 여기서 $D$는 경험적으로 선택됩니다. 그리고 글로벌 접근법과는 달리, 로컬 정렬 벡터 $a_t$는 고정된 차원을 가집니다. 예를 들어, $$a_t \in \mathbb{R}^{2D+1}$$가 됩니다. 

이에 대해 두 가지를 고려하게 됩니다.

- Monotonic alignment(단조적 정렬, local-m)
    
    원문 sequence와 목표 sequence를 거의 단조롭게 $p_t=t$로 간단하게 설정합니다. 여기서 정렬 벡터 $a_t$는 앞서 정의한대로 사용됩니다.

- Predictive alignment(예측적 정렬, local-p)

    단조롭게 정렬을 하는것 대신에 aligned position을 아래와 같이 예측한다.

    $$p_t = S \cdot sigmoid(v^\top_p tanh(W_ph_t))$$

    $W_p, v_p$는 모델의 파라미터이며 positions를 예측할때 학습되며 $S$는 원문의 길이입니다. 그리고 Sigmoid 함수의 결과로 $p_t \in [0,S]$가 됩니다. $p_t$를 중심으로하는 가우시안 분포를 생성하고 alignment 가중치는 아래와 같이 정의됩니다.

    $$a_t(s) = align(h_t, \bar{h}_s)exp(-\frac{(s-p_t)^2}{2\sigma^2})$$

    여기서도 앞서 설명한 Align function을 사용한다. 그리고 표준편차는 $\sigma = \frac{D}{2}$와 같이 설정하고 $p_t$는 window 범위안에서의 정수로 생성된 실수입니다.

결과적으로 로컬 어텐션 방식은 글로벌 어텐션에 비해 원문의 일부분만 고려하므로 계산 비용이 감소하는 것으로 글로벌 어텐션의 한계를 극복했습니다.

> ### ⅵ. Input-Feeding Approach

![fig4](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/a47622b5-dd91-431c-9402-5ded1532eb2f)
_Figure 4 : Input-feeding approach, Attentional vector $\tilde{h}_t$는 이전의 결정을 알리기 위해 다음 단계의 입력값으로 들어갑니다._

앞서 제안한 글로벌과 로컬 접근법의 어텐션 선택은 각 단계에서 최선의 선택을 하는 동적으로 부분 최적화를 이루어낼 수 있습니다. 기존의 기계번역은 번역과정에서 원문이 번역되는 과정을 지속적으로 추적합니다. 반면에 attentional NMT에서 aligment decision은 과거의 alignment 정보를 기반으로 이루어집니다. 이 과정을 통해 어텐션 벡터 $\tilde{h}_t$는 다음 단계의 입력과 결합되는 입력-피딩(input-feeding) 접근법을 제안합니다.

이러한 연결은 두 가지의 효과로 (a) 모델이 이전의 alignment choices를 전체적으로 파악하고 (b) 수직적과 수평적 두 가지 측면에서 깊은 신경망을 만들 수 있게 합니다.

바다나우와 같은 다른 방법과 비교해서 해당 접근법은 모델이 입력 시퀀스의 모든 부분을 고려하여 중복 번역이나 놓친 부분이 발생하지 않도록 하는 "coverage" 효과를 받을 수 있으며 Figure 4와 같이 일반화가 가능합니다.



> ### ⅶ. Conclusion

해당 논문의 결론은 어텐션 메커니즘을 사용한 신경 기계 번역 모델이 기존의 기계 번역 방식보다 성능이 향상되며, 특히 글로벌 어텐션과 지역 어텐션 방식의 적용이 기계 번역의 품질을 개선하는 데 큰 역할을 한다는 것입니다. 글로벌 어텐션은 원문 전체를 고려하여 문맥 벡터를 계산하며, 모델이 원문의 중요한 정보를 포착하는 데 도움이 됩니다. 지역 어텐션은 원문의 일부 영역에 집중하여 계산의 효율성을 증가시킵니다. 논문에서 제안된 어텐션 기반의 신경 기계 번역 모델은 입력 문장의 모든 부분을 고려하여 중복 번역이나 놓친 부분이 발생하지 않도록 "coverage" 효과를 얻을 수 있습니다. 이를 통해 기계 번역의 일반화가 가능하며, 실제로 높은 성능을 보여준다고 결론지었습니다.

이러한 방식으로 만들어진 Luong Attention은 "Multiplicative Attention"으로도 알려져 있으며 `글로벌 어텐션의 일종`입니다. Luong의 어텐션 메커니즘에서 인코더와 디코더 상태를 곱셈(행렬곱)을 통해 조합하고, 이를 통해 어텐션 정렬 벡터를 얻습니다.

여기서 지역 어텐션은 글로벌 어텐션의 단점을 보완하기 위해 나오긴 했지만, 해당 논문에서는 글로벌 어텐션과 지역 어텐션의 방법 중 어떤 것이 더 좋은 것인지는 나타나지 않았습니다. 따라서 입력 시퀀스 전체에 대해 어텐션을 계산하는 `글로벌 어텐션은 원문의 이미 포함된 맥락을 시간 순서에 상관없이 전체 시퀀스를 고려하는 장점`이 있는 것에 반해 입력 시퀀스의 일부분만 어텐션을 계산하는 `지역 어텐션은 글로벌 어텐션에 비해 계산 효율성이 높고 일부분만 봐도 단어 간의 관계성이 적다면 더 좋은 성능을 발휘`할 수 있습니다. 결국 문제를 정확히 파악하여 적합한 방법을 선택해야합니다.

> ## Ⅳ. References

1. [딥러닝을 이용한 자연어 처리 입문 - Seq2Seq](https://wikidocs.net/24996)
2. [딥러닝을 이용한 자연어 처리 입문 - 어텐션 매커니즘](https://wikidocs.net/22893)
3. [어텐션 매커니즘 : Seq2Seq모델에서 트랜스포머 모델로 가기까지](https://heekangpark.github.io/nlp/attention)
4. [[논문 리뷰] Neural machine translation by jointly learning to align and translate (2014 NIPS)](https://misconstructed.tistory.com/49)