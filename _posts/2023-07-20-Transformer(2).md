---
title: '[Paper]Transformer(2)'
author: east
date: 2023-07-20 00:00:00 +09:00
categories: [Paper, NLP]
tags: [Paper, NLP]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
---

2017년에 Google Brain 팀에서 발표한 ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf)에서 Transformer 기반의 구조를 제안했습니다. 트랜스포머는 NLP에서 대성공을 거두었고, seq2seq 모델. 즉, 순서가 있고, 출력 그 자체가 시퀀스인 모든 문제에 적합하여 DALL-E, GPT3 같은 언어모델의 기반이 되기도 합니다.

> ## Ⅰ. "Attention is All You Need"

시퀀스 번역 모델에서 인코더, 디코더를 가지는 복잡한 RNN이나 CNN을 사용하는 신경망이 지배적입니다. 이러한 분야에서 SOTA$$_{State-of-the-Art}$$는 인코더와 디코더를 어텐션 매커니즘을 통해 연결된 모델이 대다수입니다. 여기서 저자들은 어텐션 매커니즘에 기반한 트랜스포머$$_{Transformer}$$라는 새롭고 단순한 아키텍처를 제안했습니다. 

> ## Ⅱ. Introduction

RNN, LSTM, GRU등의 신경망은 sequence modeling과 번역 문제의 언어 모델링이나 기계번역 작업에서 특히 확고히 SOTA 접근법으로 사용되고 있었습니다. 이렇게 인코더-디코더 구조와 재귀 언어 모델의 범위안에서 지속적으로 수 많은 노력들이 시도되었습니다.

재귀 모델은 일반적으로 입력과 출력 Sequence의 Symbol Postion에 따라 계산되어 이러한 순차적인 자연어는 선천적으로 병렬화하기 불가능해 특히 긴 시퀀스에서 치명적입니다. 이로 인한 메모리 제약으로 배치를 제한합니다. 그리고 모델의 성능을 높이기 위해 조건부 학습을 진행 합니다. 최근에는 행렬 인수 분해 기법$$_{Factorization-Trick}$$을 활용하여 계산 효율성과 조건부 계산에서 상당한 성과를 얻었습니다.

`어텐션 매커니즘은 입력과 출력 시퀀스 사이의 거리에 상관없이 모델의 의존성을 허용`하게해 강렬한 시퀀스 모델링과 번역 모델 등 다양한 분야에서 필수적인 부분이 되어가고 있습니다. 그러나 이러한 어텐션 매커니즘은 재귀적 신경망에서 많이 집중되있었습니다.

따라서 해당 논문은 트랜스포머라는 새로운 구조를 제안합니다. 해당 아키텍쳐는 재귀 방법을 사용하지 않고, 대신에 입력과 출력 사이에 있는 전역 의존성을 끌어내는 어텐션 매커니즘에만 전부 의존합니다.  


> ## Ⅲ. Background

순차적인 계산을 줄이는 것이 목표인 모델에서는 기본 구성 요소로 합성곱 신경망을 사용하며, 모든 입력과 출력 위치에 대한 숨겨진 표현을 병렬로 계산합니다. 이러한 모델에서 임의의 두 입력 또는 출력 위치 사이의 신호를 연관시키는데 필요한 연산횟수는 위치 간의 거리에 따라 선형적 또는 로그 선형적으로 증가합니다. 따라서 멀리 떨어진 위치 간의 종속성을 학습하기가 더 어려워집니다. 트랜스포머에서 이는 상수 시간 내의 연산으로 줄어들지만, 어텐션 가중치에 평균화된 위치 때문에 효과적인 해상도가 감소하는 단점이 있습니다. 이러한 효과는 `멀티-헤드 어텐션`$$_{Multi-Head}$$을 이용하여 대응합니다.

`셀프-어텐션`$$_{Self-Attention}$$은 "Intra-attention"이라고 불리는 어텐션 매커니즘으로 시퀀스의 표현을 계산하기 위해 단일 시퀀스를 다른 위치와 연관시키는 역할을 합니다. 셀프-어텐션은 독해력, 추상적 요약, 텍스트 추론과 독립 시퀀스 표현을 배우는 작업의 학습과 같은 다양한 작업에서 성공적으로 사용되었습니다.

`End-to-End`방식의 메모리 신경망은 순차적 순환 처리를 대신해서 순환 어텐션 매커니즘 기반을 사용합니다. 이 모델은 단순한 언어 질의 응답과 언어 모델 작업에서 좋은 성능을 보여줬습니다.

트랜스포머는 전체적으로 `RNN이나 CNN없이 입력과 출력의 표현을 계산하는 첫번째 형태 변환 모델`입니다. 다음에 오는 내용들을 통해 트랜스포머와 셀프-어텐션의 동기와 이점에 대한 토론을 설명하겠습니다.


> ## Ⅳ. Model Architcture

트랜스포머는 아래와 같이 Self-attention과 Point-wise, 완전 연결 계층을 각각의 인코더, 디코더에 구현합니다.

![1](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/148e97e6-d691-4408-a743-4caa7b6a4204)
_Figure 1 : Transformer - model architecture_

> ### ⅰ. Encoder and Decoder Stacks

- #### `Encoder`

  인코더는 동일한 N개의 layer로 구성되어 있습니다. 각 레이어는 두개의 서브레이어로 구성되는데, `처음은 멀티-헤드 셀프 어텐션`$$_{Multi-head \; self-attention}$$ 매커니즘이고 `두번째는 Position-wise 완전연결층 순전파 신경망`입니다. 우리는 레이어 정규화와 잔차 연결$$_{Residual \; connection}$$을 두개의 서브레이어로 적용했습니다. 각 서브레이어의 출력은 아래와 같이 되고, $$SubLayer(x)$$는 앞서말한 서브레이어로 구현됩니다.
  

- #### `Decoder`

  디코더도 인코더와 동일한 N개의 layer로 구성되어있습니다. 앞서 인코더에서 말한 두개의 서브레이어에 추가적으로 인코더 스택의 결과에서 멀티-헤드 어텐션을 수행하는 세번째 서브레이어로 구성되어있습니다. 인코더와 유사하게 각 서브레이어에 레이어 정규화와 잔차 연결을 선택했습니다. 또한, 디코더 안에 셀프-어텐션 레이어는 순차적인 포지션을 기준으로 작동하며 Subsequent postions에서 집중되는 위치 이동을 예방하기 위해 사용되어, position $$i$$에 대한 예측이 $$i$$ 이전의 알려진 출력에만 의존하도록 보장합니다. 이를 위해 출력 임베딩에서 한개의 위치만큼 오프셋$$_{offset}$$(위치이동)된 마스킹$$_{Subsequent-masks}$$이 사용됩니다. 즉, 미래의 단어가 현재 단어의 예측에 영향을 미치지 않게 하는 방법입니다.


> ### ⅱ. Regularzation

$$ ResiduaConnection(x) = x + Dropout(SubLayer(LayerNorm(x)))$$

트랜스포머는 모든 어텐션 레이어 및 모든 피드 포워드 레이어 이후에 잔차 연결과 레이어 정규화를 사용합니다. 이러한 잔차 연결 및 배치 정규화를 사용하여 퍼포먼스를 향상시키고, 훈련 시간을 단축하며, 심층 네트워크의 효과적인 학습을 가능하게 합니다. 

> ### ⅲ. Layer Normalization

레이어 정규화$$_{Normalization}$$ 유형은 큰 배치 사이즈에 영향을 받아 순환에 적합하지 않습니다. 따라서 기존 트랜스포머 아키텍처는 레이어 정규화를 사용하여 해당 문제를 해결합니다. 레이어 정규화는 배치 크기가 작더라도$$_{batch_size < 8}$$ 안정적인 성은을 보입니다. 레이어 정규화를 연산하기 위해, 미니배치의 각 샘플에 대한 평균$$_{\mu_i}$$과 표준편차$$_{\sigma_i}$$를 아래와 같이 별도로 계산합니다.

$$\mu_i = \frac{1}{K}\sum^{k}_{k=1}x_{i,k}, \;\; \sigma_i = \frac{1}{K}\sum^{k}_{k=1}(x_{i,k} - \mu_i)^2$$

그 후, 정규화 과정은 아래와 같이 정의됩니다.

$$LN_{\gamma, \beta}(x_i) \equiv \gamma \frac{x-\mu_i}{\sigma_i+\epsilon} + \beta$$

여기서 $\gamma$와 $\beta$는 학습 가능한 매개변수이고, 표준편차가 0일 경우 수치 안정성$_{numerical-stability}$을 위해 작은 수인 $\epsilon$가 추가됩니다.

> ### ⅳ. Residual Connection

레지듀얼 커넥션(Residual Connection)은 이전(하위) 레이어의 출력을 현재 레이어의 출력에 추가하는 것을 의미합니다. 이렇게 하면 네트워크가 특정 레이어를 '건너뛰기' 할 수 있는 추가적인 경로를 제공함으로써 레이어 사이에 정보가 더 쉽게 전달되게 합니다. 매우 깊은 네트워크에서는 그래디언트 소실 문제가 발생할 수 있어 이러한 문제를 해결하기 위해 잔차 연결을 사용하여 네트워크의 일부분이 다른 부분과 독립적으로 학습해 깊은 네트워크를 허용할 수 있습니다. 이로 인해 특정 레이어가 나머지 네트워크의 학습에 거의 영향을 주지 않는 경우에도 효과적으로 작동할 수 있습니다. 

이렇게 함으로써 복잡한 네트워크 구조를 단순화하는 데 도움이되며, 모델의 총체적인 성능을 개선하고 훈련 시간을 단축시키게 됩니다. 


> ## Ⅴ. Attention

어텐션 매커니즘은 Seq2Seq에서 입력 시퀀스 정보 손실을 보정해주기 위해 사용되어 이전에 좋은 성과를 이루었지만, 합성곱이나 재귀에만 사용이 되었습니다. 그래서 앞서 설명한 거과 같이 어텐션 매커니즘을 보정 목적이 아닌, 인코더와 디코더로 구성한 모델인 트랜스포머의 논문의 제목 "Attention is All you need"는 과감한 표현이었다고 합니다.

Seq2Seq 모델은 입력 시퀀스를 인코더에서 컨텍스트 벡터라는 하나의 고정된 크기의 벡터 표현으로 압축하고, 디코더는 이 컨텍스트 벡터를 통해서 출력 시퀀스를 만들어냈습니다. 이는 긴 시퀀스를 요약하기에는 Context $C$의 차원이 너무 작다는 제한이 있었습니다.

![attention machanism](https://github.com/eastk1te/P.T/assets/77319450/cbbe0628-fc5d-486b-9a6c-3e4fdca77b0a){: .left .width="50%"}
_Figure 1 : 어텐션 아키텍쳐 예시_

![context sequence](https://github.com/eastk1te/P.T/assets/77319450/130c09a4-3fd1-4203-af0d-3be27e363c8b){: .right .width="50%"}
_Figure 2 : 어텐션 매커니즘_

따라서, 어텐션 메커니즘은 $C$를 가변길이$$_{variable-length}$$의 시퀀스로 만들어 주어 시퀀스 $C$의 원소인 $c^t$들을 출력 시퀀스의 원소인 $y^t$들과 매치시킵니다.

어텐션 함수는 출력에 대응되는 key-value쌍과 query로 모두 vector 형태로 묘사될수 있습니다. 출력은 Value 벡터들의 가중합으로 계산되고, 가중치는 각 value에 query와 연관된 key의 함수로 배정됩니다.

트랜스포머는 아래와 같이 보정된 내적곱 어텐션$$_{Scaled \; Dot-Product \; Attention}$$과 멀티헤드 어텐션$$_{Multi-Head \; Attention}$$ 메커니즘을 사용합니다.

![3](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/c8cefbbe-1d9d-4615-9e5a-743a906a5cbd)
_Figure 3 : (좌) 보정된 내적곱 어텐션 (우)여러개의 병렬적으로 동작하는 어텐션 레이로 구성된 멀티-헤드 어텐션_

> ### ⅰ. Scaled Dot-Product Attention

일종의 루옹 어텐션인 `"Scaled Dot-Product Attention"`으로 내적으로 주어진 Query(Q)와 Key(K)로 Dot-Product(Multiplicative) 어텐션을 사용하여 유사도를 계산하고, 이에 기반하여 Value(V)를 가중합하여 어텐션 값을 계산하는 어텐션 함수입니다. 이때, 내적 결과에 Sacling factor인 $$\frac{1}{\sqrt{d_k}}$$를 적용하여 어텐션 값의 크기를 조절하는 방법입니다. 

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt(d_k)})V$$

$d_k$의 차원으로 구성된 query와 key 그리고 $d_v$의 차원으로 구성된 value들로 만들어진 행렬들을 입력으로 받아 동시에 어텐션 함수를 계산합니다. Q와 K에 대한 내적 곱을 구하고 보정 요소 $\sqrt{d_k}$로 나누어준 다음에 softmax 함수를 적용하여 V를 가중치로 곱합니다.

아래의 그림을 통해 각 행렬의 차원에 대해 설명하겠습니다.

![image](https://velog.velcdn.com/images%2Fcha-suyeon%2Fpost%2F26e3183e-53b1-4ad0-a6ff-dc5bd5a4591d%2Fimage.png)
_Figure 4 : 쿼리(Q)는 입력의 임베딩이며, 값(V)과 키(K)는 목표로 일반적으로 동일합니다. 임베딩 된 입력문장의 행렬은 (시퀀스 길이, 임베딩 차원 수), $W^\bullet$는 (임베딩 차원수, 임베딩차원수), Q, K, V 들은 (시퀀스 길이, 임베딩 차원 수)의 shape을 가지게 됩니다._

해당 내용에서 `Q는 Query를 뜻하며 "무엇이 나랑 관련있어?"라는 질문`을 뜻하고, `K는 해당 쿼리의 응답으로 "내가 관련있어"라는 의미`로 받아들이면 됩니다. Figure 4에서 확인 가능하듯이 Q와 K, V는 입력 임베딩($$dim_{embed}, n$$)의 행렬과 학습 가능한 각 가중치 행렬($$W_Q, W_K, (dim_{model}, dim_{embed})$$)들의 내적으로 계산됩니다. 즉, 입력 토큰의 임베딩 벡터가 $$dim_{model}$$ 차원의 공간으로 가중치행렬을 통해 투영됩니다.이를 통해 Q와 K를 dot product 연산을 사용해 각 토큰간의 상관관계를 표현(즉, Q와 K가 얼마나 유사한지)할 수 있습니다.

![1](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/a1bf430e-97b7-405e-b7e4-ae5972100ff5)
_출처 : 3Blue1Brown 유튜브_

`V는 Value를 뜻하며 실제로 주목할 대상`을 나타냅니다. V에도 마찬가지로 Q,K와 동일하게 입력 임베딩에 가중치 행렬($$W_V$$)을 곱해 계산이 되는데, 위 그림에서 "blue fluffy creature"를 예시로 임베딩 차원에서 현재 토큰 임베딩 벡터에서 다음 토큰 임베딩 벡터로 가는 Value 벡터를 계산합니다. 이는 해단 토큰이 문장에서 가지는 의미를 담고있어 어느 곳으로 가야하는지 알려주는 것으로 아래식에서 $$Δ/vec{E}_i$$와 같이 표현할 수 있습니다.

$$\begin{array}
W_V E= \vec{V} \\
Δ/vec{E}_i = \vec{V} \vec{E}_i \\
/vec{E}_i + Δ/vec{E}_i = /vec{E}_i' \\
\end{array}$$

이러한 Value의 작업은 토큰 임베딩 차원에서 진행되어 ($$dim_{embed}, dim_{embed}$$)으로 구성될 수 있으나 "Low Rank" 변환을 통해 ($$dim_{embed}, dim_{embed}$$)을 고차원에서 저차원으로 그리고 저차원에서 고차원으로 투영하는 $$(dim_{embed}, dim_{model}) \cdot (dim_{model}, dim_{embed})$$의 형태로 표현하여 파라미터 수를 현격히 감소 시킨다고 합니다.

Scaled Dot-Product Attention은 MLP를 사용하여 유사도를 계산하는 Additive(Bahdanau) 어텐션과 비교하여 이론적 복잡성이 동일할 때, 행렬 내적 연산이 더 낮은 계산 복잡성과 공간 효율성을 가져 더 효율적인 모델로 최적화 됩니다.

작은 크기의 $d_k$로 구성된 두개의 매커니즘은 동일한 성능을 보이지만, 큰 크기의 $d_k$에서는 스케일링을 하지 않은 내적곱 어텐션보다 바나다우 어텐션의 성능이 좋다고 합니다. 논문은 $d_k$가 큰 차원인 경우, 내적곱이 큰 크기로 증가하여 내적곱을 softmax에 넣어 매우 작은 미분값$$_{gradient}$$을 가지도록 했습니다. 이러한 효과를 상쇄하기 위해 dot product를 $$\frac{1}{\sqrt{d_k}}$$로 보정했습니다.

> ### ⅱ. Dot-Product Attention

Dot-Product Attention에서 내적을 통해 유사도를 계산하는 이유에 대해 설명하겠습니다.

$$Q \cdot K = \Vert Q \Vert \Vert K \Vert \cdot cos(\theta)$$

Q,K의 내적에서 각 단어의 벡터들의 내적은 고차원 공간에서 두 벡터 간의 유사도를 계산하는데 사용됩니다. 내적의 값은 코사인 유사도와 관련되어 있는데, 여기서 코사인 유사도는 두 벡터사이의 코사인 각도로 두 벡터의 방향이 얼마나 유사한지를 측정합니다. 즉, 내적은 두 벡터의 방향과 크기를 모두 고려한 곱으로 내적값이 클수록 두 벡터의 방향이 유사하며 크기가 비슷하다고 생각할수있습니다.

이러한 어텐션 매커니즘 그 자체는 아주 효과적이며 행렬곱셈$_{matrix multiplication}$에 최적화된 GPU 및 TPU과 같은 최신 하드웨어에서 효율적으로 연산될 수 있습니다. 하지만 단일헤드 어텐션 레이어는 하나의 표현만을 허용하므로 다중 어텐션 헤드가 사용됩니다. 

```python
def scaled_dot_product_attention(q, k, v, mask):
  ...

  # Q와 K의 product 연산 수행
  matmul_qk = tf.matmul(q, k, transpose_b=True)

  # 차원 보정 값 연산
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # 마스킹 적용
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax를 적용하여 합계가 1인 가중치 합으로 변환
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  # V와 내적
  output = tf.matmul(attention_weights, v)

  return output, attention_weights
```


> ## Ⅶ. Multi-head Attention

`Multi-Head Attention`은 Q, K, V를 가지고 $d_{model}$ 차원의 단일-헤드$$_{Single-head}$$ 어텐션을 수행하는 것 대신에 각각 h번 $d_k$, $d_k$, $d_v$의 차원으로 선형적 투영을 하는것이 효과적이다고 보았습니다. 해당 투영은 어텐션 함수를 $d_v$ 차원의 결과 값을 병렬적으로 처리할 수 있게 해줍니다. 

즉, 모델이 서로 다른 표현의 다른 위치의 부분공간에서의 정보들을 결합하여 더 다양한 정보를 제공하는 역할을 합니다. 이와 반대로 단일 헤드 어텐션의 경우 평균화가 이를 억제합니다.

$$MultiHead(Q,K,V) = Concat(head_1, ... , head_h)W^O$$

$$where, \space head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$$

각각의 어텐션 헤드의 투영된 파라미터 매트릭스 $$QW^Q_i \in \mathbb{R}^{d_{model} x d_k}, KW^K_i \in \mathbb{R}^{d_{model} x d_k}, VW^V_i \in \mathbb{R}^{d_{model} x d_v} \; and \; VW^O_i \in \mathbb{R}^{hd_v x d_{model}}$$를 독립적(병렬적)으로 학습하고, 해당 헤드에서 어텐션 연산을 수행합니다. 

위와 같이 각 헤드에서 나온 결과를 Concat으로 결합하여 다양한 표현을 학습할 수 있습니다. 이렇게 각 헤드에서 차원 축소가 일어나기 떄문에 총 계산 비용은 단일 헤드 어텐션의 전체 차원하고 비슷해지지만 `다중 패턴 및 표현을 학습`할 수 있어집니다.

즉, 멀티 헤드 어텐션에서 각 헤드는 개별적으로 무작위 초기화 가중치 행렬들을 가지는데 이 행렬들이 고차원을 가지기 때문에 학습을 하면서 `같은 지역최솟값을 가지는 경우가 드물어 다양성을 보장`한다는 이야기입니다.

디코더에서는 `Masked Multi-head 어텐션`이 사용되는데, 서브레이어에서 고정 위치를 매우 큰 음수로 채움으로써 마스킹 됩니다. 이는 자기 회귀적인 특성 떄문에 `미래 토큰에 대한 정보를 차단`하기 위해 후속 위치를 처리함으로써 모델이 부정행위$_{cheating}$을 예방하기 위함. 이를 통해 모델은 다음 토큰을 예측하려 할 떄 이전 위치의 단어에만 주의를 기울여 실제 단어를 예측하는 데 필요한 정보를 학습할 수 있습니다. 이는 소프트맥스 함수를 통과한 결과에서 해당 위치의 값을 0 근처로 밀어넣어, 어텐션 메커니즘에서 후속 위치에 주어지는 가중치를 최소화합니다. 

```python
class MultiHeadAttention(tf.keras.layers.Layer):
  ...
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    # (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

    # (batch_size, seq_len_q, d_model)
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  
    ...
```

> ### ⅰ. Applications of Attention in our Model

트랜스포머는 세가지방식으로 멀티헤드 어텐션을 적용합니다.

1. `인코더-디코더 어텐션`
   
  디코더에서 사용되며, query는 이전 디코더 레이어에서, 메모리 key, value는 인코더의 출력에서 나옵니다. 이 과정은 디코더의 매 position마다 입력 시퀀스 전체에 종합적으로 주의를 기울이고, 전형적인 seq2seq 기반의 인코더-디코더 어텐션 매커니즘을 흉내$$_{Mimics}$$냅니다. 즉, 디코더와 인코더와의 연관 정보를 파악합니다.

2. `인코더의 셀프-어텐션 계층`
  
  self-attention의 key, value, query는 모두 같은 곳에서 나오고 이경우 인코더의 이전레이어에서 나옵니다. 인코더의 각 position은 인코더의 이전 레이어의 모든 position에 주의를 기울입니다. 즉, 셀프 어텐션을 통해 각 단어에 대한 정보와 다른 단어들과의 관계를 파악합니다.

3. `디코더의 셀프-어텐션 계층`
  
  디코더의 각 위치가 그 위치를 포함하여 디코더의 이전의 모든 위치에 주의를 기울입니다. 이는 부정행위를 방지하기 위해 값을 차단하는 마스킹이 사용됩니다. 즉, 디코더의 이전 요소들 사이의 관계를 파악합니다.

이러한 세가지 방식을 사용함으로써 다양한 관점에 주의를 기울이며, 다른 종류의 패턴 및 의미를 추출합니다.

> ## Ⅷ. Position-Wise FFN

인코더, 디코더 내부의 서브레이어에는 완전연결 순전파 신경망이 포함되어 있습니다. 각 위치의 정보를 독립적으로 처리하는 것을 목적으로 이는 아래와 같이 두개의 선형 변환사이에 ReLU 활성화함수를 사용하여 구성됩니다.

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

이 선형 변환들은 각 계층에서 다른 파라미터를 사용하여 다른 position에 있는 같은 패턴을 인식할 수 있습니다. 다른 관점에서 보면 kernel size 1인 두개의 convolution으로 설명할수있습니다. 즉, 위치별로 독립 작용하므로 각 위치에 독립적인 정보 처리가 가능합니다. 이를 통해 시퀀스 내의 각 토큰은 별도로 학습되고 처리됩니다.

> ### ⅰ. Embedding and Softmax

다른 순차적 변환 모델과 유사하게 학습된 embedding을 입력 토큰과 출력 토큰을 $d_{model}$차원의 벡터로 변환하는데 사용합니다. 또한, 학습된 선형 변환과 softmax 함수로 디코더의 다음 토큰의 확률분포를 출력합니다. 해당 모델에서 embedding layer와 pre-softmax 선형 변환 사이의 가중치 행렬을 공유합니다. 이렇게 함으로써 더 적은 변수를 추정하고, 메모리나 연산 비용이 줄고 학습 또한 개선이 됩니다. 그리고 임베딩 계층에서 $$\sqrt{d_{model}}$$를 곱하여 스케일링을 수행하는데, 이는ㄴ 모델의 수렴에 도움이 되며, 과적합 등을 밪이할 수 있습니다.

> ### ⅱ. Positional Encoding

해당 모델에서는 모델이 순차적으로 시퀀스의 순서를 가지고 토근의 상대 위치(relatevie postion)에 대한 정보를 사용하는 RNN이나 CNN을 사용하지않습니다. 따라서 우리는 시퀀스 토큰과 관련되거나 절대적인 position에 대한 정보를 넣어야합니다. 그래서 우리는 "positional encoding"을 입력 임베딩을 인코더와 디코더 맨 아래에 추가했습니다. 포지셔널 인코딩은 $$d_{model}$$의 차원으로 임베딩과 같은 차원입니다. 따라서 두개의 덧셈이 수행가능해집니다. 포지셔널 인코딩은 학습되거나 고정된 다양한 방법이 존재합니다. 

여기서 다른 주기에 sine과 cosine 함수를 적용했습니다.

$$Embedding = xE \; + \; PE \\ where, \; x:input, \;E:Embedding Maxtrix$$

$$PE_{(pos, 2i)} = sin(pos/100000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = cos(pos/100000^{2i/d_{model}})$$

$pos$는 position이고, i는 dimension입니다. 포지셔널 인코딩의 각 차원은 사인곡선$$_{sinusoid}$$에 해당하여 $[2\pi, 10000 \cdot 2\pi]$의 기하학적 수열의 파장의 길이를 가집니다. 모델이 연관된 position에 주의를 주는 것을 쉽게 배우도록 가정한 것이 이 함수를 사용한 이유입니다. 따라서 고정된 offset $k$를 가지는  $PE_{pos+k}$는 $PE_{pos\cdot}$의 선형 함수로 표현될 수 있습니다. 위와 같은 공식을 그대로 사용하면 숫자 오버 플로우가 발생할 수도 있어 보통 log 공간에서 사용이 됩니다.

```python
def positional_encoding(position, d_model):

  # p : position, i : dimention
  angle_rads = pos * (1 / np.power(10000, (2 * (i//2)) / np.float32(d_model)))
  # 짝수 칸에 적용
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # 홀수 칸에 적용
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]
```

> ## Ⅸ. Why Self-Attention

`Self-Attention(or Encoder-Decoder Attention)`은 Query, Key, Value라는 `입력 시퀀스의 값들을 행렬로 변환`하고 어텐션 가중치와 출력을 계산하는 매커니즘입니다. 해당 계층은 가변 길이의 시퀀스를 동일한 길이의 다른 시퀀스로 매핑하는 역할을 하는 RNN이나 CNN 계층과 여러 측면에서 비교됩니다. 셀프어텐션을 사용하는 이유는 다음과 같은 세 가지 요인을 고려합니다. 첫 번째는 `레이어당 총 계산 복잡성`이고, 두 번째는 `순차적 작업이 요구되는 최소 횟수`로 측정되는 병렬적으로 수행 가능한 계산량의 총량입니다. 마지막으로 세 번째 요인은 `네트워크의 장기 의존성 사이의 경로 길이`입니다.

장기 의존성을 학습하는 것은 많은 시퀀스 변환 작업에서 주요 도전 과제입니다. 이러한 의존성을 학습하는 데 중요한 영향을 미치는 한 가지 요소$$_{Factor}$$는 순전파와 역전파 신호가 네트워크를 지나는 통로의 길이입니다. 입력과 출력 시퀀스 내의 위치들 간의 결합이 작을수록, 장기 의존성 문제를 학습하기 더 쉬워집니다. 따라서 다른 유형의 계층으로 구성된 네트워크 내에서 입력과 출력 위치 사이의 최대 길이를 비교해야 합니다. 

셀프 어텐션 계층은 지속적으로 작업이 시작되는 모든 위치와 연결됩니다. 반면에 순환 계층은 $O(n)$의 시간 복잡도가 요구됩니다. 계산 복잡도 측면에서 시퀀스 길이 $n$이 표현 차원 $d$보다 작으면, 셀프 어텐션 계층은 순환 계층보다 더 빠릅니다. 이는 기계 번역의 최첨단 모델에서 가장 많이 사용되는 차원 $d$(예: word-piece, byte-pair 표현)와 같습니다. 매우 긴 시퀀스를 처리하는 계산 성능을 향상시키기 위해 셀프 어텐션은 출력 시퀀스의 각 위치를 중심으로 주변 입력 시퀀스를 근접한 이웃 크기 $r$만큼 제한합니다. 이렇게 하면 최대 통로 길이가 $O(n/r)$까지 증가합니다. 이 접근법은 추후 작업으로 계속 실험할 예정입니다.

$k < n$의 kernel 너비를 가지는 단일 컨볼루션 계층은 입력과 출력의 위치의 모든 쌍을 연결하지않습니다. 그렇게 하려면 연속적인 kernel의 경우 $O(n/k)$의 컨볼루션 계층을 쌓아 확장된 컨볼루션에 $O(log_k(n))$을 쌓아야합니다. 결과적으로 네트워크의 어떤 두개의 위치 사이의 가징 긴 통로의 길이가 증가하게 됩니다. 일반적으로 $k$ 요소를 가지는 컨볼루션 계층은 순환 계층보다 비용이 높습니다. 그러나 분리가능한 컨볼루션은 고려할 수 있는 복잡성이 $O(k\cdot n\cdot d + n\cdot d^2)$으로 감소합니다. $k = n$일 때, 분리가능한 컨볼루션의 복잡성은 셀프어텐션 계층과 Point-wise FFN 계층의 결합과 동일해져 이 접근법을 모델에 사용했습니다.

이점의 관점에서 셀프어텐션은 좀 더 설명가능한 모델을 만들 수 있습니다. 우리는 현재 토론되는 예제들의 어텐션 분포를 조사했습니다. 개별적인 어텐션 헤드는 다른 작업을 학습하는 것뿐만 아니라 문장의 구문과 문맥적 구조와 연관된 행동의 증거도 나타났습니다.

쉽게 설명하면, 셀프어텐션은 동일한 문장 자체에서 가져온 Q와 K를 가지는 형태로 학습되고 이에 변형으로 교차 어텐션(Cross Attention)은 다른 언어나 모달리티로 Q와 K를 다르게 설정하는 형태(ex. Q=영어,K=한글, 어디에 연관되어 있을지 몰라 마스킹 적용 안함)입니다.



![4](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/ef8c0648-7a7a-4637-9718-885ca316011b){: .width="500"}
_Figure 4 : 인코더의 self-attention에서 긴 길이의 의존성을 가지는 어텐션 매커니즘의 예시이다. 어텐션 헤드 중 많은 헤드는 동사 'making'과의 먼 의존성을 집중해서 처리하며, 'making...more difficult'라는 문구를 완성합니다. 여기서 보여진 주의는 단어 'making'에 대해서만 적용되었습니다. 다른 색상은 다른 헤드를 나타냅니다. 색상으로 보는 것이 가장 좋습니다._

![5](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/00ec8699-fb2e-44a3-ac0a-fb5cb24f5b6c){: .width="300"}
_Figure 5 : 많은 어텐션 헤드들이 문장의 구조와 연관된 행동을 보여주고 있다. 위의 인코더의 셀프어텐션으로 부터 얻은 두개의 다른 헤드에서 각 헤드는 다른 작업을 학습하는 것을 분명히 보여준다._



> #### Update. 추가적인 설명(24.04.23)
{: .prompt-info }

트랜스포머에 관한 영상으로 이해하기 쉽게 잘 되어있어 이해에 도움이 되었습니다.

{% include embed/youtube.html id='eMlx5fFNoYc' %}

이러한 어텐션 매커니즘은 다변량 통계 관점에서 요인 분석(Factor analysis)을 통해 변수 간의 상관 관계를 분석하여 관측된 데이터를 더 적은 수의 요인이나 구성 요소로 설명할 수 있다고 합니다. 즉, 통계 공부도 열심히 하자!


> ## Ⅻ. References

1. ["스케일드 닷-프로덕트 어텐션(Scaled dot-product Attention)"](https://velog.io/@cha-suyeon/%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%93%9C-%EB%8B%B7-%ED%94%84%EB%A1%9C%EB%8D%95%ED%8A%B8-%EC%96%B4%ED%85%90%EC%85%98Scaled-dot-product-Attention)
2. [밑바닥부터 이해하는 어텐션 매커니즘](https://glee1228.tistory.com/3)
3. [딥러닝을 이용한 자연어 처리 입문 - 트랜스포머](https://wikidocs.net/31379)
4. ["차근차근 이해하는 Transformer(1)"](https://tigris-data-science.tistory.com/entry/%EC%B0%A8%EA%B7%BC%EC%B0%A8%EA%B7%BC-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-Transformer1-Scaled-Dot-Product-Attention)
5. [NLP - wikidocs](https://wikidocs.net/22893)
6. [Huggingface Transformer](https://huggingface.co/docs/transformers/model_summary)
7. [Tensorflow Transformer](https://www.tensorflow.org/text/tutorials/transformer?hl=ko#multi-head_attention)