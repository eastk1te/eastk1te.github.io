---
title: '[Paper]Model Compression'
author: east
date: 2024-05-02 00:00:00 +09:00
categories: [Paper, Model compression]
tags: [Paper, Model compression, Knowledge Distillation, Pruning, Low-Rank Factorization]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
# 
# 
---

딥 러닝 모델의 발전으로 하드웨어적인 요구 사항도 높아져 모델 배포 측면에서 이러한 요구사항을 갖추지 못한 곳에서는 큰 제약이 따르게 되었습니다. 이러한 제약을 극복하고자 모델을 압축하는 다양한 방법들이 개발되었는데, 해당 내용들을 다루어 보겠습니다.

> ## Ⅰ. Model Compression Methods

상용 LLM 모델 중 하나인 "GPT-175B"의 인퍼런스 모델들을 배포하기 위해서는 적어도 80GB의 메모리를 갖춘 A100 GPU를 적어도 5개 정도 갖추어야 한다고 합니다. 그러나 많은 경우 이러한 강력한 하드웨어를 사용할 수 없습니다.

또한, 이러한 큰 규모의 모델에서는 응답을 처리하고 생성하는 데 시간이 오래 걸립니다. 따라서, 모델을 압축하고 경량화하여 성능의 제약을 없애고 정확도를 다소 포기하는 대신 빠른 추론의 속도를 올려 적절한 균형을 찾아야 합니다.


![1](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/e344ffad-4bd4-4632-be69-8b3585428159){: w="500"}
_Figure 1 : model compression Taxonomy_

해당 논문에서 위 그림 처럼 이러한 모델 압축 방법을 분류합니다.

> ## Ⅱ. Pruning

가지치기 방법은 **필요없거나 중복되는 구성 요소들을 제거함으로써 모델을 압축하는 방법**입니다. 이러한 가지치기는 비구조적 가지치기와 구조적 가지치기로 target과 네트워크 구조 결과에 따라 두 가지 범주로 구분됩니다. 구조적 가지치기는 특정 룰에 따라 연결을 제거하여 네트워크 구조를 유지하는 반면 비구조적 가지치기는 독립적인 파라미터를 제거하는 방식입니다.

> ### ⅰ. Unstructured


비구조적 방법은 내부 구조를 고려하지 않고 특정 매개변수를 제거하는 방법으로, 임계값을 활용하여 매개변수를 0으로 처리합니다. 

이와 같은 방법은 SparseGPT와 같은 대형 GPT 모델에서도 60%까지 상당한 비구조화 희소성을 달성했습니다. LoRAPrune은 LORA에서 파생된 값 및 그래디언트를 사용하여 고유한 파라미터 중요도 기준등을 도입한다고 합니다.

```python
def unstructured_pruning(model, threshold):
    pruned_model = generate_model()
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            # 임계값 이하 가중치 0으로 제거.
            weights[0][np.abs(weights[0]) < threshold] = 0 
            pruned_model.layers[i].set_weights(weights)
    return pruned_model

```

> ### ⅱ. Structured

구조적 가지치기는 신경, 채널 및 층과 같은 전체 네트워크 구조의 구성 요소를 제거하여 네트워크를 단순화하는 방법입니다. 이 방법은 가중치 집합 전체를 목표로 하고 있습니다.

GUM 또는 LLM-Prunder와 같은 방법들이 주로 이러한 구조적 가지치기를 수행하는데 사용될 수 있습니다.

```python
class LnStructured(BasePruningMethod):

   ...

   def compute_mask(self, t, default_mask):

      ...
      
      # L_n norm을 활용하여 중요한 가중치를 고르는 metric
      norm = _compute_norm(t, self.n, self.dim)
      
      # 상위 k개의 가중치만 선택
      topk = torch.topk(norm, k=nparams_tokeep, largest=True)

      # 가지치기를 위한 이진 mask 생성
      def make_mask(t, dim, indices):
         ...
         return mask

      if nparams_toprune == 0: 
         mask = default_mask
      else:
         # 상위 k 개 제외한 가중치 가지치기 진행.
         mask = make_mask(t, self.dim, topk.indices)
         mask *= default_mask.to(dtype=mask.dtype)

      return mask
```

> ## Ⅲ. Knowledge Disillation

지식 증류는 큰 모델로부터 얻은 지식을 작은 모델로 전달하는 일련의 과정입니다. 모델 배포의 관점에서 복잡한 모델이나 앙상블 모델을 대규모 사용자에게 배포하는 것은 어려운 일입니다. 이에 대한 해결책으로 지식 증류는 큰 모델이 가진 지식을 간소화하여 작은 모델로 이전하는 과정을 포함합니다.

이러한 방법은 맨 아래에서 다시 한번 다루도록 하고, 현재는 관련된 분류를 확인하겠습니다.


![2](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/e57a1973-c2aa-446c-94ff-e0896156dc4f){: w="500"}
_Figure 2 : knowledge distillation_

지식 증류 기법은 학생 모델이 교사 모델의 정보를 어디까지 활용하느냐에 따라 화이트박스와 블랙 박스 두 가지 범주로 나뉜다고 합니다.

> ### ⅰ. WhiteBox KD

교사 모델의 파라미터를 활용할 수 있는 방법으로 교사 LLM의 예측 뿐만 아니라 그 매개변수에도 접근하여 활용함으로써, 학생 모델이 교사 모델의 내부 구조와 지식을 보다 깊이 이해할 수 있게하여 복잡한 지식을 더 효과적으로 배울 수 있도록 도와준다고 합니다.

> ### ⅱ. BlackBox KD

교사 모델의 최종 예측 결과에만 접근할 수 있는 방법으로 최근 연구에서는 LLM API를 통해 생성된 프롬프트와 응답 쌍을 활용하여 작은 모델을 미세조정하는데 있어 유망한 결과를 보여주었다고 합니다. 이 방법은 교사 모델의 복잡한 내부 구조나 파라미터에 접근하지 않고도 작은 모델을 높은 성능으로 학습할 수 있다는 장점이 있습니다.

![3](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/1407dd67-92ea-4c02-b741-2b7545e7d8dc){: w="500"}
_Figure 3 : Black Box KD overview, (a) in context learning distillation (b) Chain of Thought distillation (c) Instruction Following distillation_

해당 블랙박스 기법은 위 그림과 같이 학습 과정에서 다른 접근 방식을 취해 아래와 같이 세 가지의 다른 접근 방식이 개발되었습니다.

> #### `In-context learning distillation(ICL)`

구조화된 자연어 프롬프트를 사용하여 작업 설명과 몇 가지 예시를 포함하는 방식으로, 명시적인 그래디언트 업데이트 없이 새로운 작업을 이해하고 수행할 수 있습니다. 이는 Meta ICT와 Multitask ICT 등의 두 가지 오브젝트로 더 상세하게 나뉘어 모델을 미세조정합니다.

> #### `Chain of Thought distillation(CoT)`

프롬프트에 중간 추론 단계를 포함하여 최종 출력을 이끌어내는 ICL과는 다른 접근 방식을 취합니다. 이 방법은 LLM이 생성한 설명을 활용하여 더 작은 resoners의 학습을 개선시키는 것을 목표로 멀티태스크 프레임워크를 활용하여 ㅈ가은 모델에 강력한 추론 능력과 설명 생성 능력을 부여합니다.


> #### `Instruction Following distillation(IF)`

작업 설명을 읽고 소수의 예시에 의존하지 않고 완전히 새로운 작업을 수행하는 언어 모델의 능력을 향상시키기 위한 방법입니다. 다양한 명령어로 표현된 작업들을 사용하여 미세 조정을 거쳐, 언어 모델은 이전에 보지 못한 명령에 따라 작업을 정확하게 수행하는 능력을 갖추게 됩니다.

> ## Ⅳ. Quantization

전통적인 부동 소수점$$_{floating-point}$$을 정수나 다른 이산형태로 변환하는 과정입니다. 이 변환을 통해 저장 공간과 계산 복잡도를 크게 줄일 수 있습니다. 즉, 흔히 사용되는 32-bit의 부동 소수점을 8-bit나 더 낮은 비트값으로 표현하여 값을 저장하는 메모리 크기가 현저히 줄이고 더 적은 비트 계산이 이루어지게 하여 계산 복잡도를 감소시킵니다. 이러한 과정은 가중치와 활성화 값의 범위를 조정(스케일링 등)하는 방식으로 약간의 정보 손실은 발생할 수 있습니다.

해당 방법은 QAT(Quantization-Aware Training), PTQ(Post-Training Quantization)의 학습 과정에 따른 분류와 PTQ에서 가중치만 하느냐와 가중치와 활성화를 모두 양자화하느냐에 따라 상세학게 더 나뉘어 Figure 1 에 표시된것처럼 세가지 범주로 나뉜다고 합니다. 

> ### ⅰ. Quantization-Aware Training(QAT)

학습 과정에 자연스럽게 통합되며, 학습 중 low-precision 표현을 사용하여 양자화로 인한 정밀도 손실을 처리할 수 있는 대규모 언어 모델의 능력을 향상시킵니다. 

> ### ⅱ. Post-Training Quantization(PTQ)

PTQ는 학습이 완료된 후 LLM의 매개변수를 양자화하는 과정을 포함합니다. 이 방법은 재학습 과정 없이 모델의 저장 및 계산 복잡성을 줄이는 간단하고 효율적인 방법입니다. 그러나 양자화 과정에서 일정 수준의 정밀도 손실이 발생할 수 있습니다.

양자화 과정은 단순히 가중치만 양자화하는 것이 아니라, 가중치와 활성화를 모두 양자화하려는 시도를 포함합니다. ZeroQuant와 같은 방법은 하드웨어 친화적인 양자화 체계를 제시하며, INT8로 데이터를 줄임으로써 더욱 효율적인 연산이 가능하도록 합니다. SmoothQuant와 같은 기술들은 양자화 과정에서 발생할 수 있는 정밀도 손실을 완화하는 데에 초점을 맞춥니다.

```python

### QAT
# 양자화를 고려한 학습
self.quant = torch.quantization.QuantStub()
self.dequant = torch.quantization.DeQuantStub()

model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
model_qat = torch.quantization.prepare_qat(model, inplace=False)
# 양자화를 고려한 학습이 여기서 진행됩니다.
model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)


# PTQ
# 학습 후 동적 양적화
model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

# 학습 후 정적 양적화
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
```



> ## Ⅴ. Low-Rank Factorization

해당 방법은 가중치 행렬을 두 개 이상의 작은 행렬로 분해하는 것으로 원래의 큰 가중치 행렬(W)를 두 개의 작은 행렬 U와 V의 행렬 곱으로 근사하는 과정입니다.

$$U \approx UV$$

위 처럼 큰 규모의 행렬을 저차원의 행렬곱으로 표현하게 되면 매개 변수의 수와 계산의 복잡성을 크게 줄일 수 있습니다. LoRA와 TensorGPT와 같은 기술들이 이러한 방법을 통해 임베딩을 저차원으로 저장하여 효율적으로 압축합니다.

> ## Ⅵ. Inference Efficient

LLM의 추론 효율성은 여러 가지 지표를 사용하여 평가됩니다. 이 지표들에는 모델의 파라미터 수, 모델 사이즈, 압축 비율(compression ratio), 추론 시간(inference time), 그리고 FLOPs(Floating Point Operations, 부동소수점 연산의 수)등이 존재하고, 특히 FLOPs는 모델의 계산 요구사항을 추정하는 데 효과적인 방법을 제공합니다. 



> ## Ⅶ. Hinton KD

앞서 설명한 지식 증류를 정리한 Hinton 교수님의 "Distilling the Knowledge in a Neural Network"을 살펴보겠습니다.

지식 증류 기법은 앙상블 지식을 단일 모델로 압축하여 배포하는데 훨씬 간편한 모델로 만들 수 있는 것을 보여줍니다. 앙상블 모델이나 Dropout 같은 정규화기법으로 학습된 매우 큰 단일 모델(번거러운$$_{cumbersome}$$ 모델)에서 증류라는 다른 종류의 학습을 사용하여 작은 모델로 지식을 전달 할 수 있습니다.

모델의 지식을 단순히 학습된 매개변수 값으로만 생각하는 경향이 있지만 더 추상적인 관점에서는 입력 벡터에서 출력 벡터로의 학습된 매핑이라고 생각할 수 있습니다. 학습에 사용된 목적 함수는 실제 목표를 가능한 한 정확하게 반영해야 하지만, 모델은 보통 새로운 데이터에 대해 잘 일반화되는 것이 실제 목표입니다. 즉, 명확하게 일반화되도록 학습하는 것이 이상적이지만 이러한 일반화 방법에 대한 정보가 일반적으로는 사용할 수 없습니다.

위와 같이 `큰 모델의 일반화 능력을 작은 모델로 옮기는 명확한 방법은 큰 모델에서 생성된 클래스 확률을 "soft target"으로써 작은 모델의 학습에 사용`하는 것입니다. 큰 모델이 작은 모델의 큰 앙상블인 경우 개별 예측 분포의 산술 또는 기하 평균을 소프트 타겟으로 사용 가능합니다. 즉, 큰 모델의 각 클래스 예측 분포를 평균을 통해 작은 모델의 소프트 타겟으로 사용하여 작은 모델을 학습시킵니다.(ex. [[0.3, 0.2, 0.4],[0.5, 0.4, 0.2]] => [0.4, 0.3, 0.3])

이러한 '소프트 타겟'에는 각 클래스에 대한 확률 분포가 포함(평균을 통해 정보가 결합됨)되어 있어 각 클래스에 대한 예측의 불확실성에 대한 정보를 더 많이 전달할 수 있습니다.

이렇게 `소프트 타겟을 예측하도록 학습된 작은 모델이 큰 모델이 예측한 것과 유사한 예측을 하게되는 것이 지식 증류의 주요 아이디어`입니다.

> ### ⅰ. Distillation

![knowledge_distillation](https://github.com/eastk1te/eastk1te.github.io/assets/77319450/b0c3d065-5c28-4123-96d4-658f5e82e5e9){: w="500"}
_Figure 4 : KD distillation Overview_

보통 softmax의 출력 층을 사용하여 클래스 확률을 logit으로 아래와 같이 변환합니다. 여기서 T(tempreature)는 보통 1로 설정되는데, T를 높은 값으로 사용하면 softer한 확률 분포를 사용할 수 있습니다.

$$q_i = \frac{exp(z_i/T)}{\sum_jexp(z_j/T)}$$

간단한 형태의 지식 증류는 큰 모델에서 작은 모델로 지식을 전달하는 과정입니다. 증류된 모델은 transfer set에서 학습되고, 각 케이스의 soft target 분포를 사용합니다. 

여기서 모든 데이터에 대해 정확한 레이블이 알려진 경우 증류된 모델이 올바른 레이블을 생성하도록 학습함으로써 크게 개선될 여지가 있습니다. 이를 위한 한 가지 방법은 올바른 레이블을 사용하여 소프트 타겟을 수정하는 것입니다. 즉, 원본 모델의 출력(hard label)과 원래 모델의 출력(soft label)을 같이 사용해 "soft target"을 학습하는 것입니다.


해당 논문에서 두 개의 다른 목적 함수를 가중 평균함으로써 더 간단한 방법을 발견했다고 합니다 소프트 타겟 손실 함수는 다음과 같이 두 가지 주요 부분으로 구성됩니다.

$$L(x;W)=α∗H(y,σ(z_i;T=1))+β∗H(σ(z_t;T=τ),σ(z_i,T=τ))$$

첫 번째 항은 교사 모델로부터 생성된 소프트 타겟 분포와 학생 모델의 출력 분포 간의 교차 엔트로피로 학생 모델이 교사 모델의 예측에 가까워지도록 유도합니다. 

두 번쨰 부분은 학생 모델의 출력 분포와 더 높은 온도로 조정된 교사 모델의 출력 분포 간의 교차 엔트로피로 학생 모델이 더 부드러운 분포를 학습하여 더 많은 정보를 제공하도록 유도합니다.

아래와 같이 전이 집합의 각 케이스는 크로스 엔트로피 그래디언트($$dL/dz_i$$)에 기여합니다. 

$$\frac{\partial L}{\partial z_i} = \frac{1}{T}(q_i - p_i) = \frac{1}{T}(\frac{exp(z_i/T)}{\sum_jexp(z_j/T)} - \frac{exp(v_i/T)}{\sum_jexp(v_j/T)})$$

번거로운 모델의 로짓이 $$v_i$$이라고 할때, 로짓의 크기에 비해 온도가 높다면 아래와 같이 근사할 수 있습니다.

$$\frac{\partial L}{\partial z_i} \approx \frac{1}{T}(\frac{1 + z_i/T}{N + \sum_jz_j/T} - \frac{1 + v_i/T}{N + \sum_jv_j/T})$$

만약 각 전달 사례에 대해 개별적으로 0으로 평균화되었다고 가정하면 아래와 같이 됩니다.

$$\frac{\partial L}{\partial z_i} \approx \frac{1}{NT^2}(z_i - v_i)$$

따라서, 높은 온도 limit에서 증류는 위의 공식을 최소화하는 것과 동일합니다. 이는 크로스 엔트로피 손실에 대한 근사값을 제공하며 T가 클수록 그래디언트가 낮아져 부드러워(평평해)집니다. 즉, T가 작을수록(낮은 온도) 출력 간의 차이가 크게되어 높은 온도에서 부드러운 확률 분포를 사용함으로써 최소화합니다. 

> ### ⅱ. Code

```python

outputs_teacher_train = get_outputs(teacher_model, trainloader)
outputs_teacher_val = get_outputs(teacher_model, valloader)

train_kd(model, outputs_teacher_train, 
         optim.Adam(net.parameters()),loss_kd,trainloader, 
         temparature, alpha)

def train_kd(model, teacher_out, optimizer, loss_kd, dataloader, temparature, alpha):
   model.train()
   for i,(images, labels) in enumerate(dataloader):
      ...
      optimizer.zero_grad()
      outputs = model(inputs)
      outputs_teacher = torch.from_numpy(teacher_out[i]).to(device)

      # KD loss
      loss = nn.KLDivLoss()(F.log_softmax(outputs/temparature, 
             dim=1),F.softmax(outputs_teacher/temparature,dim=1)) * 
             (alpha * temparature * temparature) + 
             F.cross_entropy(outputs, labels) * (1. — alpha)

      loss.backward()
      optimizer.step()
   
```


> ## Ⅷ. REFERENCES

1. [A Survey on Model Compression for Large Language Models](https://arxiv.org/pdf/2308.07633)
2. [G. Hinton at el, "Distilling the Knowledge in a Neural Network", NIPS 2014](https://arxiv.org/abs/1503.02531)
3. [temp](https://het-shah.github.io/blog/2020/Knowledge-Distillation/)
4. [temp](https://intellabs.github.io/distiller/knowledge_distillation.html)
5. [Z. Liu at el, "Post-Training Quantization for Vision Transformer", NeurIPS 2021](https://arxiv.org/abs/2106.14156)
6. [torch.nn.utils.prune](https://pytorch.org/docs/stable/_modules/torch/nn/utils/prune.html#ln_structured)
7. [torch quantization](https://pytorch.org/docs/stable/quantization.html)

<br><br>
---