---
title: '[Paper]Langchain'
author: east
date: 2024-01-17 00:00:00 +09:00
categories: [Paper, LLM]
tags: [Paper, LLM]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
---

chat gpt 개량

fine tunning
n-shot learning n개의 출력 예시를 제시하여 딥러닝이 용도에 알맞은 출력을 하도록 조정.
in context learning 문맥을 제시하고 문맥 기반으로 모델이 출력하도록 조정



https://www.youtube.com/watch?v=ehP4vphl_Us&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=14

[Langchain](https://revf.tistory.com/280)

작성시 목차에 의존하지 말고 내가 직접 생각하면서 정리하기!
만약 목차대로 작성한다면 그냥 "깜지"랑 다를게 없지 않은가?

RAG 시스템.
Retriever augmented generation 검색 증강 생성 설명

PDF 챗봇 구축.
문서 업로드 -> 문서분할 -> 문서 임베딩 -> 임베딩 검색 -> 답변 생성.
문서 임베딩

https://medium.com/getpieces/how-to-build-a-langchain-pdf-chatbot-b407fcd750b9


인코더는 말을 잘 이해함
디코더는 말을 잘함. LLM은 디코더 중심의 빠른 발전
LLM 


RAG 외부 데이터를 참조해아 ㅕLLM이 답변할 수 있도록 해주는 프레임워크.
Langchain
document loaders text spliltters vector embeddings -> vector storage -> retrievers -> chain -> LLM이 문장 생성.

```python

$ pip install openai
$ pip install langchain

chat = ChatOpenAI(openai_api_key=openai_api_key)


export OPENAI_API_KEY="SK-..."

import os
os.environ["OPENAI_API_KEY"] = "..."

```



{: .prompt-info }
해당 포스트는 공부를 위해 개인적으로 정리한 내용으로 해당 도서에는 다양한 예시를 통해 좀 더 직관적인 이해가 가능합니다.

ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ
ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻ
⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓
⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿
Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, Μ μ, Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ/ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω.⋅
<!-- align, equation, matrix, array, theorem, proof -->


<!-- $$
\begin{align}
z_uz_v^T \approx & \text{graph의 u,v 에서 rankdom walk가} \\
&\text{함께 나타날 확률}
\end{align}
$$ -->

<!-- 
$$
\begin{align*}
\text{수식의 왼쪽 항} &= \text{오른쪽 항의 첫 번째 줄} \\
&= \text{오른쪽 항의 두 번째 줄}
\end{align*}
$$ 
\partial
$$\max\limits_{f}$$
-->

<!-- Categories -->
Statistics 
   - 대과목
Course
   - 주제
Open Source 
   - Markdown : marp, Latex, Mermaid
Paper

ML
   - Graph
python 
   - Guide : PEP
   - library : argparse
Challenge
   - Certificate
   - Project
DevOps 
   - OS : linux
   - Version Management : git
   - Container : Docker, Kube
ETC.
   - BLOG
   - Career
   - Book Review


ML 트리에 맞춰서 블로그 포스트 작성
Machine Learning
├── Supervised
│   ├── Dimensionality Reduction
│   │   └── Linear Discriminant Analysis (LDA)
│   ├── Regression
│   │   ├── Linear Regression
│   │   ├── Multivariate Adaptive Regression Splines (MARS)
│   │   ├── K-Nearest Neighbors Regression (KNN)
│   │   ├── Random Forest Regression
│   │   ├── Decision Tree Regression (CART)
│   │   ├── Support Vector Regression (SVR)
│   │   └── Locally Weighted Scatterplot Smoothing (LOWESS)
│   └── Classification
│       ├── Decision Tree Classification (CART)
│       ├── Random Forest Classification
│       ├── Adaptive Boosting (AdaBoost)
│       ├── Gradient Boosted Trees
│       ├── Extreme Gradient Boosting (XGBoost)
│       ├── K-Nearest Neighbors Classification (KNN)
│       ├── Logistic Regression
│       ├── Naive Bayes
│       └── Support Vector Machines (SVM)
├── Neural Networks
│   ├── Generative Adversarial Network
│   │   ├── Wasserstein GAN (WGAN)
│   │   ├── Cycle GAN
│   │   ├── Deep Convolutional GAN (DCGAN)
│   │   ├── Conditional GAN (cGAN)
│   │   └── Generative Adversarial Network (GAN)
│   ├── Autoencoders
│   │   ├── Sparse Autoencoder (SAE)
│   │   ├── Denoising Autoencoder (DAE)
│   │   ├── Variational Autoencoder (VAE)
│   │   └── Undercomplete Autoencoder (AE)
│   ├── Recurrent Neural Networks
│   │   ├── Recurrent Neural Network (RNN)
│   │   ├── Long Short Term Memory (LSTM)
│   │   └── Gated Recurrent Unit (GRU)
│   ├── Feedforward Neural Networks
│   │   ├── Deep Feed Forward (DFF)
│   │   └── Feed Forward (FF)
│   └── Convolutional Neural Networks
│       ├── Transposed Convolutional Network
│       └── Deep Convolutional Network (DCN)
├── Unsupervised
│   ├── Dimensionality Reduction
│   │   ├── LLE Embedding (Locally Linear Embedding)
│   │   ├── t-SNE (t-Distributed Stochastic Neighbor Embedding)
│   │   ├── UMAP (Uniform Manifold Approximation and Projection)
│   │   ├── PCA (Principal Component Analysis)
│   │   ├── MDS (Multidimensional Scaling)
│   └── Clustering
│       ├── K-Means
│       ├── Gaussian Mixture Models (GMM)
│       ├── Hierarchical Agglomerative Clustering (HAC)
│       └── DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
├── Reinforcement
│   ├── Value Based Method
│   │   ├── SARSA (State-Action-Reward-State-Action)
│   │   ├── Q-Learning
│   │   └── Deep Q Neural Network (DQN)
│   └── Policy Based Method
│       ├── Policy Gradient (REINFORCE)
│       └── Proximal Policy Optimization (PPO)
├── Semi-Supervised
│   ├── Label Spreading
│   ├── Self Training Classifier
│   └── Label Propagation
└── Other
    └── Probabilistic Graphical Models
        └── Bayesian Belief Networks (BBN)

출처 : https://towardsdatascience.com/denoising-autoencoders-dae-how-to-use-neural-networks-to-clean-up-your-data-cd9c19bc6915



> ## . REFERENCES

1. temp
2. temp
3. temp
4. temp


<br><br>
---