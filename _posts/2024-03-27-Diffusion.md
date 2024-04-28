---
title: '[Paper]Diffusion'
author: east
date: 2024-01-01 00:00:00 +09:00
categories: [Paper, Diffusion]
tags: [Paper, Diffusion]
math: true
mermaid: true
# pin: true
# image:
#    path: image preview url
#    alt: image preview text
---

2015 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585)

2021 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
Low-Rank adptation 기존 모델에 새로운 피사체를 학습시키는 추가 학습 기법의 일종.

2022 [R.Rombach at el, "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR, 2022](https://arxiv.org/abs/2112.10752)

2023 [Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation](https://arxiv.org/pdf/2303.00848.pdf)

2023 [L. Zhang at el, "Adding Conditional Control to Text-to-Image Diffusion Models", 2023](https://arxiv.org/pdf/2302.05543.pdf)

2024 Stable Diffusion 3 [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206)
[Stable Diffusion 3 : Research Paper](https://stability.ai/news/stable-diffusion-3-research-paper)







LLM 학습 기술. continusal learning RAFT SFT DPO Knowledge distilation
할루시네이션 방지 언어 모델 활용 기법 ReRank RAG RAT USP Reflextion ReAct
quantization, prunning 기법







<script src="https://gist.github.com/eastk1te/3902d14cfc94582e8097d68715d28519.js"></script>

작성시 목차에 의존하지 말고 내가 직접 생각하면서 정리하기!
만약 목차대로 작성한다면 그냥 "깜지"랑 다를게 없지 않은가?

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