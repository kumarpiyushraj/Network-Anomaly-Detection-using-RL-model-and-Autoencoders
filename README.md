<!----------------------------------------------------------------------------->
<!--  HERO — Full-width waving banner                                         -->
<!----------------------------------------------------------------------------->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020d1f,20:0a1628,55:0d2d6e,85:1e40af,100:60a5fa&height=280&section=header&text=NetGuard%20AI&fontSize=72&fontColor=dbeafe&fontAlignY=38&fontStyle=bold&desc=Network%20Anomaly%20Detection%20%E2%80%94%20Autoencoders%20%2B%20PPO%20Reinforcement%20Learning&descAlignY=60&descSize=17&descColor=93c5fd&animation=fadeIn" width="100%"/>

</div>

<!----------------------------------------------------------------------------->
<!--  BADGES                                                                  -->
<!----------------------------------------------------------------------------->

<div align="center">

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-0d1f3c?style=for-the-badge&logo=python&logoColor=60a5fa&labelColor=071020&color=0d1f3c)](https://www.python.org/)&nbsp;
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-0d1f3c?style=for-the-badge&logo=tensorflow&logoColor=fb923c&labelColor=071020&color=0d1f3c)](https://www.tensorflow.org/)&nbsp;
[![Stable-Baselines3](https://img.shields.io/badge/RL-PPO%20%7C%20SB3-0d1f3c?style=for-the-badge&logo=openai&logoColor=34d399&labelColor=071020&color=0d1f3c)](https://stable-baselines3.readthedocs.io/)&nbsp;
[![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-0d1f3c?style=for-the-badge&logo=openaigym&logoColor=a78bfa&labelColor=071020&color=0d1f3c)](https://gymnasium.farama.org/)&nbsp;
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-0d1f3c?style=for-the-badge&logo=databricks&logoColor=f472b6&labelColor=071020&color=0d1f3c)](https://www.unb.ca/cic/datasets/ids-2017.html)

<br/><br/>

*Detect threats intelligently &nbsp;·&nbsp; Learn from network flows &nbsp;·&nbsp; Adapt through reward-driven decisions*

<br/><br/>

</div>

<!----------------------------------------------------------------------------->
<!--  STATS STRIP                                                             -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:060e20,100:0a1a3a&height=90&text=Unsupervised%20Feature%20Learning%20%C2%B7%20Adaptive%20RL%20Decision%20Boundary%20%C2%B7%20Multi-Threshold%20Evaluation%20%C2%B7%20~92%25%20F1-Score&fontSize=14&fontColor=93c5fd&fontAlignY=35&desc=Three%20autoencoder%20architectures%20%E2%80%94%20one%20unified%20PPO%20agent%20%E2%80%94%20trained%20on%202.8M%2B%20real%20network%20flows&descSize=13&descColor=dbeafe&descAlignY=68" width="100%"/>

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  AT A GLANCE                                                             -->
<!----------------------------------------------------------------------------->

<div align="center">

### 📊 &nbsp;At a Glance

| 🧠 Autoencoders | ⚙️ RL Algorithm | 📐 Latent Dim | 📦 Dataset | 🎯 Best F1 | 🏆 Peak Reward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **FF-AE · Conv-AE · DAE** | **PPO (MLP Policy)** | **32 – 64** | **CICIDS-2017** | **~0.92** | **4932** |

</div>

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  TABLE OF CONTENTS                                                       -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f1e3d,100:020710&height=64&text=%F0%9F%93%8B%20%20Table%20of%20Contents&fontSize=22&fontColor=dbeafe&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<div align="center">

| # | Section | # | Section |
|:---:|:---|:---:|:---|
| 01 | <a href="#overview">🛡️ Project Overview</a> | 07 | <a href="#rl-environment">🎮 RL Environment Design</a> |
| 02 | <a href="#pipeline">🔄 Three-Stage Pipeline</a> | 08 | <a href="#evaluation">📊 Evaluation & Results</a> |
| 03 | <a href="#ff-ae">⚡ Feedforward Autoencoder</a> | 09 | <a href="#dataset">📁 Dataset Summary</a> |
| 04 | <a href="#conv-ae">🔷 Convolutional Autoencoder</a> | 10 | <a href="#quickstart">🚀 Quick Start</a> |
| 05 | <a href="#dae">🌀 Denoising Autoencoder</a> | 11 | <a href="#future">🔮 Future Enhancements</a> |
| 06 | <a href="#ppo-agent">🤖 PPO Agent Training</a> | 12 | <a href="#references">📜 References</a> |

</div>

<br/><br/>

<a name="overview"></a>
<!----------------------------------------------------------------------------->
<!--  PROJECT OVERVIEW                                                        -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1e4a,100:020810&height=64&text=%F0%9F%9B%A1%EF%B8%8F%20%20Project%20Overview&fontSize=22&fontColor=93c5fd&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

**NetGuard AI** is a hybrid **Network Intrusion Detection System (NIDS)** that fuses **unsupervised deep representation learning** with **adaptive reinforcement learning** to accurately classify network traffic as *benign* or *malicious*. Rather than relying on hand-crafted features or static thresholds, the system autonomously discovers compact latent representations of raw network flows and trains a policy-gradient agent to make detection decisions through **reward-driven optimization**.

<br/>

<div align="center">

| &nbsp; | What makes it different? | How it achieves it |
|:---:|:---|:---|
| 🧠 | **Deep Representation Learning** | Three autoencoder architectures compress 76 raw features into a 32–64 dimensional latent space |
| 🎮 | **Adaptive Decision Boundary** | PPO RL agent learns to detect anomalies through structured reward signals — not static thresholds |
| ⚖️ | **Asymmetric Reward Shaping** | Missed attacks penalized more heavily than false alarms to prioritize security recall |
| 📐 | **Multi-Threshold Evaluation** | Final detection evaluated at thresholds 0.25, 0.30, and 0.35 for fine-grained control |
| 📦 | **Large-Scale Validation** | Trained and tested on 2.8M+ real network flow records from the CICIDS-2017 benchmark |

</div>

<br/><br/>

<a name="pipeline"></a>
<!----------------------------------------------------------------------------->
<!--  THREE-STAGE PIPELINE                                                    -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d2060,100:030816&height=64&text=%F0%9F%94%84%20%20Three-Stage%20Detection%20Pipeline&fontSize=22&fontColor=a5b4fc&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The project implements a tightly integrated, end-to-end pipeline where each stage feeds into the next:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1 — Feature Extraction                       │
│                                                                         │
│   Raw Network Flow (76 features)                                        │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│   │  Feedforward  │  │ Convolutional │  │   Denoising   │             │
│   │  Autoencoder  │  │  Autoencoder  │  │  Autoencoder  │             │
│   │  (FF-AE)      │  │  (Conv-AE)    │  │  (DAE)        │             │
│   └───────┬───────┘  └───────┬───────┘  └───────┬───────┘             │
│           │                  │                  │                      │
│           └──────────────────┴──────────────────┘                      │
│                              │                                          │
│                   Latent Vector (32–64 dims)                            │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                      STAGE 2 — RL Agent Training                        │
│                                                                         │
│   Custom Gymnasium Environment                                          │
│   Observation: Latent Vector │ Actions: {Benign=0, Malicious=1}        │
│   Reward: +2 correct · −3/−4 missed attack · −1/−3 false alarm         │
│                                                                         │
│              PPO Agent (MlpPolicy [128, 128])                           │
│              120,000 total timesteps per training run                   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                      STAGE 3 — Evaluation                               │
│                                                                         │
│   Argmax Policy + Custom Threshold (0.25 / 0.30 / 0.35)               │
│   Precision · Recall · F1-Score · Confusion Matrix · PR-AUC            │
└─────────────────────────────────────────────────────────────────────────┘
```

<br/><br/>

<a name="ff-ae"></a>
<!----------------------------------------------------------------------------->
<!--  FEEDFORWARD AUTOENCODER                                                -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:1a2e6a,100:04091a&height=64&text=%E2%9A%A1%20%20Stage%201A%20%E2%80%94%20Feedforward%20Autoencoder%20(FF-AE)&fontSize=20&fontColor=bfdbfe&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The **Feedforward Autoencoder** serves as the baseline architecture — a dense encoder-decoder stack that learns to compress and reconstruct high-dimensional network traffic features.

<br/>

### 🏗️ Architecture

```python
# Encoder
Input(76)  →  Dense(128, relu) + Dropout(0.3) + L2(1e-5)
           →  Dense(64,  relu) + Dropout(0.3) + L2(1e-5)
           →  Dense(32,  relu)                            # ← Latent Space

# Decoder
Dense(64,  relu) + Dropout(0.3)
Dense(128, relu)
Dense(76,  linear)                                        # Reconstruction
```

<br/>

<div align="center">

| Hyperparameter | Value |
|:---|:---:|
| **Input Dimensionality** | 76 features |
| **Latent Dimensionality** | **32** |
| **Optimizer** | Adam (`lr=1e-4`, `clipvalue=1.0`) |
| **Loss Function** | MSE |
| **Epochs** | 50 (Early Stopping: patience=5) |
| **Batch Size** | 2048 |
| **Regularization** | L2 (`1e-5`) + Dropout (`0.3`) |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, min_lr=1e-6) |

</div>

<br/>

### 🎯 PPO Configuration (FF-AE Pipeline)

The FF-AE latent encoder feeds a PPO agent with a **balanced reward structure** — equal penalties for missed attacks and false alarms:

```python
# Balanced Reward Shaping (FF-AE variant)
reward = +2   # Correct classification (benign or malicious)
reward = -3   # Any misclassification (missed attack OR false alarm)
```

**PPO Hyperparameters:**

```python
PPO(
    learning_rate = 1e-3,
    n_steps       = 1024,
    batch_size    = 512,
    gamma         = 0.995,
    clip_range    = 0.2,
    ent_coef      = 0.04,
    net_arch      = {"pi": [128, 128], "vf": [128, 128]}
)
```

<br/><br/>

<a name="conv-ae"></a>
<!----------------------------------------------------------------------------->
<!--  CONVOLUTIONAL AUTOENCODER                                              -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:1e1a60,100:060410&height=64&text=%F0%9F%94%B7%20%20Stage%201B%20%E2%80%94%20Convolutional%20Autoencoder%20(Conv-AE)&fontSize=20&fontColor=c4b5fd&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The **Convolutional Autoencoder** treats each network flow record as a **1D signal**, applying convolutional filters to extract local spatial patterns across feature dimensions. This captures correlational structure between adjacent features that dense layers may miss.

<br/>

### 🏗️ Architecture

```python
# Input Reshape: (76,) → (76, 1)  — treated as 1D signal

# Encoder
Input(76, 1)  →  Conv1D(32, kernel=3, relu, same)
              →  MaxPooling1D(2)
              →  Conv1D(16, kernel=3, relu, same)
              →  MaxPooling1D(2)
              →  Flatten()
              →  Dense(32, relu)                    # ← Latent Space

# Decoder
Dense(flattened_dim, relu)
Reshape → UpSampling1D(2) → Conv1D(16, relu, same)
        → UpSampling1D(2) → Conv1D(1,  sigmoid, same)   # Reconstruction
```

<br/>

<div align="center">

| Hyperparameter | Value |
|:---|:---:|
| **Input Shape** | `(76, 1)` — 1D signal |
| **Conv Filters** | 32 → 16 (encoder) |
| **Latent Dimensionality** | **32** |
| **Optimizer** | Adam (`lr=1e-4`, `clipvalue=1.0`) |
| **Loss Function** | MSE |
| **Epochs** | 50 (Early Stopping: patience=5) |
| **Batch Size** | **4096** |
| **Val Loss Converged** | **~0.67** |

</div>

<br/>

### 🎯 PPO Configuration (Conv-AE Pipeline)

The Conv-AE pipeline uses an **asymmetric reward** that penalizes **missed attacks more severely** than false alarms:

```python
# Asymmetric Reward Shaping (Conv-AE variant)
reward = +2   # Correct classification
reward = -4   # Missed attack   (true_label=1, action=0) — security-critical
reward = -1   # False alarm     (true_label=0, action=1)
```

**PPO Hyperparameters:**

```python
PPO(
    learning_rate = 5e-4,
    n_steps       = 512,
    batch_size    = 256,
    gamma         = 0.995,
    gae_lambda    = 0.95,
    vf_coef       = 1.0,
    clip_range    = 0.2,
    ent_coef      = 0.02,
    net_arch      = {"pi": [128, 128], "vf": [128, 128]}
)
```

> 💡 **Key Insight:** The Conv-AE encoder reshapes each sample to `(features, 1)` before inference, enabling the convolutional encoder to properly generate latent vectors for the RL environment's observation space.

<br/><br/>

<a name="dae"></a>
<!----------------------------------------------------------------------------->
<!--  DENOISING AUTOENCODER                                                  -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:1a0e5a,100:040210&height=64&text=%F0%9F%8C%80%20%20Stage%201C%20%E2%80%94%20Denoising%20Autoencoder%20(DAE)&fontSize=20&fontColor=ddd6fe&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The **Denoising Autoencoder** is the most robust of the three architectures. It deliberately injects **Gaussian noise** (`σ=0.05`) into input flows during training, forcing the encoder to learn noise-invariant latent representations — capturing the *true* statistical structure of network traffic patterns rather than memorizing exact input values.

<br/>

### 🏗️ Architecture

```python
# Noise Injection Layer
Input(76)  →  GaussianNoise(0.05)                         # Deliberate corruption

# Encoder (learns to denoise)
→  Dense(128, relu) + Dropout(0.4) + L2(1e-5)
→  Dense(64,  relu) + Dropout(0.4) + L2(1e-5)
→  Dense(64,  relu)                                       # ← Latent Space (64-dim)

# Decoder (reconstructs clean signal)
→  Dense(64,  relu) + Dropout(0.4)
→  Dense(128, relu)
→  Dense(76,  sigmoid)                                    # Bounded reconstruction
```

<br/>

<div align="center">

| Hyperparameter | Value |
|:---|:---:|
| **Noise Layer** | `GaussianNoise(stddev=0.05)` |
| **Latent Dimensionality** | **64** (larger capacity) |
| **Dropout Rate** | **0.4** (higher than FF-AE) |
| **Decoder Activation** | `sigmoid` (bounded output) |
| **Optimizer** | Adam (`lr=1e-4`, `clipvalue=1.0`) |
| **Loss Function** | MSE |
| **Epochs** | **80** (extended training) |
| **Batch Size** | **4096** |
| **Val Loss Converged** | **~0.52** ✅ Best Reconstruction |

</div>

<br/>

> 🏆 **Best Performer:** The DAE achieved the **lowest validation loss (~0.52)** among all three autoencoders, demonstrating that noise-robust training produces superior latent representations — particularly valuable in real-world network environments where traffic patterns contain inherent variability and noise.

<br/>

### 🎯 PPO Configuration (DAE Pipeline)

The DAE pipeline shares the same asymmetric reward and refined PPO configuration as Conv-AE, with GAE-lambda tuning for better advantage estimation:

```python
PPO(
    learning_rate = 5e-4,
    n_steps       = 512,
    batch_size    = 256,
    gamma         = 0.995,
    gae_lambda    = 0.95,     # Generalized Advantage Estimation
    vf_coef       = 1.0,
    clip_range    = 0.2,
    ent_coef      = 0.02,
    net_arch      = {"pi": [128, 128], "vf": [128, 128]}
)
```

<br/>

### 📊 Autoencoder Comparison Summary

<div align="center">

| Architecture | Latent Dim | Dropout | Noise Injection | Decoder Activation | Val Loss | Strength |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **FF-AE** | 32 | 0.3 | ❌ | `linear` | — | Baseline, fast convergence |
| **Conv-AE** | 32 | ❌ | ❌ | `sigmoid` | **~0.67** | Spatial pattern capture |
| **DAE** | 64 | 0.4 | ✅ `σ=0.05` | `sigmoid` | **~0.52** ⭐ | Noise-robust, best features |

</div>

<br/><br/>

<a name="ppo-agent"></a>
<!----------------------------------------------------------------------------->
<!--  PPO AGENT                                                              -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d2a5e,100:020810&height=64&text=%F0%9F%A4%96%20%20PPO%20Agent%20%E2%80%94%20Policy%20Training%20Deep%20Dive&fontSize=22&fontColor=7dd3fc&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

Each autoencoder pipeline feeds its encoder into a **custom Gymnasium environment** where a **Proximal Policy Optimization (PPO)** agent learns to classify network flows through trial-and-error interaction.

<br/>

### 🏗️ Model Architecture

```
Latent Vector (32 or 64 dims)
       │
       ▼
  Dense(128, relu)      ← Policy Network (π)
       │                ← Value Network (V) — shared trunk
  Dense(128, relu)
       │
  ┌────┴────┐
  ▼         ▼
Policy     Value
Head       Head
(Discrete  (Scalar
 2-class)   reward
            estimate)
```

<br/>

### 🎮 Training Process

```
Total Timesteps : 120,000 per training run
Eval Frequency  : Every 4,000 steps (via EvalCallback)
Best Model Save : ./best_model/  (checkpointed by EvalCallback)
Logging         : TensorBoard → ./ppo_logs/
Max Steps/Ep    : 3,000 (each episode processes 3,000 network samples)
Device          : Auto (GPU if available, CPU fallback)
```

<br/>

### 📈 Training Behaviour Across All Pipelines

<div align="center">

| Training Signal | Observed Trend |
|:---|:---|
| **KL Divergence** | Stabilizes after initial policy updates |
| **Value Loss** | Decreases steadily, indicating improving state-value estimation |
| **Entropy** | Decreases over time → policy becomes more **confident** |
| **Explained Variance** | Improves across iterations |
| **Training Speed** | ~220 fps during training loops |

</div>

<br/>

### 🏆 PPO Mean Reward Results

High mean rewards across all three autoencoder pipelines confirm strong agent performance:

<div align="center">

| Pipeline | Mean Reward | Trend |
|:---:|:---:|:---:|
| FF-AE + PPO | **4415** | Stable convergence |
| DAE + PPO | **4839** | Improved latent quality |
| Conv-AE + PPO | **4932** ⭐ | Peak performance |

</div>

<br/><br/>

<a name="rl-environment"></a>
<!----------------------------------------------------------------------------->
<!--  RL ENVIRONMENT                                                         -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:12205e,100:030710&height=64&text=%F0%9F%8E%AE%20%20Custom%20RL%20Environment%20Design&fontSize=22&fontColor=bfdbfe&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

### `AnomalyDetectionEnv` — Gymnasium-Compatible

A custom `gym.Env` wraps the autoencoder encoder and the labeled dataset, presenting each network sample as an **observation** and rewarding correct binary classification decisions.

<br/>

```python
class AnomalyDetectionEnv(gym.Env):
    """
    Observation  : 32 or 64-dim latent vector (from encoder)
    Action Space : Discrete(2) — {0: Benign, 1: Malicious}
    Episode      : 3000 sequential samples per episode

    Reward Structure (Conv-AE / DAE variant):
        +2  →  Correct classification (any class)
        -4  →  Missed attack  (false negative — security-critical!)
        -1  →  False alarm    (false positive)
    """

    def _get_encoded_state(self, idx):
        # Encodes the raw sample on-the-fly using the frozen encoder
        return self.encoder(self.X[idx:idx+1]).numpy()[0]
```

<br/>

### 🔄 Episode Flow

```
reset() → Sample index = 0, Step count = 0
    │
    ▼
step(action):
    1. Fetch true_label from dataset
    2. Compute reward (correct/missed/false alarm)
    3. Advance to next sample (circular indexing)
    4. Encode next sample → new observation
    5. Return (obs, reward, done, False, {})
    │
    ▼
done when step_count >= 3000
```

<br/>

### ⚖️ Reward Philosophy

The **asymmetric reward structure** in the Conv-AE and DAE pipelines explicitly encodes a security-first philosophy:

<div align="center">

| Scenario | Action | True Label | Reward | Why |
|:---|:---:|:---:|:---:|:---|
| Correct benign detection | 0 | 0 | **+2** | True Negative — no threat missed |
| Correct attack detection | 1 | 1 | **+2** | True Positive — threat caught |
| **Missed attack** | **0** | **1** | **−4** | **False Negative — most dangerous!** |
| False alarm | 1 | 0 | **−1** | False Positive — minor disruption |

</div>

> 🔐 **Design Rationale:** Undetected attacks cause far greater harm than false alarms in real network environments. The `−4` penalty for missed attacks forces the agent to prioritize **recall** over precision during learning.

<br/><br/>

<a name="evaluation"></a>
<!----------------------------------------------------------------------------->
<!--  EVALUATION & RESULTS                                                   -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1e50,100:020610&height=64&text=%F0%9F%93%8A%20%20Evaluation%20%26%20Results&fontSize=22&fontColor=93c5fd&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

Evaluation is performed in two modes: **Argmax Policy** (deterministic greedy action selection) and **Custom Threshold** (probability-based, adjustable sensitivity).

<br/>

### 🔍 Evaluation Modes

```python
# Mode 1 — Argmax (Deterministic)
action, _ = model.predict(obs, deterministic=True)

# Mode 2 — Custom Threshold (Probabilistic)
def predict_with_threshold(model, obs, threshold=0.3):
    distribution = model.policy.get_distribution(obs_tensor)
    probs = distribution.distribution.probs   # [batch, 2]
    return (probs[:, 1] > threshold).long()   # Malicious prob > threshold
```

<br/>

### 📐 Threshold Analysis

Three thresholds were systematically evaluated to find the optimal precision-recall trade-off:

<div align="center">

#### Threshold = 0.25 — High Sensitivity

| Metric | Score |
|:---:|:---:|
| **Precision** | ~0.91 |
| **Recall** | ~0.92 |
| **F1-Score** | ~0.92 |

*Slightly more liberal — catches more attacks, slightly more false alarms.*

<br/>

#### Threshold = 0.30 — Optimal Balance ⭐

| Metric | Score |
|:---:|:---:|
| **Precision** | ~0.92 |
| **Recall** | ~0.92 |
| **F1-Score** | ~0.92 |

*Best overall balance — recommended default threshold for deployment.*

<br/>

#### Threshold = 0.35 — High Precision

| Metric | Score |
|:---:|:---:|
| **Precision** | ~0.92 |
| **Recall** | ~0.89 |
| **F1-Score** | ~0.90 |

*More conservative — fewer false alarms, but slightly more missed attacks.*

</div>

<br/>

### 🔬 Key Findings

<div align="center">

| Observation | Detail |
|:---|:---|
| 🎯 **Optimal Threshold** | `0.30` delivers the best Precision-Recall balance across all pipeline variants |
| 🔴 **False Positives** | Remain extremely low across all threshold values |
| 🏆 **RL vs. Reconstruction** | PPO significantly outperforms raw autoencoder reconstruction-error thresholds |
| 🧩 **Latent Quality Matters** | Better autoencoders (lower val_loss) directly improve RL stability and classification quality |
| 📈 **PR-AUC** | Precision-Recall curves plotted and AUC computed for both argmax and threshold modes |
| 🔄 **Consistent Performance** | Results are stable across multiple training runs (rewards: 4415 → 4839 → 4932) |

</div>

<br/><br/>

<a name="dataset"></a>
<!----------------------------------------------------------------------------->
<!--  DATASET                                                                -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0e1a48,100:030712&height=64&text=%F0%9F%93%81%20%20Dataset%20%E2%80%94%20CICIDS-2017&fontSize=22&fontColor=a5b4fc&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The **Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset (CICIDS-2017)** is a widely used benchmark for network intrusion detection research, providing realistic, labeled network traffic spanning a full week of captured flows.

<br/>

<div align="center">

| Property | Detail |
|:---|:---|
| 📊 **Total Records** | **2.8M+ network flow records** |
| 🔢 **Raw Features** | **80 flow-level features** |
| ✂️ **Features After Preprocessing** | **~76 numerical features** |
| 🏷️ **Label Types** | Binary: `0 = Benign`, `1 = Malicious` |
| ⚖️ **Class Imbalance** | Majority benign — requires careful reward design |
| 🌐 **Attack Categories** | DDoS, DoS (Hulk, GoldenEye, Slowloris), Brute Force, Web Attacks, Botnet, Port Scan, Infiltration |

</div>

<br/>

### 🔧 Preprocessing Pipeline

```python
# Step 1 — Column Standardization
df.columns = df.columns.str.strip().str.lower()

# Step 2 — Binary Label Encoding
df['label'] = df['label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)

# Step 3 — Drop Non-Numerical / Low-Value Columns
cols_to_drop = ['destination port', 'flow duration']
df.drop(cols_to_drop, axis=1)

# Step 4 — Handle Inf and NaN
df.replace([np.inf, -np.inf], np.nan)
df.dropna(axis=1, thresh=0.7 * len(df))   # Drop cols with >30% missing
df.fillna(df.median())                     # Fill remaining with column median

# Step 5 — Outlier Detection (IQR-based, diagnostic)
# IQR bounds computed per column for inspection

# Step 6 — StandardScaler Normalization
X_scaled = StandardScaler().fit_transform(X)

# Step 7 — Stratified Train/Test Split
X_train, X_test = train_test_split(X_scaled, test_size=0.2, stratify=y)

# Step 8 — Safety Clipping
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e5, neginf=-1e5)
```

> 📌 For the Conv-AE pipeline, training data is additionally reshaped to `(N, features, 1)` to match the 1D convolutional input requirement.

<br/><br/>

<a name="quickstart"></a>
<!----------------------------------------------------------------------------->
<!--  QUICK START                                                            -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a2258,100:020812&height=64&text=%F0%9F%9A%80%20%20Quick%20Start&fontSize=22&fontColor=7dd3fc&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

### Prerequisites

<div align="center">

| Requirement | Version |
|:---|:---:|
| 🐍 **Python** | 3.9+ |
| 🔧 **pip** | Latest recommended |
| 🖥️ **GPU** | Optional (CUDA-compatible for faster training) |
| 💾 **RAM** | 16GB+ recommended for 2.8M record dataset |

</div>

<br/>

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders
cd Network-Anomaly-Detection-using-RL-model-and-Autoencoders

# 2. Install all dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn \
            shimmy>=2.0 stable-baselines3[extra] gymnasium
```

<br/>

### Dataset Setup

```bash
# Create the Dataset directory
mkdir Dataset

# Place all CICIDS-2017 CSV files inside:
# Dataset/Monday-WorkingHours.pcap_ISCX.csv
# Dataset/Tuesday-WorkingHours.pcap_ISCX.csv
# ... and so on
```

<br/>

### Run the Notebook

```bash
# Launch Jupyter
jupyter notebook Network_Anomaly_Detection.ipynb
```

<br/>

The notebook is organized into **three sequential sections**, one for each autoencoder pipeline:

<div align="center">

| Section | Autoencoder | Notes |
|:---|:---:|:---|
| **Section 1** | FF-AE | Run top-to-bottom; trains encoder then PPO |
| **Section 2** | DAE | Independent pipeline; re-runs data loading |
| **Section 3** | Conv-AE | Requires extra reshape step for 1D conv input |

</div>

<br/>

### 📦 Full Requirements

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
gymnasium
shimmy>=2.0
stable-baselines3[extra]
torch   (pulled in by stable-baselines3)
```

<br/><br/>

<a name="future"></a>
<!----------------------------------------------------------------------------->
<!--  FUTURE ENHANCEMENTS                                                    -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0c1a50,100:020610&height=64&text=%F0%9F%94%AE%20%20Future%20Enhancements&fontSize=22&fontColor=93c5fd&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<details>
<summary><b>🧠 Advanced Architectures</b></summary>

<br/>

- [ ] **Variational Autoencoders (VAE)** — probabilistic latent space for uncertainty quantification
- [ ] **Transformer-based encoders** — attention over feature sequences for richer representations
- [ ] **Graph Neural Networks** — model inter-flow correlations as graph edges
- [ ] **Sparse Autoencoders** — enforce sparsity constraints for more interpretable latent features

<br/>
</details>

<details>
<summary><b>🤖 Reinforcement Learning Improvements</b></summary>

<br/>

- [ ] **Multi-agent RL** — cooperative agents handling different traffic slices simultaneously
- [ ] **Hierarchical RL** — macro-agent selects detection strategy, micro-agent classifies flows
- [ ] **Prioritized Experience Replay** — oversample rare attack types during training
- [ ] **Curiosity-driven exploration** — intrinsic rewards for novel network patterns

<br/>
</details>

<details>
<summary><b>🚀 Deployment & Real-Time Integration</b></summary>

<br/>

- [ ] **Real-time Suricata / Zeek integration** — process live PCAP streams
- [ ] **Online learning mode** — continuously update model weights on streaming traffic
- [ ] **Model quantization** — reduce inference latency for edge deployment
- [ ] **REST API / gRPC endpoint** — serve detection as a microservice
- [ ] **Multi-class detection** — distinguish between attack categories (DDoS, Brute Force, etc.)

<br/>
</details>

<br/><br/>

<a name="references"></a>
<!----------------------------------------------------------------------------->
<!--  REFERENCES                                                             -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1840,100:020610&height=64&text=%F0%9F%93%9C%20%20References&fontSize=22&fontColor=bfdbfe&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

```bibtex
@dataset{cic_ids2017,
  author    = {Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A},
  title     = {Intrusion Detection Evaluation Dataset (CICIDS2017)},
  year      = {2017},
  publisher = {Canadian Institute for Cybersecurity},
  url       = {https://www.unb.ca/cic/datasets/ids-2017.html}
}

@article{ppo_schulman2017,
  author    = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  title     = {Proximal Policy Optimization Algorithms},
  year      = {2017},
  journal   = {arXiv preprint},
  url       = {https://arxiv.org/abs/1707.06347}
}

@software{stable_baselines3,
  author    = {Raffin, Antonin and Hill, Ashley and Gleave, Adam and Kanervisto, Anssi and Ernestus, Maximilian and Dormann, Noah},
  title     = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  year      = {2021},
  journal   = {Journal of Machine Learning Research},
  url       = {https://jmlr.org/papers/v22/20-1364.html}
}
```

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  ACKNOWLEDGMENTS                                                        -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:091530,100:020510&height=64&text=%F0%9F%99%8F%20%20Acknowledgments&fontSize=22&fontColor=93c5fd&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<div align="center">

| &nbsp; | Acknowledgment |
|:---:|:---|
| 🎓 | **Canadian Institute for Cybersecurity** — For releasing the CICIDS-2017 benchmark dataset |
| 🤝 | **Stable-Baselines3 Team** — For production-grade PPO implementations |
| 🧠 | **TensorFlow / Keras Team** — For the deep learning framework powering the autoencoders |
| 🎮 | **Farama Foundation** — For the Gymnasium RL environment standard |
| 🌟 | **Open Source ML Community** — For the ecosystem that makes this research reproducible |

</div>

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  CONTACT                                                                -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f2050,100:030810&height=64&text=%F0%9F%93%9E%20%20Contact%20%26%20Support&fontSize=22&fontColor=bfdbfe&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<div align="center">

**Questions? Issues? Contributions?**

<br/>

[![GitHub Issues](https://img.shields.io/badge/Issues-Report%20Bug-ef4444?style=for-the-badge&logo=github)](https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders/issues)&nbsp;
[![GitHub Discussions](https://img.shields.io/badge/Discussions-Ask%20Question-3b82f6?style=for-the-badge&logo=github)](https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders/discussions)&nbsp;
[![Email](https://img.shields.io/badge/Email-Contact%20Developer-22c55e?style=for-the-badge&logo=gmail)](mailto:kmpiyushraj@gmail.com)

</div>

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  FOOTER                                                                 -->
<!----------------------------------------------------------------------------->

<div align="center">

<br/>

**Built with ❤️ for intelligent network security &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Stable-Baselines3 &nbsp;·&nbsp; Gymnasium**

<br/>

[![Star this repo](https://img.shields.io/badge/⭐%20Star%20this%20repo-If%20it%20helps%20your%20research-60a5fa?style=for-the-badge&logo=github&labelColor=0d1117)](https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders)

<br/>

*© 2025 Kumar Piyush Raj &nbsp;·&nbsp; [GitHub @kumarpiyushraj](https://github.com/kumarpiyushraj)*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:60a5fa,40:1e40af,70:0a1628,100:020d1f&height=160&section=footer" width="100%"/>

</div>
