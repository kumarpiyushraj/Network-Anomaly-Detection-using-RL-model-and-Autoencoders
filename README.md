<!----------------------------------------------------------------------------->
<!--  HERO — Full-width waving banner                                         -->
<!----------------------------------------------------------------------------->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020f14,20:053428,55:0b6b55,85:0e9e7e,100:2dd4bf&height=280&section=header&text=NetGuard%20AI&fontSize=72&fontColor=ccfbf1&fontAlignY=38&fontStyle=bold&desc=Network%20Anomaly%20Detection%20via%20Autoencoders%20and%20PPO%20Reinforcement%20Learning&descAlignY=60&descSize=17&descColor=99f6e4&animation=fadeIn" width="100%"/>

</div>

<!----------------------------------------------------------------------------->
<!--  BADGES                                                                  -->
<!----------------------------------------------------------------------------->

<div align="center">

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-0a1f1c?style=for-the-badge&logo=python&logoColor=2dd4bf&labelColor=061410&color=0a1f1c)](https://www.python.org/)&nbsp;
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-0a1f1c?style=for-the-badge&logo=tensorflow&logoColor=fb923c&labelColor=061410&color=0a1f1c)](https://www.tensorflow.org/)&nbsp;
[![Stable-Baselines3](https://img.shields.io/badge/RL-PPO%20%7C%20SB3-0a1f1c?style=for-the-badge&logo=openai&logoColor=34d399&labelColor=061410&color=0a1f1c)](https://stable-baselines3.readthedocs.io/)&nbsp;
[![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-0a1f1c?style=for-the-badge&logo=openaigym&logoColor=a7f3d0&labelColor=061410&color=0a1f1c)](https://gymnasium.farama.org/)&nbsp;
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS--2017-0a1f1c?style=for-the-badge&logo=databricks&logoColor=5eead4&labelColor=061410&color=0a1f1c)](https://www.unb.ca/cic/datasets/ids-2017.html)

<br/><br/>

*Detect threats intelligently &nbsp;·&nbsp; Learn from network flows &nbsp;·&nbsp; Adapt through reward-driven decisions*

<br/><br/>

</div>

<!----------------------------------------------------------------------------->
<!--  STATS STRIP                                                             -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:050f0d,100:091a16&height=90&text=Unsupervised%20Feature%20Learning%20%C2%B7%20Adaptive%20RL%20Decision%20Boundary%20%C2%B7%20Multi-Threshold%20Evaluation%20%C2%B7%20~92%25%20F1-Score&fontSize=14&fontColor=5eead4&fontAlignY=35&desc=Three%20autoencoder%20architectures%20%E2%80%94%20one%20unified%20PPO%20agent%20%E2%80%94%20trained%20on%202.8M%2B%20real%20network%20flows&descSize=13&descColor=ccfbf1&descAlignY=68" width="100%"/>

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  AT A GLANCE                                                             -->
<!----------------------------------------------------------------------------->

<div align="center">

### 📊 &nbsp;At a Glance

| 🧠 Autoencoders | ⚙️ RL Algorithm | 📐 Latent Dim | 📦 Dataset | 🎯 Best F1 | 🏆 Peak Reward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **FF-AE · DAE · Conv-AE** | **PPO (MlpPolicy)** | **32 (FF/Conv) · 64 (DAE)** | **CICIDS-2017** | **~0.92** | **4932** |

</div>

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  TABLE OF CONTENTS                                                       -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d2b24,100:020e09&height=64&text=%F0%9F%93%8B%20%20Table%20of%20Contents&fontSize=22&fontColor=ccfbf1&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<div align="center">

| # | Section | # | Section |
|:---:|:---|:---:|:---|
| 01 | <a href="#overview">🛡️ Project Overview</a> | 07 | <a href="#rl-environment">🎮 RL Environment Design</a> |
| 02 | <a href="#pipeline">🔄 Three-Stage Pipeline</a> | 08 | <a href="#evaluation">📊 Evaluation and Results</a> |
| 03 | <a href="#ff-ae">⚡ Feedforward Autoencoder</a> | 09 | <a href="#dataset">📁 Dataset and Preprocessing</a> |
| 04 | <a href="#dae">🌀 Denoising Autoencoder</a> | 10 | <a href="#quickstart">🚀 Quick Start</a> |
| 05 | <a href="#conv-ae">🔷 Convolutional Autoencoder</a> | 11 | <a href="#future">🔮 Future Enhancements</a> |
| 06 | <a href="#ppo-agent">🤖 PPO Agent Training</a> | 12 | <a href="#references">📜 References</a> |

</div>

<br/><br/>

<a name="overview"></a>
<!----------------------------------------------------------------------------->
<!--  PROJECT OVERVIEW                                                        -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a2b24,100:020e09&height=64&text=%F0%9F%9B%A1%EF%B8%8F%20%20Project%20Overview&fontSize=22&fontColor=99f6e4&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

**NetGuard AI** is a hybrid **Network Intrusion Detection System (NIDS)** that fuses **unsupervised deep representation learning** with **adaptive reinforcement learning** to classify network traffic as *benign* or *malicious*. Rather than relying on hand-crafted features or static thresholds, the system autonomously discovers compact latent representations of raw network flows and trains a **PPO** policy-gradient agent to make detection decisions through **reward-driven optimization**.

<br/>

<div align="center">

| &nbsp; | What makes it different? | How it achieves it |
|:---:|:---|:---|
| 🧠 | **Deep Representation Learning** | Three autoencoder architectures compress raw features into a 32–64 dimensional latent space |
| 🎮 | **Adaptive Decision Boundary** | PPO RL agent learns to detect anomalies through structured reward signals — not static thresholds |
| ⚖️ | **Cost-Sensitive Reward Shaping** | Missed attacks penalized at `−4`, false alarms at `−1` — explicitly prioritizing security recall |
| 📐 | **Multi-Threshold Evaluation** | Final detection evaluated at thresholds `0.25`, `0.30`, and `0.35` for fine-grained sensitivity control |
| 📦 | **Large-Scale Validation** | Trained and tested on 2.8M+ real network flow records from the CICIDS-2017 benchmark |

</div>

<br/><br/>

<a name="pipeline"></a>
<!----------------------------------------------------------------------------->
<!--  THREE-STAGE PIPELINE                                                    -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d3329,100:030f0a&height=64&text=%F0%9F%94%84%20%20Three-Stage%20Detection%20Pipeline&fontSize=22&fontColor=6ee7b7&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The project implements a tightly integrated, end-to-end pipeline executed independently for each autoencoder. The notebook is organized in this exact order: **Pipeline 1: FF-AE → Pipeline 2: DAE → Pipeline 3: Conv-AE**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1 — Feature Extraction                           │
│                                                                           │
│   Raw Network Flow (76+ features after preprocessing)                    │
│              │                                                            │
│   ┌──────────┼──────────────────────────────────┐                        │
│   ▼          ▼                                  ▼                        │
│ [FF-AE]   [DAE]                             [Conv-AE]                    │
│ Dense     GaussianNoise(0.05)→Dense         Conv1D→MaxPool→              │
│ 128→64→32 128→64→64(latent)                 Flatten→Dense(32)            │
│                                                                           │
│   All produce: Latent Vector (32 or 64 dims)                              │
└──────────────────────────┬────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────────────┐
│                    STAGE 2 — Custom Gymnasium RL Environment              │
│                                                                           │
│   Observation : Latent vector (32 or 64 dims from frozen encoder)         │
│   Actions     : Discrete(2) → {0: Benign, 1: Malicious}                  │
│   Rewards     : +2 correct · −4 missed attack · −1 false alarm            │
│   Episode len : max_steps = 3000 samples per episode                     │
│                                                                           │
│              PPO Agent — MlpPolicy [128, 128] hidden layers               │
│              Total training: 120,000 timesteps                            │
│              EvalCallback every 4,000 steps → saved to ./best_model/     │
└──────────────────────────┬────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────────────┐
│                    STAGE 3 — Evaluation                                   │
│                                                                           │
│   Mode A: Argmax Policy (deterministic=True)                              │
│   Mode B: Custom Threshold — CategoricalDistribution probs[:,1] > thr    │
│   Thresholds: 0.25 / 0.30 / 0.35 (DAE and Conv-AE) · 0.30 only (FF-AE) │
│   Metrics: Precision · Recall · F1 · Confusion Matrix · PR-AUC           │
└───────────────────────────────────────────────────────────────────────────┘
```

<br/><br/>

<a name="ff-ae"></a>
<!----------------------------------------------------------------------------->
<!--  FEEDFORWARD AUTOENCODER                                                -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:103b31,100:030f0a&height=64&text=%E2%9A%A1%20%20Pipeline%201%20%E2%80%94%20Feedforward%20Autoencoder%20(FF-AE)&fontSize=20&fontColor=a7f3d0&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The **Feedforward Autoencoder** is the first and baseline pipeline. A dense encoder–decoder stack learns to compress and reconstruct high-dimensional network traffic features into a compact **32-dimensional** latent representation passed to the PPO agent.

<br/>

### 🏗️ Full Architecture

```python
input_dim  = X_train.shape[1]   # dynamic — features after preprocessing
latent_dim = 32

# ── ENCODER ──────────────────────────────────────────────────────────
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(input_layer)
encoded = Dropout(0.3)(encoded)
encoded = Dense(64,  activation='relu', kernel_regularizer=l2(1e-5))(encoded)
encoded = Dropout(0.3)(encoded)
encoded = Dense(32,  activation='relu')(encoded)          # Latent Space (32-dim)

# ── DECODER ──────────────────────────────────────────────────────────
decoded = Dense(64,        activation='relu')(encoded)
decoded = Dropout(0.3)(decoded)
decoded = Dense(128,       activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)  # linear — unbounded reconstruction

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-4, clipvalue=1.0), loss='mse')

# ── ENCODER SUBMODEL (passed to RL environment) ───────────────────────
encoder = Model(inputs=input_layer, outputs=encoded)
```

<br/>

<div align="center">

| Hyperparameter | Value |
|:---|:---:|
| **Latent Dimensionality** | **32** |
| **Encoder Hidden Layers** | 128 → 64 → 32 |
| **Decoder Hidden Layers** | 64 → 128 → `input_dim` |
| **Decoder Final Activation** | `linear` |
| **Dropout Rate** | `0.3` |
| **L2 Regularization** | `1e-5` (encoder Dense layers only) |
| **Optimizer** | Adam (`lr=1e-4`, `clipvalue=1.0`) |
| **Loss** | MSE |
| **Epochs** | 50 (EarlyStopping: patience=5, restore_best_weights=True) |
| **Batch Size** | **2048** |
| **Validation** | `validation_split=0.1` |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6) |

</div>

<br/>

### 🎯 PPO Configuration (FF-AE) — Balanced Reward

The FF-AE environment uses a **symmetric / balanced reward** — any misclassification (whether missing an attack or triggering a false alarm) costs the same `−3`:

```python
# ── Reward Logic (FF-AE) ─────────────────────────────────────────────
if action == true_label:
    reward = 2    # Correct classification — any class
else:
    reward = -3   # Any misclassification — symmetric penalty

# ── PPO Hyperparameters (FF-AE) ──────────────────────────────────────
model = PPO(
    "MlpPolicy", env,
    policy_kwargs   = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    learning_rate   = 1e-3,    # Higher LR than DAE / Conv-AE
    n_steps         = 1024,    # Larger rollout buffer
    batch_size      = 512,
    gamma           = 0.995,
    clip_range      = 0.2,
    ent_coef        = 0.04,    # Higher entropy coefficient
    # gae_lambda and vf_coef not set — SB3 defaults used
    verbose         = 1,
    tensorboard_log = "./ppo_logs/",
    device          = "auto"
)
model.learn(total_timesteps=120000, callback=eval_callback, progress_bar=True)
model.save("ppo_anomaly_detector")
```

> 📌 **Threshold Evaluation:** The FF-AE pipeline evaluates at a **single fixed threshold = 0.30 only** — it does not sweep multiple thresholds like DAE and Conv-AE.

<br/><br/>

<a name="dae"></a>
<!----------------------------------------------------------------------------->
<!--  DENOISING AUTOENCODER                                                  -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0e3d31,100:030f0a&height=64&text=%F0%9F%8C%80%20%20Pipeline%202%20%E2%80%94%20Denoising%20Autoencoder%20(DAE)&fontSize=20&fontColor=ccfbf1&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The **Denoising Autoencoder** is the most robust of the three architectures. It deliberately injects **Gaussian noise** (`stddev=0.05`) into the input via a `GaussianNoise` layer during training, forcing the encoder to learn noise-invariant latent representations that capture the *true* statistical structure of network traffic. This pipeline also introduces a **cost-sensitive reward** environment, uses **PyTorch tensors directly inside `_get_encoded_state`**, and expands the latent space to **64 dimensions**.

<br/>

### 🏗️ Full Architecture

```python
input_dim  = X_train.shape[1]
latent_dim = 64               # Larger than FF-AE (32)

# ── NOISE INJECTION (unique to DAE) ──────────────────────────────────
input_layer = Input(shape=(input_dim,))
noisy_input = GaussianNoise(0.05)(input_layer)   # stddev=0.05

# ── ENCODER (learns to denoise and compress) ──────────────────────────
encoded = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(noisy_input)
encoded = Dropout(0.4)(encoded)                  # Higher dropout than FF-AE
encoded = Dense(64,  activation='relu', kernel_regularizer=l2(1e-5))(encoded)
encoded = Dropout(0.4)(encoded)
encoded = Dense(64,  activation='relu')(encoded) # Latent Space (64-dim)

# ── DECODER (reconstructs clean signal from noisy encoding) ───────────
decoded = Dense(64,        activation='relu')(encoded)
decoded = Dropout(0.4)(decoded)
decoded = Dense(128,       activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # sigmoid — bounded output

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-4, clipvalue=1.0), loss='mse')

# ── ENCODER SUBMODEL ─────────────────────────────────────────────────
encoder = Model(inputs=input_layer, outputs=encoded)
```

<br/>

<div align="center">

| Hyperparameter | Value |
|:---|:---:|
| **Noise Layer** | `GaussianNoise(stddev=0.05)` |
| **Latent Dimensionality** | **64** (2× FF-AE) |
| **Encoder Hidden Layers** | 128 → 64 → 64 |
| **Decoder Final Activation** | **`sigmoid`** (bounded — key difference vs FF-AE's `linear`) |
| **Dropout Rate** | **`0.4`** (higher than FF-AE's `0.3`) |
| **L2 Regularization** | `1e-5` (encoder Dense layers) |
| **Optimizer** | Adam (`lr=1e-4`, `clipvalue=1.0`) |
| **Loss** | MSE |
| **Epochs** | **80** (extended from FF-AE's 50) |
| **Batch Size** | **4096** (double FF-AE's 2048) |
| **Validation** | `validation_split=0.1` |
| **Val Loss Converged** | **~0.52** ✅ Best reconstruction accuracy across all three |

</div>

<br/>

### 🔬 Unique: PyTorch Tensor Inference Inside RL Environment

The DAE pipeline's `_get_encoded_state` is the only one in the notebook that explicitly uses **PyTorch tensors** for encoder inference — bridging TensorFlow-trained weights via SB3's PyTorch runtime:

```python
def _get_encoded_state(self, idx):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.tensor(self.X[idx:idx+1], dtype=torch.float32).to(device)
    with torch.no_grad():
        enc_out = self.encoder(x_tensor).cpu().numpy()[0]
    return enc_out
```

<br/>

### 🎯 PPO Configuration (DAE) — Cost-Sensitive Reward

The DAE environment is **explicitly labeled "Cost-Sensitive"** in the notebook. Missed attacks are penalized 4× harder than false alarms:

```python
# ── Cost-Sensitive Reward Logic (DAE) ────────────────────────────────
if action == true_label:
    reward = 2           # Correct classification — any class
else:
    reward = -4 if true_label == 1 else -1
    # -4: missed attack (false negative) — security-critical
    # -1: false alarm   (false positive) — minor operational cost

# ── PPO Hyperparameters (DAE) ─────────────────────────────────────────
model = PPO(
    "MlpPolicy", env,
    policy_kwargs   = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    learning_rate   = 5e-4,    # Lower than FF-AE's 1e-3
    n_steps         = 512,     # Shorter rollout than FF-AE's 1024
    batch_size      = 256,     # Smaller than FF-AE's 512
    gamma           = 0.995,
    gae_lambda      = 0.95,    # Added — not present in FF-AE
    vf_coef         = 1.0,     # Added — not present in FF-AE
    clip_range      = 0.2,
    ent_coef        = 0.02,    # Lower than FF-AE's 0.04
    verbose         = 1,
    tensorboard_log = "./ppo_logs/",
    device          = "auto"
)
model.learn(total_timesteps=120000, callback=eval_callback, progress_bar=True)
model.save("ppo_anomaly_detector")
```

> 🏆 **Best Autoencoder:** DAE achieves the **lowest val_loss (~0.52)** across all three architectures, producing the highest-quality latent features for the RL agent.

<br/><br/>

<a name="conv-ae"></a>
<!----------------------------------------------------------------------------->
<!--  CONVOLUTIONAL AUTOENCODER                                              -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:133d2f,100:040f0a&height=64&text=%F0%9F%94%B7%20%20Pipeline%203%20%E2%80%94%20Convolutional%20Autoencoder%20(Conv-AE)&fontSize=20&fontColor=6ee7b7&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

The **Convolutional Autoencoder** treats each network flow record as a **1D signal** by reshaping inputs from `(N, features)` to `(N, features, 1)`, enabling `Conv1D` filters to detect local spatial patterns across feature dimensions. This is also the **only pipeline with a three-way train/val/test split** — using an explicit `validation_data` tuple instead of `validation_split`.

<br/>

### 🏗️ Unique: Three-Way Data Split

```python
# Conv-AE uses a different data split strategy than FF-AE and DAE
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# Further split X_temp into validation and test
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
)

# Reshape for Conv1D: (N, features) → (N, features, 1)
X_train_conv = X_train.reshape(-1, X_train.shape[1], 1)
X_val_conv   = X_val.reshape(-1,   X_val.shape[1],   1)
X_test_conv  = X_temp.reshape(-1,  X_train.shape[1],  1)   # X_temp used as test set
```

<br/>

### 🏗️ Full Architecture

```python
input_shape = (X_train.shape[1], 1)   # e.g., (76, 1) — treated as 1D signal
latent_dim  = 32

inputs = Input(shape=input_shape)

# ── ENCODER (1D Convolutional) ────────────────────────────────────────
x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
shape_before_flatten = tf.keras.backend.int_shape(x)   # captured for decoder reshape
x      = Flatten()(x)
latent = Dense(32, activation='relu')(x)               # Latent Space (32-dim)

# ── DECODER (Upsampling + Deconvolution) ─────────────────────────────
x = Dense(np.prod(shape_before_flatten[1:]), activation='relu')(latent)
x = Reshape(shape_before_flatten[1:])(x)
x = UpSampling1D(2)(x)
x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(x)

conv_autoencoder = Model(inputs, decoded)
conv_autoencoder.compile(optimizer=Adam(learning_rate=1e-4, clipvalue=1.0), loss='mse')

# ── ENCODER SUBMODEL (called encoder_output in notebook) ─────────────
encoder_output = Model(inputs, latent)
```

<br/>

<div align="center">

| Hyperparameter | Value |
|:---|:---:|
| **Input Shape** | `(features, 1)` — 1D signal |
| **Conv Filters** | 32 → 16 (encoder) · 16 → 1 (decoder) |
| **Kernel Size** | `3` with `padding='same'` |
| **Pooling / Upsampling** | `MaxPooling1D(2)` / `UpSampling1D(2)` |
| **Latent Dimensionality** | **32** |
| **Decoder Final Activation** | `sigmoid` |
| **No Dropout** | Conv-AE encoder has no dropout layers |
| **Optimizer** | Adam (`lr=1e-4`, `clipvalue=1.0`) |
| **Loss** | MSE |
| **Epochs** | 50 (EarlyStopping: patience=5) |
| **Batch Size** | **4096** |
| **Validation** | Explicit `validation_data=(X_val_conv, X_val_conv)` |
| **Val Loss Converged** | **~0.67** |

</div>

<br/>

### 🎯 PPO Configuration (Conv-AE)

Conv-AE shares the same **asymmetric cost-sensitive reward** and refined PPO config as DAE. The key uniqueness in `_get_encoded_state` is the per-sample reshape before encoder inference:

```python
def _get_encoded_state(self, idx):
    # Conv-AE must reshape each sample to (features, 1) for Conv1D encoder
    sample = self.X[idx:idx+1].reshape(-1, self.X.shape[1], 1)
    return self.encoder(sample).numpy()[0]   # encoder = encoder_output submodel
```

```python
model = PPO(
    "MlpPolicy", env,
    policy_kwargs   = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    learning_rate   = 5e-4,
    n_steps         = 512,
    batch_size      = 256,
    gamma           = 0.995,
    gae_lambda      = 0.95,
    vf_coef         = 1.0,
    clip_range      = 0.2,
    ent_coef        = 0.02,
    verbose         = 1,
    tensorboard_log = "./ppo_logs/",
    device          = "auto"
)
model.learn(total_timesteps=120000, callback=eval_callback, progress_bar=True)
model.save("ppo_anomaly_detector")
```

<br/>

### 📊 Full Architecture Comparison — All Three Pipelines

<div align="center">

| Feature | FF-AE | DAE | Conv-AE |
|:---|:---:|:---:|:---:|
| **Architecture Type** | Feedforward Dense | Dense + Noise | 1D Convolutional |
| **Latent Dim** | **32** | **64** | **32** |
| **Noise Injection** | ❌ | ✅ `GaussianNoise(0.05)` | ❌ |
| **Dropout (Encoder)** | `0.3` | `0.4` | ❌ None |
| **Decoder Activation** | `linear` | `sigmoid` | `sigmoid` |
| **Batch Size** | 2048 | 4096 | 4096 |
| **Max Epochs** | 50 | **80** | 50 |
| **Validation Method** | `validation_split=0.1` | `validation_split=0.1` | **explicit `validation_data`** |
| **Data Split** | 80/20 | 80/20 | **80/10/10 (3-way)** |
| **Val Loss** | — | **~0.52 ⭐** | ~0.67 |
| **Reward Strategy** | Symmetric (`+2/−3`) | Asymmetric (`+2/−4/−1`) | Asymmetric (`+2/−4/−1`) |
| **PPO `learning_rate`** | `1e-3` | `5e-4` | `5e-4` |
| **PPO `n_steps`** | `1024` | `512` | `512` |
| **PPO `batch_size`** | `512` | `256` | `256` |
| **PPO `ent_coef`** | `0.04` | `0.02` | `0.02` |
| **`gae_lambda`** | SB3 default | ✅ `0.95` | ✅ `0.95` |
| **`vf_coef`** | SB3 default | ✅ `1.0` | ✅ `1.0` |
| **Threshold Sweep** | Fixed `0.30` only | `[0.25, 0.30, 0.35]` | `[0.25, 0.30, 0.35]` |

</div>

<br/><br/>

<a name="ppo-agent"></a>
<!----------------------------------------------------------------------------->
<!--  PPO AGENT                                                              -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f3a2e,100:030f0a&height=64&text=%F0%9F%A4%96%20%20PPO%20Agent%20%E2%80%94%20Policy%20Training%20Deep%20Dive&fontSize=22&fontColor=a7f3d0&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

Each autoencoder pipeline feeds its frozen encoder into a **custom Gymnasium environment** where a **Proximal Policy Optimization (PPO)** agent learns to classify network flows through repeated interaction.

<br/>

### 🏗️ Shared Policy Network (All Pipelines)

```
Latent Vector (32-dim or 64-dim)
         │
         ▼
    Dense(128, relu)   ──── Shared net_arch trunk ────
         │
    Dense(128, relu)
         │
    ┌────┴────┐
    ▼         ▼
 Policy     Value
  Head       Head
(π: action  (V: scalar
 logits)     estimate)

net_arch = {"pi": [128, 128], "vf": [128, 128]}
```

<br/>

### 📈 Training Behaviour (Common Across All Pipelines)

<div align="center">

| Training Signal | Observed Trend |
|:---|:---|
| **KL Divergence** | Stabilizes after initial policy updates |
| **Value Loss** | Decreases steadily — improving state-value estimation |
| **Policy Entropy** | Decreases over time → agent becomes more **confident** |
| **Explained Variance** | Improves across training iterations |
| **Training Speed** | ~220 fps during training loops |
| **TensorBoard Logging** | `./ppo_logs/` — metrics logged for all runs |
| **Best Model Checkpointing** | `./best_model/` via `EvalCallback(eval_freq=4000)` |

</div>

<br/>

### 🏆 PPO Mean Reward — Final Results

> Evaluated via `evaluate_policy(model, test_env, n_eval_episodes=10, deterministic=True)`

<div align="center">

| Pipeline | Mean Reward | Reward Design |
|:---:|:---:|:---|
| FF-AE + PPO | **4415** | Balanced symmetric `+2/−3` |
| DAE + PPO | **4839** | Cost-sensitive `+2/−4/−1` |
| Conv-AE + PPO | **4932** ⭐ | Cost-sensitive `+2/−4/−1` |

</div>

<br/><br/>

<a name="rl-environment"></a>
<!----------------------------------------------------------------------------->
<!--  RL ENVIRONMENT                                                         -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:123829,100:030f09&height=64&text=%F0%9F%8E%AE%20%20Custom%20RL%20Environment%20%E2%80%94%20AnomalyDetectionEnv&fontSize=20&fontColor=99f6e4&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

A custom `gym.Env` named `AnomalyDetectionEnv` wraps the frozen autoencoder encoder and the labeled dataset, presenting each network flow as an encoded **observation** and rewarding correct binary classification decisions.

<br/>

### 🔧 Full Environment Code (DAE variant — most complete form)

```python
class AnomalyDetectionEnv(gym.Env):
    """
    Cost-Sensitive reward shaping:
      +2  correct classification (benign or malicious)
      -1  false positive  (benign predicted as malicious)
      -4  missed malicious (malicious predicted as benign) — security-critical!
    """
    def __init__(self, X, y, encoder, latent_dim, max_steps=3000):
        super().__init__()
        self.X          = X.astype(np.float32)
        self.y          = y
        self.encoder    = encoder
        self.latent_dim = latent_dim
        self.max_steps  = max_steps
        self.current_idx  = 0
        self.step_count   = 0

        self.action_space      = spaces.Discrete(2)    # 0=benign, 1=malicious
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        self.step_count  = 0
        return self._get_encoded_state(0), {}

    def step(self, action):
        true_label = self.y[self.current_idx]
        if action == true_label:
            reward = 2
        else:
            reward = -4 if true_label == 1 else -1   # asymmetric cost-sensitive!
        self.current_idx = (self.current_idx + 1) % len(self.X)
        self.step_count += 1
        done     = self.step_count >= self.max_steps
        next_obs = (
            self._get_encoded_state(self.current_idx)
            if not done else np.zeros(self.latent_dim)
        )
        return next_obs, reward, done, False, {}

    def _get_encoded_state(self, idx):
        # DAE uses explicit PyTorch tensors for inference
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = torch.tensor(self.X[idx:idx+1], dtype=torch.float32).to(device)
        with torch.no_grad():
            enc_out = self.encoder(x_tensor).cpu().numpy()[0]
        return enc_out
```

<br/>

### 🔄 Vectorized Environment Wrapping (All Pipelines)

```python
env = AnomalyDetectionEnv(X_train, y_train, encoder, latent_dim, max_steps=3000)
env = Monitor(env)                          # Stable-Baselines3 Monitor wrapper
env = make_vec_env(lambda: env, n_envs=1)  # Vectorized (single process)
```

<br/>

### ⚖️ Reward Comparison — Full Table

<div align="center">

| Scenario | True Label | Action | FF-AE | DAE / Conv-AE |
|:---|:---:|:---:|:---:|:---:|
| Correct benign | 0 | 0 | **+2** | **+2** |
| Correct attack | 1 | 1 | **+2** | **+2** |
| **Missed attack (FN)** | **1** | **0** | **−3** | **−4** ⚠️ |
| False alarm (FP) | 0 | 1 | **−3** | **−1** |

</div>

> 🔐 The asymmetric `−4 / −1` structure in DAE and Conv-AE encodes a **security-first philosophy** — undetected attacks cause exponentially more damage than false alarms in operational network environments.

<br/><br/>

<a name="evaluation"></a>
<!----------------------------------------------------------------------------->
<!--  EVALUATION & RESULTS                                                   -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d3527,100:030f09&height=64&text=%F0%9F%93%8A%20%20Evaluation%20and%20Results&fontSize=22&fontColor=a7f3d0&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

All pipelines are evaluated in two modes: **deterministic Argmax Policy** and **probabilistic Custom Threshold**.

<br/>

### 🔍 Evaluation Code

```python
# ── MODE A: Argmax (Deterministic Policy) ────────────────────────────
y_pred_argmax = []
obs, _ = test_env.reset()
for _ in range(len(X_test)):
    action, _ = model.predict(obs, deterministic=True)
    y_pred_argmax.append(action)
    obs, _, done, _, _ = test_env.step(action)
    if done:
        break

print(classification_report(y_test[:len(y_pred_argmax)], y_pred_argmax))

# Confusion Matrix
cm = confusion_matrix(y_test[:len(y_pred_argmax)], y_pred_argmax)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test[:len(y_pred_argmax)], y_pred_argmax)
pr_auc = auc(recall, precision)

# ── MODE B: Custom Threshold (Probabilistic) ─────────────────────────
def predict_with_threshold(model, obs, threshold=0.3):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs   # CategoricalDistribution — shape [batch, 2]
    predicted_actions = (probs[:, 1] > threshold).long().cpu().numpy()
    return predicted_actions

# DAE and Conv-AE sweep all three thresholds in a loop:
for threshold in [0.25, 0.30, 0.35]:
    ...
```

<br/>

### 📐 Threshold Sweep Results (DAE and Conv-AE)

> FF-AE uses `threshold=0.30` only. DAE and Conv-AE run the full `[0.25, 0.30, 0.35]` loop.

<div align="center">

#### Threshold = 0.25

| Metric | Score |
|:---:|:---:|
| **Precision** | ~0.91 |
| **Recall** | ~0.92 |
| **F1-Score** | ~0.92 |

*More liberal — higher recall, marginally more false alarms.*

<br/>

#### Threshold = 0.30 — Optimal Balance ⭐

| Metric | Score |
|:---:|:---:|
| **Precision** | ~0.92 |
| **Recall** | ~0.92 |
| **F1-Score** | ~0.92 |

*Best overall balance — recommended default for deployment.*

<br/>

#### Threshold = 0.35

| Metric | Score |
|:---:|:---:|
| **Precision** | ~0.92 |
| **Recall** | ~0.89 |
| **F1-Score** | ~0.90 |

*More conservative — fewer false alarms, slightly more missed attacks.*

</div>

<br/>

### 🔬 Key Findings

<div align="center">

| Observation | Detail |
|:---|:---|
| 🎯 **Optimal Threshold** | `0.30` delivers the best Precision-Recall balance |
| 🔴 **False Positives** | Remain extremely low across all threshold values |
| 🏆 **RL vs. Reconstruction** | PPO significantly outperforms raw autoencoder reconstruction-error baselines |
| 🧩 **Latent Quality Matters** | Lower val_loss (DAE: ~0.52) directly improves RL stability and final classification quality |
| 📈 **PR-AUC** | Computed via `auc(recall, precision)` for both Argmax and Threshold modes in all pipelines |
| 📊 **Confusion Matrices** | Plotted via `sns.heatmap(cmap='Blues')` for every evaluation run |
| 🔄 **Reward Progression** | 4415 (FF-AE) → 4839 (DAE) → 4932 (Conv-AE) |

</div>

<br/><br/>

<a name="dataset"></a>
<!----------------------------------------------------------------------------->
<!--  DATASET & PREPROCESSING                                                -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0e3328,100:030f09&height=64&text=%F0%9F%93%81%20%20Dataset%20and%20Preprocessing%20%E2%80%94%20CICIDS-2017&fontSize=22&fontColor=6ee7b7&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<div align="center">

| Property | Detail |
|:---|:---|
| 📊 **Total Records** | **2.8M+ network flow records** |
| 📁 **File Format** | Multiple CSV files loaded via `glob.glob("Dataset/*.csv")` then `pd.concat` |
| 🔢 **Original Features** | ~80 flow-level features |
| ✂️ **Features After Preprocessing** | ~76 numerical features |
| 🏷️ **Label Encoding** | `0 = Benign`, `1 = Malicious` (all other labels) |
| ⚖️ **Class Imbalance** | Majority benign — addressed via stratified splits and asymmetric reward shaping |
| 🌐 **Attack Categories** | DDoS, DoS (Hulk, GoldenEye, Slowloris), Brute Force (FTP/SSH), Web Attacks (XSS, SQL Injection, Clickjacking), Botnet ARES, Port Scan, Infiltration |

</div>

<br/>

### 🔧 Full Preprocessing Pipeline (Identical Across All Three Notebooks)

```python
# Step 1: Load all CSV files with glob
all_files = glob.glob("C:\\Users\\dell\\Downloads\\Dataset\\*.csv")
df = pd.concat([pd.read_csv(f) for f in all_files], axis=0, ignore_index=True)

# Step 2: Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Step 3: Binary label encoding
df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)

# Step 4: Print label distribution for inspection
print(df['label'].value_counts())

# Step 5: Drop non-useful columns
cols_to_drop = ['destination port', 'flow duration']
df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

# Step 6: Handle infinities and NaNs
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(axis=1, thresh=int(0.7 * len(df)))   # drop columns with >30% missing
df = df.fillna(df.median())                           # fill remainder with column median

# Step 7: IQR-based outlier detection (diagnostic)
# Q1 / Q3 / IQR computed per numeric column
# Outlier bounds: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# Total count printed; may include duplicates across columns

# Step 8: Feature matrix + StandardScaler
X        = df.drop('label', axis=1).values
y        = df['label'].values
X_scaled = StandardScaler().fit_transform(X)

# Step 9: Stratified train/test split (FF-AE and DAE)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 10: NaN/Inf safety clipping on training data
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e5, neginf=-1e5)

# Step 11 (Conv-AE only): Additional val split + 1D reshape
# X_train, X_val split from X_temp; then:
# X_train_conv = X_train.reshape(-1, X_train.shape[1], 1)
```

<br/>

### 🖥️ GPU and Hardware Setup

```python
# TensorFlow GPU detection and memory growth configuration
print("Num GPUs Available (TensorFlow):", len(tf.config.list_physical_devices('GPU')))
print("PyTorch CUDA available:", torch.cuda.is_available())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

> Both **TensorFlow** (for autoencoder training via Keras) and **PyTorch** (for PPO via SB3 and DAE encoder inference) are active simultaneously. `device="auto"` in PPO selects GPU when available.

<br/><br/>

<a name="quickstart"></a>
<!----------------------------------------------------------------------------->
<!--  QUICK START                                                            -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0c3226,100:020e09&height=64&text=%F0%9F%9A%80%20%20Quick%20Start&fontSize=22&fontColor=ccfbf1&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

### Prerequisites

<div align="center">

| Requirement | Version / Detail |
|:---|:---:|
| 🐍 **Python** | 3.9+ |
| 🖥️ **GPU** | Optional — CUDA-compatible for faster AE training and PPO |
| 💾 **RAM** | 16 GB+ recommended (2.8M record dataset) |
| 📦 **Frameworks** | TensorFlow + PyTorch (both required simultaneously) |

</div>

<br/>

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders
cd Network-Anomaly-Detection-using-RL-model-and-Autoencoders

# 2. Install all dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn \
            shimmy>=2.0 "stable-baselines3[extra]" gymnasium torch
```

<br/>

### Dataset Setup

```bash
# Create Dataset directory and place all CICIDS-2017 CSV files inside
mkdir Dataset
# Dataset/Monday-WorkingHours.pcap_ISCX.csv
# Dataset/Tuesday-WorkingHours.pcap_ISCX.csv
# Dataset/Wednesday-workingHours.pcap_ISCX.csv
# ... (remaining daily capture files)

# Update glob path in the notebook if needed:
# all_files = glob.glob("Dataset/*.csv")
```

<br/>

### Run the Notebook

```bash
jupyter notebook Network_Anomaly_Detection.ipynb
```

<div align="center">

| Section | Notebook Header | Autoencoder | Reward | Thresholds |
|:---:|:---|:---:|:---:|:---:|
| **1** | RL model PPO with Simple FeedForward AutoEncoder | FF-AE | Symmetric | `0.30` only |
| **2** | RL model PPO with Denoising AutoEncoder | DAE | Asymmetric | `[0.25, 0.30, 0.35]` |
| **3** | RL model PPO with Simple Convolutional Autoencoder | Conv-AE | Asymmetric | `[0.25, 0.30, 0.35]` |

</div>

<br/>

### 📦 Full Dependency List

```
tensorflow              # Autoencoder training (Keras API)
torch                   # SB3 backend + DAE encoder inference
numpy
pandas
scikit-learn            # StandardScaler, train_test_split, classification metrics
matplotlib
seaborn                 # Confusion matrix heatmaps (cmap='Blues')
gymnasium               # RL environment base class
shimmy>=2.0             # Gymnasium compatibility shim for Stable-Baselines3
stable-baselines3[extra]  # PPO, Monitor, make_vec_env, EvalCallback, evaluate_policy
glob                    # CSV loading (Python stdlib)
```

<br/><br/>

<a name="future"></a>
<!----------------------------------------------------------------------------->
<!--  FUTURE ENHANCEMENTS                                                    -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d3225,100:030f09&height=64&text=%F0%9F%94%AE%20%20Future%20Enhancements&fontSize=22&fontColor=99f6e4&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<details>
<summary><b>🧠 Advanced Autoencoder Architectures</b></summary>

<br/>

- [ ] **Variational Autoencoders (VAE)** — probabilistic latent space with uncertainty quantification
- [ ] **Transformer-based encoders** — multi-head self-attention over flow feature sequences
- [ ] **Graph Neural Networks (GNN)** — model inter-flow correlations as graph edges
- [ ] **Sparse Autoencoders** — enforce sparsity for interpretable latent features
- [ ] **LSTM / GRU encoders** — temporal sequence modelling across flow sessions

<br/>
</details>

<details>
<summary><b>🤖 Reinforcement Learning Improvements</b></summary>

<br/>

- [ ] **Multi-agent RL** — cooperative agents handling different traffic slices simultaneously
- [ ] **Hierarchical RL** — macro-agent selects strategy, micro-agent classifies individual flows
- [ ] **Multi-class action space** — extend `Discrete(2)` to `Discrete(N)` for per-attack-type classification
- [ ] **SAC / TD3** — continuous-action alternatives to PPO for finer probability control
- [ ] **Prioritized Experience Replay** — oversample rare attack types (Infiltration, Botnet) during training

<br/>
</details>

<details>
<summary><b>🚀 Real-Time Deployment</b></summary>

<br/>

- [ ] **Suricata / Zeek integration** — consume live PCAP streams for real-time classification
- [ ] **Online learning mode** — continuously update autoencoder and PPO weights on streaming traffic
- [ ] **Model quantization / TFLite** — reduce inference latency for edge deployment
- [ ] **REST API / gRPC endpoint** — serve the detection pipeline as a containerized microservice
- [ ] **TensorBoard live dashboards** — expose `./ppo_logs/` metrics for production monitoring

<br/>
</details>

<br/><br/>

<a name="references"></a>
<!----------------------------------------------------------------------------->
<!--  REFERENCES                                                             -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0c2f23,100:020e09&height=64&text=%F0%9F%93%9C%20%20References&fontSize=22&fontColor=6ee7b7&fontAlignY=52&fontAlign=50" width="100%"/>

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
  journal   = {arXiv preprint arXiv:1707.06347},
  url       = {https://arxiv.org/abs/1707.06347}
}

@software{stable_baselines3,
  author    = {Raffin, Antonin and Hill, Ashley and Gleave, Adam and Kanervisto, Anssi and Ernestus, Maximilian and Dormann, Noah},
  title     = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  year      = {2021},
  journal   = {Journal of Machine Learning Research},
  volume    = {22},
  pages     = {1--8},
  url       = {https://jmlr.org/papers/v22/20-1364.html}
}

@misc{gymnasium2022,
  author    = {Farama Foundation},
  title     = {Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  year      = {2022},
  url       = {https://gymnasium.farama.org}
}
```

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  ACKNOWLEDGMENTS                                                        -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0b2d22,100:020e09&height=64&text=%F0%9F%99%8F%20%20Acknowledgments&fontSize=22&fontColor=a7f3d0&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<div align="center">

| &nbsp; | Acknowledgment |
|:---:|:---|
| 🎓 | **Canadian Institute for Cybersecurity** — For releasing the CICIDS-2017 benchmark dataset |
| 🤝 | **Stable-Baselines3 Team** — For production-grade PPO, EvalCallback and evaluation utilities |
| 🧠 | **TensorFlow / Keras Team** — For the deep learning framework powering all three autoencoders |
| 🎮 | **Farama Foundation** — For the Gymnasium RL environment standard and shimmy compatibility layer |
| 🔥 | **PyTorch Community** — For the tensor infrastructure underlying SB3 and DAE encoder inference |
| 🌟 | **Open Source ML Community** — For the ecosystem that makes this research reproducible |

</div>

<br/><br/>

<!----------------------------------------------------------------------------->
<!--  CONTACT                                                                -->
<!----------------------------------------------------------------------------->

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d2f23,100:030f09&height=64&text=%F0%9F%93%9E%20%20Contact%20and%20Support&fontSize=22&fontColor=ccfbf1&fontAlignY=52&fontAlign=50" width="100%"/>

<br/>

<div align="center">

**Questions? Issues? Contributions?**

<br/>

[![GitHub Issues](https://img.shields.io/badge/Issues-Report%20Bug-ef4444?style=for-the-badge&logo=github)](https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders/issues/new)&nbsp;
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

[![Star this repo](https://img.shields.io/github/stars/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders?style=for-the-badge&logo=github&color=2dd4bf&labelColor=0d1117&label=Star%20this%20repo)](https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders/stargazers)

<br/>

*© 2025 Kumar Piyush Raj &nbsp;·&nbsp; [GitHub @kumarpiyushraj](https://github.com/kumarpiyushraj)*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2dd4bf,40:0e9e7e,70:053428,100:020f14&height=160&section=footer" width="100%"/>

</div>
