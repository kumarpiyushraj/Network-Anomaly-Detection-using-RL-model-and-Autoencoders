# 🛡️ Network Anomaly Detection using Autoencoders & PPO Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-PPO-green)](https://stable-baselines3.readthedocs.io/)
[![Dataset: CICIDS2017](https://img.shields.io/badge/Dataset-CIC--IDS2017-critical)](https://www.unb.ca/cic/datasets/ids-2017.html)

A hybrid **Network Intrusion Detection System (NIDS)** combining **unsupervised feature learning** through autoencoders with **adaptive decision-making** using the **PPO reinforcement learning algorithm**. The pipeline builds latent representations of network flows, embeds them into a custom RL environment, and trains an agent to detect intrusions through reward-driven optimization.

📘 Live Notebook:
[`Network_Anomaly_Detection.ipynb`](Network_Anomaly_Detection.ipynb)

---

# 🎯 Project Overview

This project implements a **three-stage anomaly detection pipeline**:

### **1️⃣ Feature Extraction (Three Autoencoders)**

The notebook trains three autoencoder architectures:

#### **✔ Feedforward Autoencoder (FF-AE)**

* Dense layers (128 → 64 → 32 latent)
* Learns compressed representation of 76 network features
* Training converged with decreasing loss; stable reconstruction behaviour

#### **✔ Convolutional Autoencoder (Conv-AE)**

* 1D Conv layers with MaxPooling
* Produces **32-dimensional latent vectors**
* Final validation loss converged to **~0.67**
* Strongest reconstruction performance among the three models

#### **✔ Deep Hybrid Autoencoder (3rd AE block)**

* Multiple dense layers + dropout
* Captures non-linear traffic patterns
* Validation loss stabilized at **~0.52**, showing best reconstruction accuracy

These latent representations are used as input features for the PPO agent.

---

### **2️⃣ Reinforcement Learning with PPO**

A custom **Gymnasium-compatible anomaly detection environment** is built:

* **Observation:** 32-dimensional latent vector from autoencoder
* **Actions:**

  * `0 = Benign`
  * `1 = Malicious`
* **Rewards:**

  * `+2` correct detection
  * `−4` missed attack
  * `−1` false alarm
* PPO policy: MLP with hidden layers [128, 128]
* Trained over **3000+ timesteps per iteration**
* Repeated training runs show stable policy convergence

---

### **3️⃣ Evaluation & Metrics**

The notebook evaluates PPO using deterministic actions and probability thresholds.
Below are the **actual extracted outputs from your notebook**:

### **📌 PPO Mean Reward**

Across experiments:

* **4415**
* **4839**
* **4932**

These high mean rewards indicate strong agent performance.

---

# 📊 Classification Results

The notebook evaluates at multiple probability thresholds:

### **Threshold = 0.25**

| Metric    | Score |
| --------- | ----- |
| Precision | ~0.91 |
| Recall    | ~0.92 |
| F1-Score  | ~0.92 |

---

### **Threshold = 0.30**

| Metric    | Score |
| --------- | ----- |
| Precision | ~0.92 |
| Recall    | ~0.92 |
| F1-Score  | ~0.92 |

---

### **Threshold = 0.35**

| Metric    | Score |
| --------- | ----- |
| Precision | ~0.92 |
| Recall    | ~0.89 |
| F1-Score  | ~0.90 |

---

### **General Observations**

* Best balance achieved around **threshold = 0.30**
* False positives remain extremely low
* PPO significantly outperforms raw autoencoder reconstruction thresholds
* Latent-space feature learning drastically improves RL stability and classification quality

---

# 📁 Dataset Summary

**Dataset:** CICIDS-2017

* 2.8M+ network flow records
* 76 cleaned numerical features
* Multiple attack categories (DDoS, DoS, Brute Force, Web Attacks, Botnet, etc.)
* Highly imbalanced (majority benign)

Preprocessing applied:

* Removal of non-numerical fields
* Min-max scaling
* Train/test split
* Latent encoding using autoencoders

---

# 🧠 Model Architecture

### **Autoencoders**

```
Input (76)
   ↓ Dense / Convolutional layers
   ↓ Latent representation (32)
   ↑ Decoder layers
Output (76 reconstructed)
```

### **PPO Policy**

```
Latent Vector (32)
   ↓ Dense(128)
   ↓ Dense(128)
   ↓ Policy & Value Heads
```

---

# 📈 Training Behavior

### **Autoencoder Training**

* Convergence visible over epochs
* **Conv-AE:** val_loss → ~0.67
* **Deep AE:** val_loss → ~0.52 (best)
* Reconstructions become smoother and more stable

### **PPO Training**

* KL divergence and value loss stabilize
* Entropy decreases steadily → policy confidence increases
* fps ~220 during training loops
* Explained variance improves over iterations

---

# 🚀 Quick Start

```bash
git clone https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders
cd Network-Anomaly-Detection-using-RL-model-and-Autoencoders

pip install -r requirements.txt
```

### Run the notebook:

```bash
jupyter notebook Network_Anomaly_Detection.ipynb
```

Place CICIDS2017 CSV files inside `Dataset/`.

---

# 📦 Requirements

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
gymnasium
stable-baselines3[extra]
seaborn
```

(Tested with Python 3.9+)

---

# 🔮 Future Enhancements

* Variational Autoencoders (VAE) for uncertainty modelling
* Graph Neural Network–based flow correlation
* Multi-agent RL approaches
* Deployment on real-time log engines (Suricata / Zeek)
* Online learning mode for streaming traffic

---

# 📜 References

```bibtex
@dataset{cic_ids2017,
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A},
  title={Intrusion Detection Evaluation Dataset (CICIDS2017)},
  year={2017},
  url={https://www.unb.ca/cic/datasets/ids-2017.html}
}
```

---

# ⭐ Acknowledgement

Made with ❤️ by **[kumarpiyushraj](https://github.com/kumarpiyushraj)**.
If this repository helps your research, please consider starring the project!

---
<p align="center">
  <sub>© 2024 Kumar Piyush Raj. All rights reserved.</sub>
</p>
