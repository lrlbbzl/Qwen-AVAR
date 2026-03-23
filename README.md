<div align="center">

# From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning

<p align="center">
  <strong>[ICLR 2026 Poster]</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.03825"><img src="https://img.shields.io/badge/arXiv-2603.03825-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/Antimage01/AVAR-Thinker-7B"><img src="https://img.shields.io/badge/🤗%20Model-AVAR--Thinker--7B-yellow" alt="Model"></a>
  <a href="https://huggingface.co/datasets/Antimage01/AVAR-ColdStart-30K"><img src="https://img.shields.io/badge/🤗%20Dataset-AVAR--ColdStart--30K-blue" alt="Dataset"></a>
  <a href="https://github.com/lrlbbzl/Qwen-AVAR/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.03825">📄 Paper</a> •
  <a href="https://huggingface.co/Antimage01/AVAR-Thinker-7B">🤗 Model</a> •
  <a href="https://huggingface.co/datasets/Antimage01/AVAR-ColdStart-30K">📊 Dataset</a> •
  <a href="https://github.com/lrlbbzl/Qwen-AVAR">💻 Code</a>
</p>

</div>

---

## 📰 News

- **[2026/03]** 🎉 Paper accepted at **ICLR 2026** as a **Poster**!
- **[2026/03]** 🚀 We release the code, model [AVAR-Thinker-7B](https://huggingface.co/Antimage01/AVAR-Thinker-7B), and dataset [AVAR-ColdStart-30K](https://huggingface.co/datasets/Antimage01/AVAR-ColdStart-30K).

---

## 📖 Overview

Multimodal large reasoning models critically depend on cold-start initialization, yet the underlying mechanisms remain poorly understood. We introduce **Visual Attention Score (VAS)**, an attention-based metric that quantifies a model's focus on visual tokens, and reveal:

- 📈 **Strong Correlation**: Reasoning performance is strongly correlated with VAS (r=0.9616).
- 🔍 **Lazy Attention Anchoring**: A counter-intuitive finding — multimodal cold-start fails to effectively boost VAS, while text-only cold-start leads to significant VAS increases.
- ✅ **Causal Validation**: Training-free inference interventions that directly modulate attention allocation achieve 1%–2% performance gains, confirming the causal role of this phenomenon.

Based on these insights, we propose **AVAR** (**A**ttention-guided **V**isual **A**nchoring and **R**eflection), a framework integrating:

1. **Visual Anchoring Data Synthesis** — Curating cold-start data that explicitly anchors visual reasoning.
2. **Attention-Guided Objective** — Training objectives that guide the model to attend more to visual tokens.
3. **Visual Anchoring Reward Shaping** — Reward mechanisms that encourage visually-grounded reasoning during RL training.

Applying AVAR to **Qwen2.5-VL-7B** achieves an average **7.0% gain** across 7 multimodal reasoning benchmarks.

---

## 🏗️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/lrlbbzl/Qwen-AVAR.git
cd Qwen-AVAR

# 2. Create conda environment
conda create -n avar python=3.11
conda activate avar

# 3. Install dependencies (based on EasyR1)
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quick Start

### Cold-Start Training

```bash
# Run cold-start SFT with AVAR-ColdStart-30K
bash examples/train_coldstart.sh
```

### RL Training with AVAR

```bash
# Run GRPO training with visual anchoring reward
bash examples/train_avar.sh
```

---

## 🤖 Model Zoo

| Model | Base Model | HuggingFace | Description |
|:---|:---|:---:|:---|
| **AVAR-Thinker-7B** | Qwen2.5-VL-7B | [🤗 Link](https://huggingface.co/Antimage01/AVAR-Thinker-7B) | Full AVAR pipeline trained model |

## 📊 Dataset

| Dataset | Size | HuggingFace | Description |
|:---|:---:|:---:|:---|
| **AVAR-ColdStart-30K** | 30,419 samples | [🤗 Link](https://huggingface.co/datasets/Antimage01/AVAR-ColdStart-30K) | Curated cold-start data with visual anchoring |

---

## 📈 Evaluation Results

### Performance Comparison across Benchmarks

Best scores are **bold**, second best are <ins>underlined</ins>. Closed-source models are compared with each other, open-source models with ours.

#### Closed-Source Models

| Model | MathVista | MathVision | MathVerse-VO | MMMU-VAL | MMMU-Pro | MMStar | Hallusion. | Avg. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | <ins>63.8</ins> | <ins>31.2</ins> | - | **70.7** | **54.5** | <ins>65.1</ins> | <ins>56.2</ins> | - |
| Claude-3.7-Sonnet | **74.5** | **58.6** | - | <ins>75.2</ins> | <ins>50.1</ins> | **68.8** | **58.3** | - |

#### Open-Source General Models

| Model | MathVista | MathVision | MathVerse-VO | MMMU-VAL | MMMU-Pro | MMStar | Hallusion. | Avg. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B | 68.2 | 25.2 | 41.1 | 58.1 | 38.3 | 62.1 | 50.7 | 49.1 |
| InternVL2.5-8B | 64.4 | 22.0 | 39.5 | 56.0 | 38.2 | 63.2 | 51.1 | 47.8 |
| LLaVA-OneVision-7B | 58.6 | 18.3 | 19.3 | 48.8 | 35.5 | 61.7 | 47.5 | 41.4 |
| Llama-3.2-11B-Vision-Instruct | 48.6 | 19.7 | 18.4 | 50.7 | 33.0 | 49.8 | 40.3 | 37.2 |

#### Multimodal Reasoning Models

| Model | MathVista | MathVision | MathVerse-VO | MMMU-VAL | MMMU-Pro | MMStar | Hallusion. | Avg. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mulberry-7B† | 63.1 | - | 42.9 | 55.0 | 34.8 | 61.3 | 54.1 | - |
| R1-OneVision | 64.1 | 29.9 | 40.0 | 49.1 | 32.2 | 52.2 | 46.0 | 44.8 |
| OpenVLThinker | 72.3 | 25.9 | 44.6 | 53.0 | 42.9 | 59.5 | 53.0 | 50.2 |
| ThinkLite-VL | **75.1** | <ins>32.9</ins> | 45.8 | 55.5 | 40.0 | <ins>65.0</ins> | 52.3 | <ins>53.1</ins> |
| MM-Eureka-7B | 73.0 | 26.9 | 48.1 | 52.0 | 42.4 | **65.2** | 50.7 | 51.2 |
| Vision-R1† | 73.5 | - | 47.7 | 56.3 | 39.6 | 64.8 | 51.9 | - |
| VLAA-Thinker-7B | 68.0 | 26.4 | <ins>48.2</ins> | 55.7 | 40.9 | 64.2 | 50.9 | 50.6 |
| Vision-SR1 | 68.1 | 26.7 | 47.1 | <ins>61.3</ins> | **43.8** | 64.1 | <ins>54.3</ins> | 52.2 |

#### Our Model

| Model | MathVista | MathVision | MathVerse-VO | MMMU-VAL | MMMU-Pro | MMStar | Hallusion. | Avg. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AVAR-Thinker** | <ins>74.7</ins> | **37.4** | **50.4** | **63.8** | <ins>42.9</ins> | 64.1 | **59.5** | **56.1** |
| *Δ over Qwen2.5-VL-7B* | *+6.5* | *+12.2* | *+9.3* | *+5.7* | *+4.6* | *+2.0* | *+8.8* | *+7.0* |

> † Models trained on MathVision, so their results on MathVision are omitted.

> Please refer to our [paper](https://arxiv.org/abs/2603.03825) for complete evaluation results and analysis.

---

## 🔑 Key Findings

### Visual Attention Score (VAS)

We propose **VAS** to quantify the model's attention allocation to visual tokens:

- Models with higher VAS consistently demonstrate stronger multimodal reasoning capabilities.
- The correlation between VAS and reasoning performance reaches **r=0.9616**.

### Lazy Attention Anchoring

A counter-intuitive phenomenon discovered during cold-start:

- 🔴 **Multimodal cold-start** fails to effectively increase VAS — attention distribution remains close to the base model.
- 🟢 **Text-only cold-start** paradoxically leads to significant VAS increases.

### Training-Free Intervention

We design training-free inference interventions to directly modulate attention allocation, achieving **1%–2% performance gains** and causally validating the role of visual attention in multimodal reasoning.

---

## 📂 Project Structure

```
Qwen-AVAR/
├── EasyR1/                  # Training framework (based on EasyR1)
├── LlamaFactory/            # SFT training framework
├── examples/                # Training scripts and reward functions
│   ├── train_coldstart.sh   # Cold-start training script
│   ├── train_avar.sh        # AVAR RL training script
│   └── reward_function/     # Visual anchoring reward implementation
└── README.md
```

---

## 📝 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{luonarrow,
  title={From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning},
  author={Luo, Ruilin and Shi, Chufan and Zhang, Yizhen and Yang, Cheng and Jiang, Songtao and Guan, Tongkun and Chen, Ruizhe and Chu, Ruihang and Wang, Peng and Yang, Mingkun and others},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```

---

## 🙏 Acknowledgments

- This repository is built upon [EasyR1](https://github.com/hiyouga/EasyR1) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank the developers for their excellent training frameworks.
- We thank the [Qwen](https://github.com/QwenLM/Qwen2.5-VL) team for providing the base model.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
