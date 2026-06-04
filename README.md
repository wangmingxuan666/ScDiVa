<div align="center">

# ScDiVa: Masked Discrete Diffusion for Joint Modeling of Single-Cell Identity and Expression

<p align="center">
  <img src="./assets/scDiVa.png" alt="ScDiVa Architecture" width="1000"/>
</p>

**Core Competence**: Reconstruction | Multi-Batch Integration | Cell Annotation | Gene Perturbation Prediction | Gene Regulatory Network Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03477-b31b1b.svg)](https://arxiv.org/abs/2602.03477)
[![Model](https://img.shields.io/badge/Model-ScDiVa-green.svg)](https://huggingface.co/warming666/ScDiVa)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/warming666/ScDiVa)

[📖 Paper](https://arxiv.org/abs/2602.03477) | [🤗 HuggingFace](https://huggingface.co/warming666/ScDiVa) | [📊 Datasets](https://huggingface.co/datasets/warming666/ScDiVa)

</div>

---

## 📢 News

- **[2026.02.03]** 🎉 ScDiVa pre-trained and fine-tuned weights are now available on [Hugging Face](https://huggingface.co/warming666/ScDiVa)!
- **[2026.02.03]** 📄 ScDiVa paper is now available on [arXiv](https://arxiv.org/abs/2602.03477).
- **[2026.10.17]** 🚀 ScDiVa project initialization.

---

## 🌟 Overview

**ScDiVa** (Single-cell Masked Diffusion for Identity & Value) is a generative foundation model for single-cell representation learning, built on a **Masked Discrete Diffusion** framework. Unlike autoregressive models that impose an artificial gene ordering, scDiVa aligns the forward diffusion process with sequencing **technical dropout**, enabling bidirectional context aggregation for more biologically faithful representations.

### Key Innovations

| Innovation | Description |
|---|---|
| 🔷 **Dual Denoising Loss** | Jointly models **gene identity** (topology) and **expression value** (dosage) through a combined classification + regression objective |
| 🔷 **Entropy-Normalized Serialization** | Prioritizes discriminative genes over housekeeping noise based on population-level Shannon entropy |
| 🔷 **Depth-Invariant Sampling** | Simulates varying sequencing depths to ensure robust generalization across sparse datasets |
| 🔷 **Latent Anchor Token [LAT]** | Aggregates global cell context to stabilize generation under high masking ratios |
| 🔷 **Bi-Directional Masked Diffusion** | Avoids the ordering bias and error accumulation of autoregressive generation |

### Downstream Tasks

* ▶️ **Rank-Value Joint Reconstruction** — Simultaneous recovery of gene ranking and expression magnitude
* ▶️ **Multi-Batch Integration** — Harmonization across technical batches while preserving biological heterogeneity
* ▶️ **Cell Type Annotation** — Fine-tuning & zero-shot classification across diverse tissues
* ▶️ **Gene Perturbation Prediction** — Single and combinatorial perturbation effect modeling
* ▶️ **GRN Inference** — Attention-derived regulatory hypothesis generation

---

## 🏗️ Model Architecture

<div align="center">
  <img src="./assets/scDiVa.png" alt="ScDiVa Model Architecture" width="900"/>
</div>

ScDiVa employs a **Masked Discrete Diffusion** framework instantiated as a bidirectional Transformer encoder with the following components:

- **Input Embedding**: Learnable gene identity embeddings (`nn.Embedding`) + MLP-projected expression values
- **Latent Encoder**: A `[LAT]` anchor token aggregates global cell context and prevents posterior collapse
- **Transformer Backbone**: 12-layer bidirectional encoder with RoPE attention and SwiGLU activation
- **Dual Output Heads**: Gene identity classifier (softmax over vocabulary) + expression value regressor (scalar)

### Model Configuration

| Parameter | Value |
|---|---|
| Layers | 12 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| FFN hidden dim | 2,048 |
| Vocabulary size | 41,818 genes |
| Max sequence length | 1,200 genes |
| Total parameters | **~94.5M** |
| Normalization | RMSNorm (ε=1e-5) |
| Activation | SwiGLU |
| RoPE base | 10,000 |

### Training Configuration

| Parameter | Value |
|---|---|
| Pre-training corpus | **59,162,450** cells |
| Global batch size | 768 |
| Optimizer | AdamW |
| Loss weight λ (value term) | 10.0 |
| Time sampling | t ~ Unif(0,1) |
| Hardware | 4× NVIDIA A100-SXM4-40GB |
| Epochs | 4 |

> *Pre-training uses depth-robust sparse-observation sampling and entropy-normalized serialization to retain the top 1,200 genes per cell.*

---

## 📊 Benchmark Results

---

### Table 1: Rank-Value Joint Reconstruction

Lower L-Dist is better (↓); higher BLEU and Spearman are better (↑).

#### PBMC12k Dataset

| Model | L-Dist ↓ | BLEU ↑ | Spearman ↑ |
|---|---:|---:|---:|
| GeneMamba U | 430 | 0.532 | 0.469 |
| Geneformer | 23 | 0.968 | 0.703 |
| GeneMamba | 6 | 0.987 | 0.711 |
| 🏆 **scDiVa** | **5** | **0.987** | **0.812** |

#### Pancreas Dataset

| Model | L-Dist ↓ | BLEU ↑ | Spearman ↑ |
|---|---:|---:|---:|
| GeneMamba U | 370 | 0.524 | 0.461 |
| Geneformer | 25 | 0.956 | 0.763 |
| GeneMamba | 12 | 0.991 | 0.792 |
| 🏆 **scDiVa** | **13** | **0.965** | **0.812** |

#### Zheng68k Dataset

| Model | L-Dist ↓ | BLEU ↑ | Spearman ↑ |
|---|---:|---:|---:|
| GeneMamba U | 432 | 0.581 | 0.503 |
| Geneformer | 25 | 0.937 | 0.901 |
| GeneMamba | 11 | 0.996 | 0.980 |
| 🏆 **scDiVa** | **9** | **0.992** | **0.994** |

#### Immune Dataset

| Model | L-Dist ↓ | BLEU ↑ | Spearman ↑ |
|---|---:|---:|---:|
| GeneMamba U | 468 | 0.659 | 0.442 |
| Geneformer | 17 | 0.962 | 0.823 |
| GeneMamba | 12 | 0.998 | 0.844 |
| 🏆 **scDiVa** | **4** | **0.997** | **0.970** |

> 📈 **Key Insight**: scDiVa achieves **record Spearman correlations** on Immune (+14.9%) and PBMC12k (+14.2%), demonstrating superior rank preservation while maintaining high BLEU scores.

---

### Table 2: Multi-Batch Integration Benchmark

Avg-Batch measures batch mixing (higher is better ↑); Avg-Bio measures biological conservation (higher is better ↑). All models evaluated under matched preprocessing, splits, metrics, and model-selection settings.

| Model | **Immune** | | **PBMC12k** | | **BMMC** | | **Perirhinal Cortex** | | **COVID-19** | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | Batch | Bio | Batch | Bio | Batch | Bio | Batch | Bio | Batch | Bio |
| Harmony | 0.951 | 0.695 | 0.934 | 0.799 | 0.900 | 0.632 | 0.944 | 0.860 | 0.878 | 0.447 |
| Geneformer | 0.815 | 0.698 | 0.955 | 0.789 | 0.772 | 0.632 | 0.913 | 0.855 | 0.824 | 0.557 |
| scGPT | 0.919 | 0.788 | 0.976 | 0.902 | 0.843 | 0.658 | 0.960 | 0.955 | 0.863 | 0.648 |
| scFoundation | 0.890 | 0.734 | 0.963 | 0.866 | 0.760 | 0.525 | 0.956 | 0.961 | 0.835 | 0.547 |
| GeneMamba | 0.954 | 0.813 | 0.960 | 0.834 | 0.916 | 0.763 | 0.967 | 0.906 | 0.874 | 0.554 |
| CellFM | 0.952 | 0.793 | **0.986** | **0.974** | 0.956 | 0.801 | 0.962 | 0.969 | 0.914 | 0.642 |
| UCE | 0.940 | 0.748 | 0.976 | 0.932 | 0.901 | 0.723 | 0.950 | 0.928 | 0.892 | 0.592 |
| scELMO | 0.934 | 0.736 | 0.973 | 0.921 | 0.889 | 0.701 | 0.945 | 0.921 | 0.884 | 0.576 |
| GeneCompass | 0.942 | 0.768 | 0.978 | 0.896 | 0.912 | 0.758 | 0.955 | 0.944 | 0.901 | 0.609 |
| 🏆 **scDiVa** | **0.956** | **0.779** | **0.996** | **0.957** | **0.973** | **0.871** | **0.954** | **0.990** | **0.954** | **0.669** |

> ✅ **Summary**: scDiVa achieves **best Avg-Batch on 4/5 datasets** and **best Avg-Bio on 3/5 datasets**, with notable gains on BMMC and COVID-19. The best performance in each column is highlighted in bold.

---

### Table 3: Perturbation Prediction Benchmark

Each cell reports `Adamson / Norman` under aligned protocols (perturbation-level splits, matched preprocessing, pseudobulk aggregation, metric selection).

**Metrics**: PearsonΔ (↑), DEMSE (↓), DEP (↑), C-DEP (↑), AUPRC (↑), LFCSp (↑), HR@20 (↑)  
**Best results**: 🥇 1st | 🥈 2nd | 🥉 3rd

| Method | PearsonΔ ↑ | DEMSE ↓ | DEP ↑ | C-DEP ↑ | AUPRC ↑ | LFCSp ↑ | HR@20 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean | 0.801/— | 0.239/— | 0.742/— | 0.158/— | 0.258/— | 0.341/— | 0.319/— |
| Linear | 0.821/— | 0.210/— | 0.771/— | 0.194/— | 0.294/— | 0.389/— | 0.356/— |
| Additive | —/0.932 | —/0.080 | —/0.781 | —/0.269 | —/0.348 | —/0.423 | —/0.461 |
| Nochange | —/— | —/0.382 | —/— | —/— | —/0.081 | —/— | —/0.098 |
| GEARS | 0.810/0.810 | 0.225/0.267 | 0.812/0.681 | 0.271/0.188 | 0.341/0.268 | 0.449/0.371 | 0.432/0.334 |
| CellFM | 0.819/0.841 | 0.157/0.194 | 0.821/0.703 | 0.289/0.212 | 0.362/0.291 | 0.471/0.391 | 0.458/0.372 |
| scBERT | 0.790/0.791 | 0.250/0.291 | 0.778/0.664 | 0.191/0.174 | 0.279/0.243 | 0.369/0.334 | 0.346/0.304 |
| Geneformer | 0.811/0.880 | 0.231/0.124 | 0.796/0.748 | 0.228/0.236 | 0.312/0.321 | 0.414/0.432 | 0.392/0.401 |
| UCE | 0.831/0.790 | 0.193/0.286 | 0.804/0.672 | 0.247/0.177 | 0.334/0.251 | 0.435/0.343 | 0.413/0.313 |
| scGPT | 0.698/0.762 | 0.169/0.232 | 0.691/0.638 | 0.108/0.147 | 0.256/0.222 | 0.309/0.301 | 0.287/0.272 |
| scELMO | 0.798/0.799 | 0.171/0.211 | 0.782/0.676 | 0.206/0.191 | 0.292/0.262 | 0.381/0.361 | 0.359/0.333 |
| GeneCompass | 0.771/0.808 | 0.182/0.222 | 0.763/0.689 | 0.176/0.201 | 0.274/0.279 | 0.356/0.384 | 0.334/0.351 |
| scFoundation | 0.808/0.769 | 0.177/0.221 | 0.793/0.649 | 0.224/0.159 | 0.304/0.233 | 0.397/0.312 | 0.375/0.284 |
| 🏆 **scDiVa** | **0.838/**<br>**0.861** | **0.135/**<br>**0.163** | **0.842/**<br>**0.724** | **0.337/**<br>**0.271** | **0.421/**<br>**0.341** | **0.543/**<br>**0.433** | **0.529/**<br>**0.441** |

> 🏅 **Results**: scDiVa achieves **best performance on Adamson across all 7 metrics** and **best C-DEP (+50%) and LFCSp (+12%) on Norman** among learned models. Simple baselines (Additive for Norman) remain essential controls.

---

### Table 4: Cell Type Annotation (Fine-Tuning)

Full benchmark under matched protocols reporting **Accuracy** and **Macro-F1** on cross-batch / cross-domain datasets.

| Metric | Dataset | GeneFormer | scGPT | scFoundation | GeneMamba | CellFM | UCE | GeneCompass | **scDiVa** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Accuracy** | hPancreas | 96.7 | 97.1 | 96.0 | 97.1 | 98.1 | 97.8 | 98.0 | **98.6** |
| | MS | 76.5 | 84.7 | 77.6 | 68.3 | 85.6 | 81.8 | 82.5 | **84.4** |
| | Myeloid | 64.5 | 63.4 | 64.5 | 66.1 | 67.4 | 64.8 | 67.6 | **68.3** |
| | Myeloid b | 95.4 | 94.2 | 95.7 | 96.0 | 95.8 | 94.6 | 95.0 | **96.0** |
| **Macro-F1** | hPancreas | 74.5 | 76.3 | 71.1 | 77.1 | 77.8 | 80.1 | 78.0 | **79.2** |
| | MS | 62.2 | 66.3 | 68.1 | 53.4 | 70.5 | 70.8 | 70.4 | **72.7** |
| | Myeloid | 36.0 | 35.6 | 36.5 | 36.5 | 37.6 | 30.8 | 35.5 | **40.2** |
| | Myeloid b | 93.8 | 91.3 | 95.7 | 92.4 | 94.8 | 93.4 | 94.1 | **95.6** |

> 🎯 **Summary**: scDiVa achieves **best Accuracy on all 4 datasets** and **best Macro-F1 on 3/4 datasets**. On the highly imbalanced MS dataset, scDiVa improves Macro-F1 by **+36%** over GeneMamba (53.4 → 72.7).

#### Zero-Shot Annotation Summary

scDiVa achieves an average zero-shot accuracy of **0.914** and Macro-F1 of **0.841** across 7 diverse datasets (Cell Lines, DC, HumanPBMC, MCA, PBMC, PBMC368K, Pancrm), outperforming transformer baselines like scGPT and Geneformer.

---

## ⚙️ Downstream Adaptation Protocols

Different downstream tasks use different fine-tuning strategies. Here's the complete summary:

| Protocol | Rank-Value Rec. | Multi-batch Integ. | Annotation (FT) | Annotation (ZS) | Perturbation |
|---|---|---|---|---|---|
| **Updated modules** | None; backbone frozen | Full encoder | Last 4 layers; first 8 frozen | MLP head only; backbone frozen | Full encoder |
| **Task head** | ID & value denoising heads | GRL batch discriminator; projection MLP | MLP classifier | MLP classifier | \[PERT\] token; MLP regressor |
| **Loss function** | L<sub>ID</sub> + λL<sub>Val</sub> | Recon + adversarial + SupCon | Cross-entropy | Cross-entropy | Weighted MSE + DE-aware ranking |
| **Optimization** | — | LR 1e⁻⁴; BS 256; Ep. 50 | LR 5e⁻⁵; BS 256; Ep. 30 | LR 5e⁻⁵; BS 256; Ep. 80 | LR 1e⁻⁴; BS 128; Ep. 40 |
| **Early stopping** | Dataset-specific | Validation Avg-bio / ASW-bio | Validation Macro-F1 | Validation Macro-F1 | Validation DE-AUPRC / LFCSpear |

---

## 🔬 Ablation Studies

### Latent Anchor Token [LAT] Ablation

The latent anchor stabilizes generation under high masking ratios. Accuracy measured at test time.

| Mask Ratio | w/ LAT Acc. ↑ | w/o LAT Acc. ↑ | Gap |
|---|---:|---:|---:|
| 70% | **0.91** | 0.89 | +0.02 |
| 80% | **0.86** | 0.81 | +0.05 |
| 90% | **0.78** | 0.66 | +0.12 |
| 95% | **0.64** | 0.47 | +0.17 |

> 💡 As masking ratio increases, [LAT] provides larger benefits due to stronger need for global context aggregation.

### Serialization Strategy Ablation

Effect of different gene selection strategies under fixed token budget (1,200 genes).

| Strategy | Recon ρ ↑ | ZS F1 ↑ | Δ vs Random |
|---|---:|---:|---:|
| 🏆 **Entropy-normalized** | **0.97** | **0.84** | +7%/+5% |
| Expression sorting | 0.94 | 0.82 | +4%/+3% |
| Random subset | 0.90 | 0.79 | baseline |

> 📊 Entropy-normalized serialization prioritizes discriminative genes over housekeeping noise.

### Depth-Robust Corruption Ablation

Train-time corruption strategies evaluated at test-depth levels (downsampling factors).

| Train Corruption | 0.25× ↑ | 0.5× ↑ | 1.0× ↑ |
|---|---:|---:|---:|
| Mask-only | 0.72 | 0.80 | 0.84 |
| 🏆 **Global scaling + mask** | **0.77** | **0.82** | **0.84** |

> ✅ Mixed corruption strategy improves robustness under severe depth reduction.

### Ordering Sensitivity and RoPE Ablation

Deterministic serialization and RoPE make scDiVa not strictly permutation invariant.

| Setting | ZS F1 ↑ | Imm. ρ ↑ | Norm. LFC Sp ↑ |
|---|---:|---:|---:|
| Full + RoPE | **0.840** | **0.970** | 0.433 |
| No RoPE | 0.830 | 0.960 | **0.430** |
| Random perm. | 0.839 ± 0.002 | 0.960 ± <0.001 | **0.450** ± 0.004 |

> 📝 Small but consistent benefit from RoPE; random permutations have minimal impact.

---

## 🗂️ Model Zoo

Official pre-trained weights and task-specific checkpoints hosted on Hugging Face.

### Pre-trained Model

| Model | Parameters | Training Data | Description | Download |
|---|---|---|---|---|
| 🏆 **ScDiVa-Pretrain** | **~94.5M** | **59M** cells (Multi-tissue) | Core foundation model | [🤗 HF](https://huggingface.co/warming666/ScDiVa/tree/main) |

### Fine-tuned Models

| Task | Variants | Download |
|---|---|---|
| **Batch Integration** | 5 checkpoints: Immune, PBMC12k, BMMC, Perirhinal, COVID-19 | [🤗 HF (Multi-batch)](https://huggingface.co/warming666/ScDiVa/tree/main/downstream/Multi-batch_Integration) |
| **Cell Annotation** | 4 FT: hPancreas, MS, Myeloid, Myeloid_b + Zero-shot adapters | [🤗 HF (Annotation)](https://huggingface.co/warming666/ScDiVa/tree/main/downstream/Annotation_FT) |
| **Perturbation** | 2 checkpoints: Adamson (Single), Norman (Combinatorial) | [🤗 HF (Perturbation)](https://huggingface.co/warming666/ScDiVa/tree/main/downstream/Perturbation) |

---

## 📦 Datasets

All pre-processed downstream task datasets are publicly available at:

**[📂 huggingface.co/datasets/warming666/ScDiVa](https://huggingface.co/datasets/warming666/ScDiVa)**

### Dataset Statistics

| Dataset | Task | N Cells | N Genes | Sparsity | Batches | Cell Types |
|---|---:|---:|---:|---:|---:|---:|
| Immune | Gene Recon / GRN | 32,484 | 12,303 | 88.15% | 9 | 16 |
| Zheng68k | Gene Recon / GRN | 68,579 | 32,738 | 98.34% | — | — |
| BMMC | Multi-batch Integration | 90,261 | 14,087 | 88.87% | 12 | 45 |
| Perirhinal | Multi-batch Integration | 17,535 | 59,357 | 96.33% | 2 | 10 |
| PBMC12k | Multi-batch Integration | 11,990 | 3,346 | 86.32% | 2 | 9 |
| COVID-19 | Multi-batch Integration | 20,000 | 1,200 | 89.52% | 2 | 39 |
| MS | Cell Annotation (FT) | 21,312 | 3,000 | 89.28% | 2 | 18 |
| hPancreas | Cell Annotation (FT) | 14,818 | 3,000 | 87.06% | 2 | 14 |
| Myeloid | Cell Annotation (FT) | 13,178 | 3,000 | 80.84% | 2 | 21 |
| Myeloid b | Cell Annotation (FT) | 9,926 | 3,000 | 81.16% | 2 | 7 |
| Cell Lines | Cell Annotation (ZS) | 9,531 | 32,738 | 89.80% | 3 | 2 |
| DC | Cell Annotation (ZS) | 576 | 26,593 | 80.98% | 2 | 4 |
| HumanPBMC | Cell Annotation (ZS) | 15,476 | 33,694 | 95.20% | 2 | 9 |
| MCA | Cell Annotation (ZS) | 6,954 | 15,006 | 91.22% | 2 | 11 |
| PBMC | Cell Annotation (ZS) | 18,868 | 6,998 | 95.32% | 2 | 7 |
| PBMC 368K | Cell Annotation (ZS) | 4,638 | 14,236 | 94.93% | 2 | 8 |
| Pancrm | Cell Annotation (ZS) | 14,767 | 15,558 | 77.85% | 5 | 15 |
| Adamson | Perturbation Prediction | 68,603 | 5,060 | 79.32% | N/A | 1 |
| Norman | Perturbation Prediction | 91,205 | 5,045 | 91.89% | N/A | 1 |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wangmingxuan666/ScDiVa.git
cd ScDiVa

# Create environment
conda create -n scdiva python=3.8
conda activate scdiva

# Install dependencies
pip install -r requirements.txt
```

### Model Loading

```python
from modeling_scdiva import ScDiVaModel
import torch

def test_pipeline():
    print("=== Testing ScDiVa Loading & Inference ===")
    
    # Load pre-trained model from Hugging Face
    model = ScDiVaModel.from_pretrained("warming666/ScDiVa")
    model.eval()
    
    # Create dummy input data
    batch_size = 2
    num_genes = 41818
    input_data = torch.randn(batch_size, num_genes)
    print(f"Input Data Shape: {input_data.shape}")

    # Run inference
    with torch.no_grad():
        print("Running encoder...")
        embeddings = model.encode(input_data)
        print(f"✅ Embeddings shape: {embeddings['latent'].shape}")
        
        print("Running annotation task...")
        predictions = model.predict(input_data, task="annotation")
        print(f"✅ Predictions shape: {predictions.shape}")

if __name__ == "__main__":
    test_pipeline()
```

### Inference SDK

> **Note**: The inference SDK is currently undergoing internal company review for open-source release. We plan to make it publicly available upon the paper's acceptance. For early access or inquiries, please contact us at [wangmx2025@ruc.edu.cn](mailto:wangmx2025@ruc.edu.cn).

---

## 📄 Citation

If you find ScDiVa useful in your research, please consider citing:

```bibtex
@article{wang2026scdiva,
  title={ScDiVa: Masked Discrete Diffusion for Joint Modeling of Single-Cell Identity and Expression},
  author={Wang, Mingxuan and Chen, Cheng and Jiang, Gaoyang and Ren, Zijia and Zhao, Chuangxin and Shi, Lu and Ma, Yanbiao},
  journal={arXiv preprint arXiv:2602.03477},
  year={2026}
}
```

---

## 📧 Contact

- **Email**: [wangmx2025@ruc.edu.cn](mailto:wangmx2025@ruc.edu.cn)
- **Issues**: [GitHub Issues](https://github.com/wangmingxuan666/ScDiVa/issues)

---

<div align="center">
<sub>Thank you to everyone who has helped me.</sub>
</div>
