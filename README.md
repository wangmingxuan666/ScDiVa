<div align="center">

# ScDiVa: A Foundation Model for Single-cell Genomics

<p align="center">
  <img src="./assets/scDiVa.png" alt="ScDiVa Architecture" width="1200"/>
</p>


**Core Competence**: Reconstruction | Multi-Batch Integration | Cell Annotation | Gene Perturbation Prediction | Gene Correlation Analysis

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2026.xxxxx)
[![Model](https://img.shields.io/badge/Model-ScDiVa-green.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/ScDiVa)

[📖 Paper](https://arxiv.org/) | [🤗 HuggingFace](https://huggingface.co/ScDiVa) | [🔧 ModelScope](https://modelscope.cn/ScDiVa) | [📊 Demo](https://demo.scdiva.ai)

</div>

---

## 📢 News

- **[2026.02.03]** 🎉 ScDiVa pre-trained and fine-tuned weights are now available!
- **[2026.02.03]** 📄 ScDiVa paper is released on arXiv.
- **[2026.01.15]** 🚀 ScDiVa project initialization.

---

## 🌟 Abstract

**ScDiVa** (Single-cell Deep Variational Analysis) is a foundation model designed for comprehensive single-cell genomics analysis. Built upon transformer-based architecture, ScDiVa demonstrates exceptional performance across multiple downstream tasks including:

- **Batch Effect Correction**: Seamlessly integrate single-cell data from different batches and experimental conditions
- **Cell Type Annotation**: Accurate and efficient cell type identification across diverse tissues
- **Multi-task Learning**: Unified framework supporting simultaneous execution of multiple analysis tasks
- **Multi-modal Integration**: Capability to integrate different omics modalities

ScDiVa achieves state-of-the-art performance on benchmark datasets while maintaining interpretability and biological relevance.

---

## 🏗️ Model Architecture

<div align="center">
  <img src="./assets/scDiVa.png" alt="ScDiVa Model Architecture" width="800"/>
</div>
ScDiVa employs a **Masked Discrete Diffusion** framework instantiated as a bidirectional Transformer encoder. The architecture features the following key components:

- **Dual Denoising Objective**: Simultaneously optimizes gene identity reconstruction (topology) and expression value regression (dosage).
- **Latent Encoder**: Introduces a `[LAT]` anchor token to aggregate global cell context and prevent posterior collapse during generation.
- **Entropy-Normalized Serialization**: Prioritizes discriminative genes over housekeeping noise based on population-level Shannon entropy.
- **Depth-Invariant Sampling**: Simulates varying sequencing depths to ensure robust generalization across sparse datasets.

### Model Specifications

| Model | Parameters | Hidden Size | Layers | Attention Heads | Max Seq Length |
|-------|-----------|-------------|--------|----------------|----------------|
| **ScDiVa** | **~94.5M** | 512 | 12 | 8 | 1,200 genes |

> *Note: Configuration uses **SwiGLU** activation, **RoPE** positional embeddings, and **RMSNorm** for stability.*

---

## 📊 Main Results

### 🔬 Batch Integration Performance

<div align="center">
  <img src="./assets/Multi.png" alt="Batch Integration Results" width="700"/>
</div>

ScDiVa demonstrates superior batch integration capabilities, balancing technical noise removal (Avg-Batch) with biological conservation (Avg-Bio) across diverse benchmarks:

<div align="center">

### 🔬 Comprehensive Batch Integration Performance

*Comparison of scDiVa against leading baselines across diverse benchmarks.*

| Dataset | Metric | Harmony | scGPT | **scDiVa** |
| :--- | :---: | :---: | :---: | :---: |
| **PBMC12k** | Avg-Batch | 0.9341 | 0.9755 | **0.9960** 🏆 |
| | Avg-Bio | 0.7990 | 0.9018 | **0.9566** 🏆 |
| **Immune** | Avg-Batch | 0.9514 | 0.9194 | **0.9555** 🏆 |
| | Avg-Bio | 0.6945 | **0.7879** 🏆| 0.7785 |
| **BMMC** | Avg-Batch | 0.8999 | 0.8431 | **0.9734** 🏆 |
| | Avg-Bio | 0.6316 | 0.6576 | **0.8712** 🏆 |
| **Perirhinal** | Avg-Batch | 0.9442 | **0.9600** 🏆| 0.9542 |
| | Avg-Bio | 0.8595 | 0.9552 | **0.9895** 🏆 |
| **COVID-19** | Avg-Batch | 0.8781 | 0.8625 | **0.9538** 🏆 |
| | Avg-Bio | 0.4468 | 0.6476 | **0.6689** 🏆 |

<br>

</div>

---

### 🏷️ Cell Type Annotation

<div align="center">
  <img src="./assets/Anno.png" alt="Cell Annotation Results" width="700"/>
</div>

ScDiVa achieves high accuracy in both fine-tuning (for specific tissues) and zero-shot scenarios:

<div align="center">

### 🏷️ Comprehensive Cell Type Annotation Performance

*Evaluation of fine-tuning (adaptability) and zero-shot (generalization) capabilities.*

| Dataset | Task | Metric | scDiVa Performance | vs. SOTA / Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **hPancreas** | Fine-tuning | Accuracy | **98.6%** 🏆 | State-of-the-art |
| | | Macro-F1 | **0.7919** 🏆 | High discriminative power |
| **MS** | Fine-tuning | Macro-F1 | **0.7271** 🏆 | **+36%** over GeneMamba (0.5342) |
| **Zero-shot Avg** | Zero-shot | Accuracy | **91.4%** 🏆 | Outperforms scGPT (76.3%) |
| | | Macro-F1 | **0.841** 🏆 | Strong generalization across 8 datasets |

<br>

</div>

---

## 📉 Other Task Results Display

ScDiVa extends its capabilities to complex causal inference and interpretability tasks. By fine-tuning on perturbation datasets, the model effectively bridges the causal gap to predict transcriptional responses to both single and combinatorial genetic interventions. Furthermore, ScDiVa's intrinsic attention mechanisms allow for the direct extraction of interpretable global Gene Regulatory Networks (GRN), successfully recovering known biological logic such as the SPI1 regulon and critical immune pathway interactions.


<div align="center">
  <img src="./assets/0.png" alt="Perturbation and Evaluation" width="700"/>
</div>

ScDiVa supports simultaneous execution of multiple analysis tasks, ensuring high-fidelity reconstruction while modeling complex genetic interactions:

- **Rank-Value Reconstruction**: Achieves record Spearman correlations on Immune (**0.970**) and PBMC12k (**0.812**) datasets.
- **Perturbation Prediction (Adamson)**: Achieves a Pearson correlation of **0.837** and MSE of **0.134**.
- **Combinatorial Prediction (Norman)**: Successfully models non-additive genetic interactions with a correlation of **0.709**.

---
<table>
  <tr>
    <td width="45%" align="center" valign="middle">
      <img src="./assets/2.png" alt="GRN Inference" width="100%"/>
      <br>
      <b>Gene Regulatory Network Inference</b>
    </td>
    <td width="55%" valign="top">
      <h3>🧬 Interpretability & Regulatory Logic</h3>
      <p>ScDiVa natively captures interpretable gene regulatory networks (GRN) via its attention mechanism:</p>
      <ul>
        <li>
          <b>Global Topology (Fig a):</b> The global GRN visualization confirms the model's ability to distinguish functional clusters and validated interactions.
        </li>
        <li>
          <b>Master Regulator Discovery (Fig b):</b> Successfully reconstructs the <b>SPI1 regulon</b>, capturing its precise logic of <i>promoting</i> myeloid markers (e.g., <code>MS4A3</code>) while <i>repressing</i> erythroid genes (e.g., <code>HBG1/2</code>).
        </li>
        <li>
          <b>Regulatory Hubs (Fig c):</b> Attention heatmaps reveal dense connectivity clusters, identifying high-order coupling between immune defense pathways (e.g., <code>ISG15</code>, <code>SERPING1</code>) and cytoskeletal remodeling.
        </li>
      </ul>
    </td>
  </tr>
</table>
> 💡 **For more specific experimental results, additional metrics, and comprehensive analysis, please check the [Original Paper](https://arxiv.org/).**
---

## 🗂️ Model Zoo

We provide the official pre-trained weights and task-specific fine-tuned checkpoints:

### Pre-trained Model

| Model Name | Parameters | Training Data | Description | Download |
|:---|:---|:---|:---|:---|
| **ScDiVa-Pretrain** | **~94.5M** | 59M cells (Multi-tissue) | The core foundation model pre-trained on 59 million single-cell transcriptomes. Supports zero-shot tasks and further fine-tuning. | [🤗 HF](https://huggingface.co/ScDiVa/pretrain) \| [🔧 MS](https://modelscope.cn/ScDiVa/pretrain) |

### Fine-tuned Models

We release fine-tuned weights for specific downstream tasks as reported in the paper:

| Task | Datasets / Variants | Download |
|:---|:---|:---|
| **Batch Integration** | **5 Checkpoints available:**<br>• Immune Atlas<br>• PBMC12k<br>• BMMC (Bone Marrow)<br>• Perirhinal Cortex (Brain)<br>• COVID-19 (Lung) | [🤗 HF Collection](https://huggingface.co/collections/ScDiVa/batch-integration) |
| **Cell Annotation (Fine-tuning)** | **4 Checkpoints available:**<br>• hPancreas<br>• Multiple Sclerosis (MS)<br>• Myeloid<br>• Myeloid_b | [🤗 HF Collection](https://huggingface.co/collections/ScDiVa/annotation-ft) |
| **Perturbation Prediction** | **2 Checkpoints available:**<br>• Adamson (Single-gene)<br>• Norman (Double-gene/Combinatorial) | [🤗 HF Collection](https://huggingface.co/collections/ScDiVa/perturbation) |

> **Note**: For **Zero-shot Cell Annotation**, please use the base `ScDiVa-Pretrain` model directly with the provided MLP head weights (available in the model repository).

---

## 📦 Datasets

We provide all pre-processed downstream task datasets used in our benchmarks (as detailed in Appendix B.4) in the `datasets/` folder:

```text
datasets/
├── reconstruction_grn/
│   ├── immune.h5ad
│   ├── hpancreas.h5ad
│   ├── pbmc12k.h5ad
│   └── zheng68k.h5ad
├── batch_integration/
│   ├── immune_atlas.h5ad
│   ├── pbmc12k.h5ad
│   ├── bmmc.h5ad
│   ├── perirhinal_cortex.h5ad
│   └── covid19.h5ad
├── annotation/
│   ├── fine_tuning/
│   │   ├── hpancreas.h5ad
│   │   ├── ms.h5ad
│   │   ├── myeloid.h5ad
│   │   └── myeloid_b.h5ad
│   └── zero_shot/
│       ├── cell_lines.h5ad
│       ├── dc.h5ad
│       ├── human_pbmc.h5ad
│       ├── immune.h5ad
│       ├── mca.h5ad
│       ├── pbmc.h5ad
│       ├── pbmc_368k.h5ad
│       └── pancrm.h5ad
└── perturbation/
    ├── adamson.h5ad
    └── norman.h5ad

**Download**: Due to size limitations, datasets are hosted externally. Please download from:
- 🤗 HuggingFace: [https://huggingface.co/datasets/ScDiVa/downstream-tasks](https://huggingface.co/datasets/ScDiVa/downstream-tasks)
- 🔧 ModelScope: [https://modelscope.cn/datasets/ScDiVa/downstream-tasks](https://modelscope.cn/datasets/ScDiVa/downstream-tasks)

---

## 🚀 Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ScDiVa.git
cd ScDiVa

# Create environment
conda create -n scdiva python=3.8
conda activate scdiva

# Install dependencies
pip install -r requirements.txt
```

### Model Loading

Due to proprietary considerations, the training code is not open-sourced. However, we provide **pre-trained weights** and a **model architecture definition** for inference:

```python
from modeling_scdiva import ScDiVaModel
import torch

# Load pre-trained model
model = ScDiVaModel.from_pretrained("ScDiVa/base-pretrain")
model.eval()

# Inference example
with torch.no_grad():
    # input_data: gene expression matrix (batch_size, num_genes)
    embeddings = model.encode(input_data)
    predictions = model.predict(input_data, task="annotation")
```

### Inference API

For easier usage, we provide an inference SDK:

```python
from scdiva_inference import ScDiVaInference

# Initialize inference engine
engine = ScDiVaInference(model_name="base-pretrain")

# Cell type annotation
annotations = engine.annotate(adata)

# Batch integration
integrated_adata = engine.integrate_batches(adata_list)
```

**Note**: The inference SDK is currently undergoing internal company review for open-source release. We plan to make it publicly available upon the paper's acceptance. For early access or inquiries, please contact us at [contact@scdiva.ai](mailto:contact@scdiva.ai).
---

## 📖 Documentation

For detailed tutorials and API documentation, please visit:

- 📘 [Model Architecture Details](./docs/model_architecture.md)
- 📗 [Fine-tuning Guide](./docs/finetuning.md)
- 📙 [Inference Tutorial](./docs/inference.md)
- 📕 [Benchmark Results](./docs/benchmarks.md)

---

## 🛠️ Project Structure

```
ScDiVa/
├── assets/                    # Images and figures
│   ├── scDiVa.png
│   ├── batch_immune.png
│   ├── Anno.png
│   ├── Multi.png
│   ├── 0.png
│   └── 2.png
├── docs/                      # Documentation
│   ├── model_architecture.md
│   ├── finetuning.md
│   ├── inference.md
│   └── benchmarks.md
├── modeling_scdiva.py         # Model architecture definition
├── scdiva_inference.py        # Inference SDK (placeholder)
├── requirements.txt           # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📄 Citation

If you find ScDiVa useful in your research, please consider citing:

```bibtex
@article{scdiva2026,
  title={ScDiVa: A Foundation Model for Single-cell Genomics},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

---

## 📧 Contact

- **Email**: contact@scdiva.ai
- **Issues**: [GitHub Issues](https://github.com/your-org/ScDiVa/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ScDiVa/discussions)

---

## 🙏 Acknowledgments

We thank the single-cell genomics community for their valuable datasets and tools. Special thanks to:

- [Scanpy](https://scanpy.readthedocs.io/)
- [scVI-tools](https://scvi-tools.org/)
- [Seurat](https://satijalab.org/seurat/)

---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](./LICENSE) file for details.

For commercial use or custom licensing, please contact us at [license@scdiva.ai](mailto:license@scdiva.ai).

---

<div align="center">
  <sub>Built with ❤️ by the ScDiVa Team</sub>
</div>
