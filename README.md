
<div align="center">

# ScDiVa: A Foundation Model for Single-cell Genomics

<p align="center">
  <img src="./assets/scDiVa.png" alt="ScDiVa Architecture" width="1200"/>
</p>

**Core Competence**: Reconstruction | Multi-Batch Integration | Cell Annotation | Gene Perturbation Prediction | Gene Correlation Analysis

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03477-b31b1b.svg)](https://arxiv.org/abs/2602.03477)
[![Model](https://img.shields.io/badge/Model-ScDiVa-green.svg)](https://huggingface.co/warming666/ScDiVa)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/warming666/ScDiVa)

[📖 Paper](https://arxiv.org/abs/2602.03477) | [🤗 HuggingFace](https://huggingface.co/warming666/ScDiVa) | [📊 Demo](https://demo.scdiva.ai)

</div>

---

## 📢 News

- **[2026.02.03]** 🎉 ScDiVa pre-trained and fine-tuned weights are now available on [Hugging Face](https://huggingface.co/warming666/ScDiVa)!
- **[2026.02.03]** 📄 ScDiVa paper is released on arXiv.
- **[2026.10.17]** 🚀 ScDiVa project initialization.

---

## 🌟 Abstract
ScDiVa (Single-cell Masked Diffusion for Identity & Value; *scDiVa*) is a generative foundation model for single-cell representation learning, built on a **Masked Discrete Diffusion** framework that establishes an isomorphism between the forward diffusion process and sequencing **technical dropout**.  Parameterized by a bidirectional Transformer encoder, scDiVa adopts a **Dual Denoising Loss** to jointly model **gene identity** (topology) and **expression value** (dosage), enabling accurate recovery in both **Rank** and **Value** dimensions.  To robustly learn across extreme sparsity and depth variation, scDiVa further incorporates **Entropy-Normalized Serialization** and a **Depth-Invariant Sampling** strategy.  Pre-trained on **59,162,450** single-cell transcriptomes, scDiVa is systematically evaluated across tasks of increasing complexity, including **Rank-Value Joint Reconstruction**, **Multi-batch Integration**, **Cell Type Annotation** (fine-tuning and zero-shot), and **Perturbation Prediction**, with interpretability validated via **gene correlation analysis** and **Gene Regulatory Network (GRN) Inference and Logic** derived from model representations and attention signals. 

## Downstream task list

* **Rank-Value Joint Reconstruction** 
* **Multi-batch Integration** 
* **Cell Type Annotation** (fine-tuning & zero-shot) 
* **Perturbation Prediction** 
* **Gene Regulatory Network (GRN) Inference and Logic** 

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

### 🧩 Rank-Value Joint Reconstruction

We evaluate reconstruction quality using **L-Dist** (↓), **BLEU** (↑), and **Spearman** (↑) across multiple datasets.

<table>
  <tr>
    <td width="50%" valign="top">
      <h4>PBMC12k</h4>
      <table width="100%">
        <thead>
          <tr>
            <th>Model</th>
            <th align="right">L-Dist ↓</th>
            <th align="right">BLEU ↑</th>
            <th align="right">Spearman ↑</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>GeneMamba U</td>
            <td align="right">430</td>
            <td align="right">0.532</td>
            <td align="right">0.469</td>
          </tr>
          <tr>
            <td>Geneformer</td>
            <td align="right">23</td>
            <td align="right">0.968</td>
            <td align="right">0.703</td>
          </tr>
          <tr>
            <td>GeneMamba</td>
            <td align="right">6</td>
            <td align="right"><b>0.987</b> 🏆</td>
            <td align="right">0.711</td>
          </tr>
          <tr>
            <td><b>scDiVa</b></td>
            <td align="right"><b>5</b> 🏆</td>
            <td align="right"><b>0.987</b> 🏆</td>
            <td align="right"><b>0.812</b> 🏆</td>
          </tr>
        </tbody>
      </table>
    </td>
    <td width="50%" valign="top">
      <h4>Pancreas</h4>
      <table width="100%">
        <thead>
          <tr>
            <th>Model</th>
            <th align="right">L-Dist ↓</th>
            <th align="right">BLEU ↑</th>
            <th align="right">Spearman ↑</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>GeneMamba U</td>
            <td align="right">370</td>
            <td align="right">0.524</td>
            <td align="right">0.461</td>
          </tr>
          <tr>
            <td>Geneformer</td>
            <td align="right">25</td>
            <td align="right">0.956</td>
            <td align="right">0.763</td>
          </tr>
          <tr>
            <td>GeneMamba</td>
            <td align="right"><b>12</b> 🏆</td>
            <td align="right"><b>0.991</b> 🏆</td>
            <td align="right">0.792</td>
          </tr>
          <tr>
            <td><b>scDiVa</b></td>
            <td align="right">13</td>
            <td align="right">0.965</td>
            <td align="right"><b>0.812</b> 🏆</td>
          </tr>
        </tbody>
      </table>
    </td>
  </tr>
  
  <tr>
    <td width="50%" valign="top">
      <h4>Zheng68k</h4>
      <table width="100%">
        <thead>
          <tr>
            <th>Model</th>
            <th align="right">L-Dist ↓</th>
            <th align="right">BLEU ↑</th>
            <th align="right">Spearman ↑</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>GeneMamba U</td>
            <td align="right">432</td>
            <td align="right">0.581</td>
            <td align="right">0.503</td>
          </tr>
          <tr>
            <td>Geneformer</td>
            <td align="right">25</td>
            <td align="right">0.937</td>
            <td align="right">0.901</td>
          </tr>
          <tr>
            <td>GeneMamba</td>
            <td align="right">11</td>
            <td align="right"><b>0.996</b> 🏆</td>
            <td align="right">0.980</td>
          </tr>
          <tr>
            <td><b>scDiVa</b></td>
            <td align="right"><b>9</b> 🏆</td>
            <td align="right">0.992</td>
            <td align="right"><b>0.994</b> 🏆</td>
          </tr>
        </tbody>
      </table>
    </td>
    <td width="50%" valign="top">
      <h4>Immune</h4>
      <table width="100%">
        <thead>
          <tr>
            <th>Model</th>
            <th align="right">L-Dist ↓</th>
            <th align="right">BLEU ↑</th>
            <th align="right">Spearman ↑</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>GeneMamba U</td>
            <td align="right">468</td>
            <td align="right">0.659</td>
            <td align="right">0.442</td>
          </tr>
          <tr>
            <td>Geneformer</td>
            <td align="right">17</td>
            <td align="right">0.962</td>
            <td align="right">0.823</td>
          </tr>
          <tr>
            <td>GeneMamba</td>
            <td align="right">12</td>
            <td align="right"><b>0.998</b> 🏆</td>
            <td align="right">0.844</td>
          </tr>
          <tr>
            <td><b>scDiVa</b></td>
            <td align="right"><b>4</b> 🏆</td>
            <td align="right">0.997</td>
            <td align="right"><b>0.970</b> 🏆</td>
          </tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

### 🔬 Multi-Batch Integration

ScDiVa demonstrates superior batch integration capabilities, balancing technical noise removal (Avg-Batch) with biological conservation (Avg-Bio) across diverse benchmarks:

<div align="center" markdown="1">

*Comparison of scDiVa against leading baselines across diverse benchmarks.*

### Avg-Batch

| Dataset | Harmony | Geneformer | scGPT | scFoundation | GeneMamba | **scDiVa** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **Immune** | 0.9514 | 0.8153 | 0.9194 | 0.8904 | 0.9536 | **0.9555** 🏆 |
| **PBMC12k** | 0.9341 | 0.9545 | 0.9755 | 0.9628 | 0.9604 | **0.9960** 🏆 |
| **BMMC** | 0.8999 | 0.7720 | 0.8431 | 0.7598 | 0.9157 | **0.9734** 🏆 |
| **Perirhinal Cortex** | 0.9442 | 0.9127 | 0.9600 | 0.9560 | **0.9673** 🏆 | 0.9542 |
| **COVID-19** | 0.8781 | 0.8240 | 0.8625 | 0.8346 | 0.8742 | **0.9538** 🏆 |

### Avg-Bio

| Dataset | Harmony | Geneformer | scGPT | scFoundation | GeneMamba | **scDiVa** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **Immune** | 0.6945 | 0.6983 | 0.7879 | 0.7337 | **0.8131** 🏆 | 0.7785 |
| **PBMC12k** | 0.7990 | 0.7891 | 0.9018 | 0.8662 | 0.8344 | **0.9566** 🏆 |
| **BMMC** | 0.6316 | 0.6324 | 0.6576 | 0.5250 | 0.7628 | **0.8712** 🏆 |
| **Perirhinal Cortex** | 0.8595 | 0.8547 | 0.9552 | 0.9606 | 0.9062 | **0.9895** 🏆 |
| **COVID-19** | 0.4468 | 0.5567 | 0.6476 | 0.5468 | 0.5537 | **0.6689** 🏆 |
</div>

<div align="center">
  <img src="./assets/Multi.png" alt="Multi-batch integration Results" width="700"/>
</div>

---



### 🏷️ Cell Type Annotation

ScDiVa achieves high accuracy in both fine-tuning (for specific tissues) and zero-shot scenarios:

<div align="center" markdown="1">

*Evaluation of fine-tuning (adaptability) and zero-shot (generalization) capabilities.*

| Dataset | Task | Metric | scDiVa Performance | vs. SOTA / Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **hPancreas** | Fine-tuning | Accuracy | **98.6%** 🏆 | State-of-the-art |
|  |  | Macro-F1 | **0.7919** 🏆 | High discriminative power |
| **MS** | Fine-tuning | Macro-F1 | **0.7271** 🏆 | **+36%** over GeneMamba (0.5342) |
| **Zero-shot Avg** | Zero-shot | Accuracy | **91.4%** 🏆 | Outperforms scGPT (76.3%) |
|  |  | Macro-F1 | **0.841** 🏆 | Strong generalization across 8 datasets |
</div>

<div align="center">
  <img src="./assets/Anno.png" alt="Cell Annotation Results" width="700"/>
</div>

---

## 📉 Other Task Results Display

ScDiVa extends its capabilities to complex causal inference and interpretability tasks. By fine-tuning on perturbation datasets, the model effectively bridges the causal gap to predict transcriptional responses to both single and combinatorial genetic interventions. Furthermore, ScDiVa's intrinsic attention mechanisms allow for the direct extraction of interpretable global Gene Regulatory Networks (GRN).

<div align="center">
  <img src="./assets/0.png" alt="Perturbation and Evaluation" width="700"/>
</div>

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

> 💡 **For more specific experimental results, please check the [Original Paper](https://arxiv.org/abs/2602.03477).**

---

## 🗂️ Model Zoo

We provide the official pre-trained weights and task-specific fine-tuned checkpoints hosted on Hugging Face:

### Pre-trained Model

| Model Name | Parameters | Training Data | Description | Download |
|:---|:---|:---|:---|:---|
| **ScDiVa-Pretrain** | **~94.5M** | 59M cells (Multi-tissue) | The core foundation model pre-trained on 59 million single-cell transcriptomes. | [🤗 HF](https://huggingface.co/warming666/ScDiVa/tree/main) |

### Fine-tuned Models

Fine-tuned weights are organized in the `downstream` folder of our repository:

| Task | Datasets / Variants | Download |
|:---|:---|:---|
| **Batch Integration** | **5 Checkpoints:**<br>• Immune, PBMC12k, BMMC, Perirhinal, COVID-19 | [🤗 HF (Multi-batch)](https://huggingface.co/warming666/ScDiVa/tree/main/downstream/Multi-batch_Integration) |
| **Cell Annotation** | **4 FT Checkpoints:**<br>• hPancreas, MS, Myeloid, Myeloid_b<br>**Zero-shot:** Adapter weights | [🤗 HF (Annotation)](https://huggingface.co/warming666/ScDiVa/tree/main/downstream/Annotation_FT) |
| **Perturbation** | **2 Checkpoints:**<br>• Adamson (Single), Norman (Combinatorial) | [🤗 HF (Perturbation)](https://huggingface.co/warming666/ScDiVa/tree/main/downstream/Perturbation) |

---

## 📦 Datasets

We provide all pre-processed downstream task datasets used in our benchmarks. You can access the full collection directly via our Hugging Face Datasets repository:

**[📂 warming666/ScDiVa Datasets](https://huggingface.co/datasets/warming666/ScDiVa)**

Available datasets include:
- `adamson_processed.h5ad`, `norman.h5ad` (Perturbation)
- `bmmc_processed.h5ad`, `immune_atlas.h5ad`, `pbmc.h5ad`, `covid19.h5ad` (Integration)
- `hpancreas.h5ad`, `ms.h5ad` (Annotation Fine-tuning)
- `Cell_Lines.h5ad`, `DC.h5ad`, `HumanPBMC.h5ad`, `MCA.h5ad` (Zero-shot)
- *And more...*

**Download**:
```python
from datasets import load_dataset
# Example: Download specific files manually from the repo link above

```

---

## 🚀 Usage

### Installation

```bash
# Clone the repository
git clone [https://github.com/warming666/ScDiVa.git](https://github.com/warming666/ScDiVa.git)
cd ScDiVa

# Create environment
conda create -n scdiva python=3.8
conda activate scdiva

# Install dependencies
pip install -r requirements.txt

```

### Model Loading

We provide **pre-trained weights** and a **model architecture definition** for inference. You can load the model directly from Hugging Face:

```python
from modeling_scdiva import ScDiVaModel
import torch

def test_pipeline():
    print("=== Testing ScDiVa Loading & Inference ===")
    
    # 1. Load model (Auto-download or Random Init)
    # This matches your README usage exactly
    model = ScDiVaModel.from_pretrained("warming666/ScDiVa")
    model.eval()
    
    # 2. Create Dummy Data
    # Batch size = 2, Num genes must match config default (41818)
    batch_size = 2
    num_genes = 41818
    input_data = torch.randn(batch_size, num_genes)
    print(f"Input Data Shape: {input_data.shape}")

    # 3. Run Inference
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

**Note**: The inference SDK is currently undergoing internal company review for open-source release. We plan to make it publicly available upon the paper's acceptance. For early access or inquiries, please contact us at [wangmx2025@ruc.edu.cn](wangmx2025@ruc.edu.cn).

---

## 📄 Citation

If you find ScDiVa useful in your research, please consider citing:

```bibtex
@article{wang2026scdiva,
  title={ScDiva: Masked Discrete Diffusion for Joint Modeling of Single-Cell Identity and Expression},
  author={Wang, Mingxuan and Chen, Cheng and Jiang, Gaoyang and Ren, Zijia and Zhao, Chuangxin and Shi, Lu and Ma, Yanbiao},
  journal={arXiv preprint arXiv:2602.03477},
  year={2026}
}

```

---

## 📧 Contact

* **Email**: wangmx2025@ruc.edu.cn
* **Issues**: [GitHub Issues](https://www.google.com/search?q=https://github.com/warming666/ScDiVa/issues)

---

<div align="center">
<sub>
Thank you to everyone who has helped me.</sub>
</div>

