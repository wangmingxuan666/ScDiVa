<!-- HERO: flush-left (full-bleed), 3 lines, auto-wrap on line 3 -->
<div style="
  background:#ffffff;
  margin-left:-9999px; margin-right:-9999px;
  padding-left:9999px; padding-right:9999px;
  padding-top:72px; padding-bottom:40px;
">
  <div style="
    max-width:1400px;
    margin:0;               /* <-- no centering */
    padding:0;              /* <-- no extra indent */
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',Arial,sans-serif;
    text-align:left;
  ">

    <!-- Line 1 -->
    <div style="
      color:#2F55D4;
      font-weight:800;
      font-size:75px;
      line-height:1.05;
      letter-spacing:-0.02em;
      margin:0;
    ">
      ScDiVa
    </div>

    <!-- Line 2 -->
    <div style="
      color:#333333;
      font-weight:800;
      font-size:40px;
      line-height:1.08;
      letter-spacing:-0.02em;
      margin-top:12px;
    ">
      A Foundation Model for Single-cell Genomics
    </div>

    <!-- Line 3 (force wrapping) -->
    <div style="
      color:#666666;
      font-weight:300;
      font-size:18px;
      line-height:1.35;
      margin-top:28px;

      /* KEY: allow wrapping everywhere it needs */
      white-space:normal;
      overflow-wrap:anywhere;
      word-break:break-word;
    ">
      Reconstruction | Multi-Batch Integration | Cell Annotation | Gene Perturbation Prediction | Gene Correlation Analysis
    </div>

  </div>
</div>


<!-- Image below -->
<p align="center" style="margin:24px 0 0;">
  <img src="./assets/scDiVa.png" alt="ScDiVa Architecture" width="1200"/>
</p>


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

### 🔬 Multi-Batch Integration

ScDiVa demonstrates superior batch integration capabilities, balancing technical noise removal (Avg-Batch) with biological conservation (Avg-Bio) across diverse benchmarks:

<div align="center" markdown="1">

*Comparison of scDiVa against leading baselines across diverse benchmarks.*

| Dataset        | Metric     | Harmony | scGPT   | **scDiVa** |
|:------------- |:----------:|:------:|:------:|:----------:|
| **PBMC12k**    | Avg-Batch  | 0.9341 | 0.9755 | **0.9960** 🏆 |
| **PBMC12k**    | Avg-Bio    | 0.7990 | 0.9018 | **0.9566** 🏆 |
| **Immune**     | Avg-Batch  | 0.9514 | 0.9194 | **0.9555** 🏆 |
| **Immune**     | Avg-Bio    | 0.6945 | **0.7879** 🏆 | 0.7785 |
| **BMMC**       | Avg-Batch  | 0.8999 | 0.8431 | **0.9734** 🏆 |
| **BMMC**       | Avg-Bio    | 0.6316 | 0.6576 | **0.8712** 🏆 |
| **Perirhinal** | Avg-Batch  | 0.9442 | **0.9600** 🏆 | 0.9542 |
| **Perirhinal** | Avg-Bio    | 0.8595 | 0.9552 | **0.9895** 🏆 |
| **COVID-19**   | Avg-Batch  | 0.8781 | 0.8625 | **0.9538** 🏆 |
| **COVID-19**   | Avg-Bio    | 0.4468 | 0.6476 | **0.6689** 🏆 |


<div align="center">
  <img src="./assets/Anno.png" alt="Cell Annotation Results" width="700"/>
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

