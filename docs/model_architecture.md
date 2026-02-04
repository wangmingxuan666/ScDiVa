# ScDiVa Model Architecture

## Overview

ScDiVa employs a transformer-based encoder architecture with variational components, specifically designed for single-cell genomics analysis. This document provides detailed technical specifications of the model architecture.

## Architecture Components

### 1. Gene Expression Encoder

The input layer processes raw gene expression profiles:

```
Input: (batch_size, num_genes)
  ↓
Gene Embedding Layer
  ↓
Output: (batch_size, hidden_size)
```

**Key Features:**
- Linear projection from gene space to hidden space
- Layer normalization for stable training
- Dropout for regularization

### 2. Transformer Encoder

The core encoding module consists of stacked transformer layers:

```
Input: (batch_size, hidden_size)
  ↓
Multi-Head Self-Attention × N layers
  ├── Query/Key/Value projections
  ├── Scaled dot-product attention
  └── Residual connections + LayerNorm
  ↓
Feed-Forward Network × N layers
  ├── Linear → GELU → Linear
  └── Residual connections + LayerNorm
  ↓
Output: (batch_size, hidden_size)
```

**Specifications:**
- **Base Model**: 24 layers, 16 attention heads
- **Large Model**: 32 layers, 32 attention heads
- Attention mechanism captures gene-gene interactions
- Feed-forward networks enhance representation capacity

### 3. Variational Layer

The variational component enables probabilistic representations:

```
Input: (batch_size, hidden_size)
  ↓
μ (mean) projection → (batch_size, latent_dim)
σ² (variance) projection → (batch_size, latent_dim)
  ↓
Reparameterization: z = μ + σ * ε, ε ~ N(0,1)
  ↓
Output: z ~ N(μ, σ²)
```

**Benefits:**
- Captures biological variation
- Enables uncertainty quantification
- Regularizes representation learning

### 4. Task-Specific Decoders

#### Cell Type Annotation Head
```
Latent (latent_dim) → Dense (hidden_size) → GELU → Classifier (num_cell_types)
```

#### Batch Integration Head
```
Latent (latent_dim) → Dense (hidden_size) → GELU → Decoder (num_genes)
```

## Model Variants

### ScDiVa-Base

| Component | Specification |
|-----------|--------------|
| Parameters | 350M |
| Hidden Size | 1024 |
| Layers | 24 |
| Attention Heads | 16 |
| Latent Dimension | 128 |
| Max Genes | 20,000 |

### ScDiVa-Large

| Component | Specification |
|-----------|--------------|
| Parameters | 1.5B |
| Hidden Size | 2048 |
| Layers | 32 |
| Attention Heads | 32 |
| Latent Dimension | 256 |
| Max Genes | 30,000 |

## Training Objectives

### 1. Reconstruction Loss (Batch Integration)
```
L_recon = MSE(x_reconstructed, x_original)
```

### 2. Classification Loss (Annotation)
```
L_cls = CrossEntropy(predictions, labels)
```

### 3. KL Divergence (Variational Regularization)
```
L_kl = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

### 4. Total Loss
```
L_total = L_task + β * L_kl
```
where β is a hyperparameter balancing reconstruction and regularization.

## Computational Efficiency

### Memory Requirements

| Model | Training (mixed precision) | Inference (fp16) |
|-------|---------------------------|------------------|
| Base | ~16 GB | ~4 GB |
| Large | ~48 GB | ~12 GB |

### Inference Speed

On a single A100 GPU:
- **Base Model**: ~10,000 cells/second
- **Large Model**: ~4,000 cells/second

## Implementation Details

### Initialization
- Weights: Normal distribution (μ=0, σ=0.02)
- Layer norms: γ=1, β=0

### Regularization
- Dropout: 0.1 (default)
- Weight decay: 0.01
- Gradient clipping: 1.0

### Normalization
- Layer normalization after each sub-layer
- Epsilon: 1e-12 for numerical stability

## References

The architecture draws inspiration from:
- Transformer models (Vaswani et al., 2017)
- Variational Autoencoders (Kingma & Welling, 2013)
- Single-cell foundation models (Cui et al., 2023)

---

For code implementation, see `modeling_scdiva.py` in the repository root.
