# ScDiVa Benchmark Results

This document provides comprehensive benchmark results of ScDiVa across various single-cell analysis tasks.

## Evaluation Datasets

| Dataset | Description | #Cells | #Genes | #Cell Types | #Batches |
|---------|-------------|--------|--------|-------------|----------|
| PBMC | Peripheral blood mononuclear cells | 150K | 33K | 18 | 8 |
| Immune Atlas | Human immune cell atlas | 500K | 28K | 45 | 15 |
| COVID-19 | COVID-19 immune response | 200K | 30K | 22 | 12 |
| Brain | Human brain atlas | 350K | 32K | 25 | 10 |
| Pancreas | Pancreatic islet cells | 120K | 27K | 14 | 6 |
| Heart | Human heart atlas | 180K | 29K | 20 | 8 |

## Task 1: Batch Integration

### Quantitative Metrics

| Model | ASW ↑ | kBET ↑ | Graph Conn. ↑ | PCR ↓ | iLISI ↑ |
|-------|-------|--------|---------------|-------|---------|
| **ScDiVa-Large** | **0.82** | **0.75** | **0.91** | **0.08** | **4.2** |
| **ScDiVa-Base** | **0.80** | **0.73** | **0.89** | **0.10** | **4.0** |
| Seurat v5 | 0.72 | 0.65 | 0.84 | 0.15 | 3.5 |
| Harmony | 0.70 | 0.63 | 0.82 | 0.18 | 3.3 |
| scVI | 0.75 | 0.68 | 0.86 | 0.12 | 3.7 |
| scANVI | 0.76 | 0.70 | 0.87 | 0.11 | 3.8 |

*Metrics: ASW (Average Silhouette Width), kBET (k-nearest neighbor Batch Effect Test), Graph Connectivity, PCR (Principle Component Regression), iLISI (Integration Local Inverse Simpson's Index)*

### Performance by Dataset

#### PBMC Dataset

| Method | ASW | kBET | Runtime |
|--------|-----|------|---------|
| **ScDiVa-Base** | **0.82** | **0.75** | **8s** |
| Seurat v5 | 0.74 | 0.67 | 45s |
| Harmony | 0.71 | 0.64 | 32s |
| scVI | 0.76 | 0.70 | 120s |

#### Immune Atlas Dataset

| Method | ASW | kBET | Runtime |
|--------|-----|------|---------|
| **ScDiVa-Large** | **0.79** | **0.71** | **35s** |
| **ScDiVa-Base** | **0.77** | **0.69** | **25s** |
| Seurat v5 | 0.70 | 0.62 | 180s |
| scVI | 0.73 | 0.66 | 450s |

## Task 2: Cell Type Annotation

### Overall Accuracy

| Model | Accuracy | F1-Score | Precision | Recall | Inference Time |
|-------|----------|----------|-----------|--------|----------------|
| **ScDiVa-Large** | **95.8%** | **0.95** | **0.96** | **0.95** | **0.08s/k cells** |
| **ScDiVa-Base** | **94.2%** | **0.94** | **0.94** | **0.93** | **0.05s/k cells** |
| CellTypist | 91.5% | 0.91 | 0.92 | 0.90 | 0.12s/k cells |
| scANVI | 92.3% | 0.92 | 0.93 | 0.91 | 0.25s/k cells |
| SingleR | 89.7% | 0.89 | 0.90 | 0.88 | 0.35s/k cells |

### Per-Tissue Performance

#### PBMC (18 cell types)

| Model | Accuracy | F1-Score | Rare Cell F1* |
|-------|----------|----------|---------------|
| **ScDiVa-Base** | **95.3%** | **0.94** | **0.88** |
| CellTypist | 91.8% | 0.91 | 0.82 |
| scANVI | 92.5% | 0.92 | 0.84 |

*F1 score for cell types with <5% abundance

#### Brain (25 cell types)

| Model | Accuracy | F1-Score | Rare Cell F1 |
|-------|----------|----------|--------------|
| **ScDiVa-Large** | **93.2%** | **0.92** | **0.86** |
| **ScDiVa-Base** | **92.7%** | **0.91** | **0.84** |
| CellTypist | 88.9% | 0.88 | 0.78 |

#### Pancreas (14 cell types)

| Model | Accuracy | F1-Score | Rare Cell F1 |
|-------|----------|----------|--------------|
| **ScDiVa-Base** | **96.1%** | **0.95** | **0.91** |
| CellTypist | 93.2% | 0.93 | 0.87 |
| scANVI | 94.1% | 0.94 | 0.88 |

## Task 3: Multi-task Learning

### Task Combination Performance

| Task Combination | Single-Task Avg | Multi-Task | Δ Performance |
|------------------|-----------------|------------|---------------|
| Annotation + Integration | 94.5% | 93.8% | -0.7% |
| Annotation + Clustering | 94.2% | 93.6% | -0.6% |
| Integration + Clustering | 93.8% | 93.3% | -0.5% |
| All Three Tasks | 94.2% | 92.8% | -1.4% |

### Inference Speed Comparison

| Configuration | Time per 10K cells |
|---------------|-------------------|
| Single Task | 0.5s |
| 2 Tasks Sequential | 1.0s |
| 2 Tasks Multi-task | 0.6s |
| 3 Tasks Sequential | 1.5s |
| 3 Tasks Multi-task | 0.8s |

**Speedup**: 1.5-1.9x with multi-task inference

## Task 4: Cross-species Generalization

### Human → Mouse Transfer

| Model | Accuracy (zero-shot) | Accuracy (fine-tuned) |
|-------|---------------------|----------------------|
| **ScDiVa-Large** | **78.3%** | **91.2%** |
| **ScDiVa-Base** | **75.1%** | **89.5%** |
| CellTypist | 68.5% | 85.3% |
| scVI | 72.1% | 87.8% |

## Computational Efficiency

### Training Cost

| Model | Hardware | Training Time | #GPUs |
|-------|----------|---------------|-------|
| ScDiVa-Large | A100 80GB | 120 hours | 8 |
| ScDiVa-Base | A100 40GB | 48 hours | 4 |

### Inference Throughput

| Model | Device | Cells/Second |
|-------|--------|--------------|
| ScDiVa-Large | A100 | 4,000 |
| ScDiVa-Large | V100 | 2,500 |
| ScDiVa-Base | A100 | 10,000 |
| ScDiVa-Base | V100 | 6,500 |
| ScDiVa-Base | CPU (32-core) | 1,200 |

### Memory Requirements

| Model | Training (BS=128) | Inference (BS=256) |
|-------|------------------|-------------------|
| ScDiVa-Large | 48 GB | 12 GB |
| ScDiVa-Base | 16 GB | 4 GB |

## Ablation Studies

### Architecture Components

| Configuration | Annotation Acc | Integration ASW | Δ Params |
|---------------|---------------|-----------------|----------|
| Full Model | 94.2% | 0.80 | 350M |
| - Variational Layer | 92.8% | 0.76 | 348M |
| - Multi-head Attention | 91.3% | 0.73 | 280M |
| - Layer Normalization | 90.5% | 0.71 | 350M |
| Half Depth (12 layers) | 92.1% | 0.77 | 175M |

### Pre-training Data Scale

| Pre-training Cells | Annotation Acc | Integration ASW |
|-------------------|---------------|-----------------|
| 1M | 89.5% | 0.72 |
| 5M | 92.3% | 0.76 |
| 10M | 94.2% | 0.80 |
| 20M | 95.1% | 0.82 |

## Robustness Analysis

### Performance vs Data Quality

| Noise Level | Accuracy Drop | ASW Drop |
|-------------|--------------|----------|
| 0% (clean) | 0% | 0% |
| 10% noise | -1.2% | -0.03 |
| 25% noise | -3.5% | -0.08 |
| 50% noise | -8.1% | -0.15 |

### Performance vs Dataset Size

| #Cells | Annotation Acc | Integration ASW |
|--------|---------------|-----------------|
| 1K | 88.3% | 0.68 |
| 10K | 92.5% | 0.76 |
| 100K | 94.2% | 0.80 |
| 1M | 95.1% | 0.82 |

## Summary

ScDiVa demonstrates:
- **State-of-the-art accuracy** across multiple tasks
- **Superior computational efficiency** compared to existing methods
- **Strong generalization** to unseen datasets and species
- **Robust performance** under various data conditions

---

For detailed methodology and additional results, please refer to our paper.
