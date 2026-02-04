# ScDiVa Inference Guide

This guide demonstrates how to use ScDiVa models for various single-cell analysis tasks.

## Installation

```bash
pip install scdiva
# or from source
pip install git+https://github.com/your-org/ScDiVa.git
```

## Quick Start

### 1. Cell Type Annotation

```python
import scanpy as sc
from scdiva_inference import ScDiVaInference

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize inference engine
engine = ScDiVaInference(model_name="base-annotation")

# Annotate cell types
cell_types = engine.annotate(adata, return_probabilities=False)

# Add to AnnData object
adata.obs['predicted_cell_type'] = cell_types

# Visualize
sc.pl.umap(adata, color='predicted_cell_type')
```

### 2. Batch Integration

```python
from scdiva_inference import ScDiVaInference

# Load multiple batches
adata1 = sc.read_h5ad("batch1.h5ad")
adata2 = sc.read_h5ad("batch2.h5ad")

# Initialize inference engine
engine = ScDiVaInference(model_name="base-batch-integration")

# Integrate batches
integrated_adata = engine.integrate_batches(
    [adata1, adata2],
    batch_key="batch"
)

# Visualize integration
sc.pl.umap(integrated_adata, color=['batch', 'cell_type'])
```

### 3. Latent Embedding Extraction

```python
from scdiva_inference import ScDiVaInference

# Initialize engine
engine = ScDiVaInference(model_name="base-pretrain")

# Extract embeddings
embeddings = engine.get_embeddings(adata)

# Add to AnnData
adata.obsm['X_scdiva'] = embeddings

# Use for downstream analysis
sc.pp.neighbors(adata, use_rep='X_scdiva')
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')
```

### 4. Multi-task Inference

```python
from scdiva_inference import ScDiVaInference

# Initialize engine
engine = ScDiVaInference(model_name="large-multitask")

# Run multiple tasks
results = engine.predict_multi_task(
    adata,
    tasks=["annotation", "integration", "clustering"]
)

# Access results
cell_types = results["annotation"]
integrated_data = results["integration"]
cluster_labels = results["clustering"]
```

## Advanced Usage

### Custom Model Loading

```python
from modeling_scdiva import ScDiVaModel, ScDiVaConfig

# Load model with custom configuration
config = ScDiVaConfig(
    num_genes=20000,
    hidden_size=1024,
    num_hidden_layers=24
)
model = ScDiVaModel(config)

# Load pretrained weights
model.load_state_dict(torch.load("path/to/weights.pt"))
model.eval()

# Manual inference
with torch.no_grad():
    gene_expression = torch.tensor(adata.X.toarray()).float()
    encoding = model.encode(gene_expression)
    predictions = model.predict(gene_expression, task="annotation")
```

### Batch Processing for Large Datasets

```python
from scdiva_inference import ScDiVaInference
import numpy as np

engine = ScDiVaInference(model_name="base-pretrain")

# Process in batches
batch_size = 1000
n_cells = adata.n_obs
all_embeddings = []

for start_idx in range(0, n_cells, batch_size):
    end_idx = min(start_idx + batch_size, n_cells)
    batch_adata = adata[start_idx:end_idx]
    
    embeddings = engine.get_embeddings(batch_adata)
    all_embeddings.append(embeddings)

# Concatenate results
final_embeddings = np.vstack(all_embeddings)
adata.obsm['X_scdiva'] = final_embeddings
```

### GPU Acceleration

```python
from scdiva_inference import ScDiVaInference

# Explicitly use GPU
engine = ScDiVaInference(
    model_name="base-pretrain",
    device="cuda:0",
    use_gpu=True
)

# Check device
print(f"Running on: {engine.device}")
```

### Probability-based Annotation

```python
from scdiva_inference import ScDiVaInference
import matplotlib.pyplot as plt

engine = ScDiVaInference(model_name="base-annotation")

# Get predictions with probabilities
cell_types, probabilities = engine.annotate(
    adata,
    return_probabilities=True
)

# Analyze prediction confidence
max_probs = probabilities.max(axis=1)
low_confidence_mask = max_probs < 0.5

print(f"Low confidence predictions: {low_confidence_mask.sum()}")

# Visualize confidence
adata.obs['prediction_confidence'] = max_probs
sc.pl.umap(adata, color='prediction_confidence', cmap='viridis')
```

## Performance Tips

### 1. Preprocessing Recommendations

```python
import scanpy as sc

# Standard preprocessing pipeline
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ScDiVa works best with normalized, log-transformed data
```

### 2. Memory Optimization

```python
# Use sparse matrices when possible
adata.X = scipy.sparse.csr_matrix(adata.X)

# Process in smaller batches
engine = ScDiVaInference(model_name="base-pretrain")
embeddings = engine.get_embeddings(adata, batch_size=256)
```

### 3. Speed Optimization

```python
# Use mixed precision on GPU
engine = ScDiVaInference(
    model_name="base-pretrain",
    device="cuda",
    use_gpu=True
)

# Increase batch size (if memory allows)
results = engine.annotate(adata, batch_size=512)
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use CPU inference
```python
engine = ScDiVaInference(model_name="base-pretrain", use_gpu=False)
results = engine.annotate(adata, batch_size=128)
```

### Issue: Poor Annotation Accuracy

**Solution**: Ensure proper preprocessing
```python
# Check if data is normalized
print(f"Mean expression: {adata.X.mean()}")
print(f"Std expression: {adata.X.std()}")

# If not normalized, apply standard pipeline
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
```

### Issue: Model Download Fails

**Solution**: Manual download
```bash
# Download from HuggingFace
wget https://huggingface.co/ScDiVa/base-pretrain/resolve/main/pytorch_model.bin

# Or use ModelScope mirror
wget https://modelscope.cn/ScDiVa/base-pretrain/resolve/main/pytorch_model.bin
```

## API Reference

See full API documentation at: [https://scdiva.readthedocs.io](https://scdiva.readthedocs.io)

## Support

For issues and questions:
- 📧 Email: contact@scdiva.ai
- 💬 GitHub Issues: [https://github.com/your-org/ScDiVa/issues](https://github.com/your-org/ScDiVa/issues)
