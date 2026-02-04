# ScDiVa 快速开始指南

欢迎使用 ScDiVa！本指南将帮助您在 5 分钟内开始使用。

---

## 🚀 30 秒快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/ScDiVa.git
cd ScDiVa

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载权重
huggingface-cli download ScDiVa/base-pretrain --local-dir ./weights/base-pretrain

# 4. 运行示例
python examples/quick_start.py
```

---

## 📊 完整工作流程

### 1️⃣ 环境准备

```bash
# 创建虚拟环境
conda create -n scdiva python=3.8
conda activate scdiva

# 安装 ScDiVa
cd ScDiVa
pip install -r requirements.txt
```

### 2️⃣ 下载资源

**选择一：使用 HuggingFace (国际用户)**
```bash
pip install huggingface_hub
huggingface-cli download ScDiVa/base-pretrain --local-dir ./weights/base-pretrain
```

**选择二：使用 ModelScope (中国用户)**
```python
from modelscope import snapshot_download
snapshot_download('ScDiVa/base-pretrain', cache_dir='./weights/base-pretrain')
```

### 3️⃣ 准备数据

```python
import scanpy as sc

# 加载您的数据
adata = sc.read_h5ad("your_data.h5ad")

# 预处理
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
```

### 4️⃣ 运行推理

```python
from modeling_scdiva import ScDiVaModel
import torch

# 加载模型
model = ScDiVaModel.from_pretrained("./weights/base-pretrain")
model.eval()

# 提取嵌入
with torch.no_grad():
    gene_expression = torch.tensor(adata.X.toarray()).float()
    encoding = model.encode(gene_expression)
    embeddings = encoding['latent'].numpy()

# 添加到 AnnData
adata.obsm['X_scdiva'] = embeddings
```

### 5️⃣ 下游分析

```python
# 使用 ScDiVa 嵌入进行聚类
sc.pp.neighbors(adata, use_rep='X_scdiva')
sc.tl.leiden(adata)
sc.tl.umap(adata)

# 可视化
sc.pl.umap(adata, color='leiden')
```

---

## 🎯 常见任务

### 任务 1: 细胞类型注释

```python
from scdiva_inference import ScDiVaInference

engine = ScDiVaInference(model_name="base-annotation")
cell_types = engine.annotate(adata)
adata.obs['cell_type'] = cell_types
```

### 任务 2: 批次整合

```python
engine = ScDiVaInference(model_name="base-batch-integration")
integrated_adata = engine.integrate_batches([adata1, adata2, adata3])
```

### 任务 3: 多任务分析

```python
engine = ScDiVaInference(model_name="large-multitask")
results = engine.predict_multi_task(adata, tasks=["annotation", "clustering"])
```

---

## 📖 推荐学习路径

### 新手路径 (1-2 小时)
1. ✅ 阅读本快速开始指南
2. ✅ 运行 `examples/quick_start.py`
3. ✅ 查看 `docs/inference.md` 了解详细用法

### 进阶路径 (3-5 小时)
4. ✅ 阅读 `docs/model_architecture.md` 理解架构
5. ✅ 研究 `modeling_scdiva.py` 源码
6. ✅ 查看 `docs/benchmarks.md` 了解性能

### 专家路径 (1-2 天)
7. ✅ 在自己的数据上运行完整分析
8. ✅ 微调模型或开发新功能
9. ✅ 参与社区贡献

---

## ⚡ 性能提示

### 加速推理
```python
# 使用 GPU
engine = ScDiVaInference(model_name="base-pretrain", device="cuda")

# 增加批次大小
results = engine.annotate(adata, batch_size=512)

# 使用 Base 模型（更快）
model = ScDiVaModel.from_pretrained("ScDiVa/base-pretrain")  # 而不是 large
```

### 节省内存
```python
# 使用稀疏矩阵
import scipy.sparse
adata.X = scipy.sparse.csr_matrix(adata.X)

# 减小批次大小
results = engine.annotate(adata, batch_size=128)

# 使用 CPU
engine = ScDiVaInference(model_name="base-pretrain", use_gpu=False)
```

---

## 🆘 遇到问题?

### 常见错误及解决方案

**错误 1: "CUDA out of memory"**
```python
# 解决方案：使用 CPU 或减小批次大小
engine = ScDiVaInference(model_name="base-pretrain", use_gpu=False)
```

**错误 2: "模型加载失败"**
```bash
# 解决方案：检查权重文件是否完整
ls -lh ./weights/base-pretrain/
md5sum ./weights/base-pretrain/pytorch_model.bin
```

**错误 3: "导入模块失败"**
```bash
# 解决方案：确保依赖已安装
pip install -r requirements.txt
```

### 获取帮助
- 📖 查看 [FAQ](docs/faq.md)
- 💬 提交 [GitHub Issue](https://github.com/your-org/ScDiVa/issues)
- 📧 邮件: contact@scdiva.ai

---

## 📚 完整文档

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 项目总览和结果展示 |
| [docs/inference.md](docs/inference.md) | 详细使用教程 |
| [docs/model_architecture.md](docs/model_architecture.md) | 模型架构说明 |
| [docs/benchmarks.md](docs/benchmarks.md) | 性能基准测试 |
| [docs/faq.md](docs/faq.md) | 常见问题解答 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 项目结构说明 |

---

## 🎉 成功案例

完成快速开始后，您应该能够：
- ✅ 加载和使用 ScDiVa 模型
- ✅ 对单细胞数据进行细胞类型注释
- ✅ 整合来自不同批次的数据
- ✅ 提取高质量的细胞嵌入用于下游分析

---

**准备好开始了吗？运行第一个示例！**

```bash
python examples/quick_start.py
```

祝您使用愉快！ 🚀
