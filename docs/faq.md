# ScDiVa 常见问题解答 (FAQ)

## 📋 目录

- [安装与环境](#安装与环境)
- [模型下载与加载](#模型下载与加载)
- [数据准备](#数据准备)
- [推理与使用](#推理与使用)
- [性能优化](#性能优化)
- [错误排查](#错误排查)
- [隐私与开源](#隐私与开源)

---

## 安装与环境

### Q1: ScDiVa 对 Python 版本有什么要求?

**A**: ScDiVa 需要 Python 3.8 或更高版本。推荐使用 Python 3.8-3.10。

```bash
# 检查 Python 版本
python --version

# 推荐使用 conda 创建环境
conda create -n scdiva python=3.8
conda activate scdiva
```

### Q2: 安装依赖时出现错误怎么办?

**A**: 请尝试以下步骤：

```bash
# 更新 pip
pip install --upgrade pip

# 使用国内镜像源（如果在中国）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果仍有问题，逐个安装核心依赖
pip install torch numpy pandas scanpy
```

### Q3: 是否需要 GPU?

**A**: 不是必需的，但强烈推荐：
- **CPU 推理**: 可以运行，但速度较慢（~1,000 cells/s）
- **GPU 推理**: 速度快 10-20 倍（~10,000 cells/s on A100）

支持的 GPU: NVIDIA GPU with CUDA 11.0+

---

## 模型下载与加载

### Q4: 模型权重文件在哪里下载?

**A**: 有三种方式：

1. **HuggingFace** (国际用户推荐)
   ```bash
   huggingface-cli download ScDiVa/base-pretrain --local-dir ./weights/base-pretrain
   ```

2. **ModelScope** (中国用户推荐)
   ```python
   from modelscope import snapshot_download
   snapshot_download('ScDiVa/base-pretrain', cache_dir='./weights')
   ```

3. **直接链接**: 查看 `weights/README.md`

### Q5: 下载速度很慢怎么办?

**A**: 
- 中国用户请使用 ModelScope
- 国际用户可以使用 HuggingFace 镜像站
- 支持断点续传，可以多次尝试

### Q6: 如何选择 Base 还是 Large 模型?

**A**: 根据您的资源和需求选择：

| 指标 | Base | Large |
|------|------|-------|
| 推理速度 | 快 (10K cells/s) | 慢 (4K cells/s) |
| 内存需求 | 低 (~4 GB) | 高 (~12 GB) |
| 准确率 | 94.2% | 95.8% |
| **推荐场景** | 快速分析、资源受限 | 高精度要求、充足资源 |

### Q7: 加载模型时出现 "权重文件不匹配" 错误?

**A**: 可能原因：
1. 权重文件下载不完整 → 重新下载
2. 模型版本不匹配 → 确保使用最新版本的代码
3. 文件损坏 → 验证文件 MD5

```bash
# 验证文件完整性
md5sum ./weights/base-pretrain/pytorch_model.bin
```

---

## 数据准备

### Q8: ScDiVa 支持哪些输入格式?

**A**: 支持以下格式：
- **AnnData (.h5ad)** - 推荐
- **Loom (.loom)**
- **CSV/TSV** - 基因表达矩阵
- **NumPy array** - 直接从内存

```python
# 加载不同格式
import scanpy as sc

# h5ad
adata = sc.read_h5ad("data.h5ad")

# loom
adata = sc.read_loom("data.loom")

# csv
adata = sc.read_csv("data.csv").T  # 转置: genes × cells → cells × genes
```

### Q9: 数据需要预处理吗?

**A**: 是的，ScDiVa 期望输入经过以下预处理：

```python
import scanpy as sc

# 标准预处理流程
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化
sc.pp.log1p(adata)  # log 转换

# 现在可以输入到 ScDiVa
```

### Q10: 如果我的数据基因数量超过 20,000 怎么办?

**A**: 有两种方案：

1. **使用 Large 模型** (支持 30,000 基因)
2. **特征选择** (推荐)
   ```python
   # 选择高变基因
   sc.pp.highly_variable_genes(adata, n_top_genes=20000)
   adata = adata[:, adata.var.highly_variable]
   ```

---

## 推理与使用

### Q11: 为什么没有训练代码?

**A**: 出于以下原因，训练代码未开源：
- 涉及专有数据处理流程
- 包含敏感的训练细节
- 预训练需要大量计算资源 (数百个 GPU-days)

但我们提供：
- ✅ 完整的模型架构定义
- ✅ 预训练和微调权重
- ✅ 推理和使用接口
- ✅ 下游任务数据集

### Q12: 如何进行细胞类型注释?

**A**: 使用简化的推理接口：

```python
from scdiva_inference import ScDiVaInference
import scanpy as sc

# 加载数据
adata = sc.read_h5ad("your_data.h5ad")

# 初始化引擎
engine = ScDiVaInference(model_name="base-annotation")

# 进行注释
cell_types = engine.annotate(adata)

# 添加到数据中
adata.obs['predicted_cell_type'] = cell_types
```

### Q13: 如何进行批次整合?

**A**: 

```python
from scdiva_inference import ScDiVaInference

# 加载多个批次
adata1 = sc.read_h5ad("batch1.h5ad")
adata2 = sc.read_h5ad("batch2.h5ad")

# 初始化引擎
engine = ScDiVaInference(model_name="base-batch-integration")

# 整合批次
integrated = engine.integrate_batches([adata1, adata2])
```

### Q14: 推理 SDK 什么时候发布?

**A**: 完整的推理 SDK (`scdiva_inference.py`) 将在论文接收后发布。

**早期访问**: 如果您需要提前使用，请联系 contact@scdiva.ai

---

## 性能优化

### Q15: 如何加速推理?

**A**: 几种优化策略：

1. **使用 GPU**
   ```python
   engine = ScDiVaInference(model_name="base-pretrain", device="cuda")
   ```

2. **增加批次大小** (如果内存允许)
   ```python
   results = engine.annotate(adata, batch_size=512)
   ```

3. **使用混合精度**
   ```python
   # 在模型加载时启用 fp16
   model = model.half()  # 使用半精度
   ```

4. **使用 Base 模型** (速度提升 2.5x)

### Q16: 推理时内存不足怎么办?

**A**: 尝试以下方法：

1. **减小批次大小**
   ```python
   engine.annotate(adata, batch_size=128)
   ```

2. **使用 CPU**
   ```python
   engine = ScDiVaInference(model_name="base-pretrain", use_gpu=False)
   ```

3. **分批处理大数据集**
   ```python
   # 将数据分成小块
   batch_size = 10000
   for i in range(0, adata.n_obs, batch_size):
       batch_adata = adata[i:i+batch_size]
       results = engine.annotate(batch_adata)
   ```

### Q17: 在多个 GPU 上并行推理?

**A**: 

```python
import torch

# 指定 GPU
model = ScDiVaModel.from_pretrained("ScDiVa/base-pretrain")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

---

## 错误排查

### Q18: 出现 "CUDA out of memory" 错误?

**A**: 
1. 减小批次大小
2. 使用更小的模型 (Base 而不是 Large)
3. 清理 GPU 缓存: `torch.cuda.empty_cache()`
4. 使用 CPU 推理

### Q19: 注释结果准确率低?

**A**: 检查以下内容：

1. **数据预处理**
   ```python
   # 确保数据已归一化和log转换
   print(f"Mean: {adata.X.mean():.2f}")  # 应该在 0-5 范围
   print(f"Std: {adata.X.std():.2f}")
   ```

2. **使用正确的模型**
   - 使用 `base-annotation` 或 `large-multitask`

3. **数据质量**
   - 检查细胞和基因过滤
   - 确保数据不是原始计数

### Q20: 模块导入失败?

**A**: 

```python
# 确保在正确的目录
import sys
sys.path.append('/path/to/ScDiVa')

# 验证安装
import modeling_scdiva
print(modeling_scdiva.__file__)
```

---

## 隐私与开源

### Q21: 为什么不开源训练代码?

**A**: 主要原因：
1. **数据隐私**: 训练使用了部分私有数据
2. **商业考虑**: 保护知识产权
3. **计算成本**: 预训练需要极大计算资源，普通用户无法复现

我们仍然开源了：
- 模型架构完整定义
- 所有预训练权重
- 推理和使用接口
- 基准测试数据集

### Q22: 可以商业使用吗?

**A**: 可以！ScDiVa 使用 **Apache 2.0** 许可证：
- ✅ 允许商业使用
- ✅ 允许修改和二次开发
- ✅ 允许分发
- ⚠️ 需要保留版权声明

详情请查看 `LICENSE` 文件。

### Q23: 可以基于 ScDiVa 发表论文吗?

**A**: 当然可以！如果使用了 ScDiVa，请引用：

```bibtex
@article{scdiva2026,
  title={ScDiVa: A Foundation Model for Single-cell Genomics},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

### Q24: 如何贡献代码?

**A**: 我们欢迎社区贡献！

1. Fork 仓库
2. 创建分支
3. 提交 Pull Request

贡献类型：
- 文档改进
- Bug 修复
- 新功能（推理相关）
- 示例代码

---

## 更多问题?

- 📧 Email: contact@scdiva.ai
- 💬 GitHub Issues: [提交问题](https://github.com/your-org/ScDiVa/issues)
- 📖 文档: [完整文档](../README.md)

---

**最后更新**: 2026-02-03
