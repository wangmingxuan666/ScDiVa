"""
ScDiVa 快速入门示例

这个脚本展示了如何使用 ScDiVa 进行基本的单细胞分析任务。
"""

import scanpy as sc
import numpy as np
from modeling_scdiva import ScDiVaModel, ScDiVaConfig
import torch

print("=" * 60)
print("ScDiVa 快速入门示例")
print("=" * 60)

# ============================================================================
# 1. 数据加载和预处理
# ============================================================================

print("\n[步骤 1] 加载和预处理数据...")

# 加载示例数据（替换为您自己的数据）
# adata = sc.read_h5ad("your_data.h5ad")

# 使用 scanpy 的内置 PBMC 数据集作为演示
adata = sc.datasets.pbmc3k()

print(f"  - 细胞数: {adata.n_obs}")
print(f"  - 基因数: {adata.n_vars}")

# 标准预处理流程
print("  - 执行质量控制...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

print("  - 归一化和对数转换...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 特征选择（如果基因数超过模型限制）
if adata.n_vars > 20000:
    print("  - 选择高变基因...")
    sc.pp.highly_variable_genes(adata, n_top_genes=20000)
    adata = adata[:, adata.var.highly_variable]

print(f"  ✓ 预处理完成！最终数据: {adata.n_obs} cells × {adata.n_vars} genes")

# ============================================================================
# 2. 加载 ScDiVa 模型
# ============================================================================

print("\n[步骤 2] 加载 ScDiVa 模型...")

# 方式 1: 从 HuggingFace 加载（需要先下载权重）
# model = ScDiVaModel.from_pretrained("ScDiVa/base-pretrain")

# 方式 2: 从本地路径加载
# model = ScDiVaModel.from_pretrained("./weights/base-pretrain")

# 方式 3: 创建模型架构（用于演示）
config = ScDiVaConfig(
    num_genes=adata.n_vars,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16
)
model = ScDiVaModel(config)
print("  ✓ 模型加载完成！")

# 设置为评估模式
model.eval()

# GPU 加速（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"  - 运行设备: {device}")

# ============================================================================
# 3. 提取潜在表示
# ============================================================================

print("\n[步骤 3] 提取细胞潜在表示...")

# 准备输入数据
if hasattr(adata.X, 'toarray'):
    gene_expression = torch.tensor(adata.X.toarray()).float().to(device)
else:
    gene_expression = torch.tensor(adata.X).float().to(device)

# 如果数据太大，分批处理
batch_size = 256
all_embeddings = []

with torch.no_grad():
    for i in range(0, len(gene_expression), batch_size):
        batch = gene_expression[i:i+batch_size]
        encoding = model.encode(batch)
        all_embeddings.append(encoding['latent'].cpu().numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  - 已处理: {i+batch_size}/{len(gene_expression)} cells")

# 合并结果
embeddings = np.vstack(all_embeddings)
print(f"  ✓ 提取完成！嵌入维度: {embeddings.shape}")

# 将嵌入添加到 AnnData
adata.obsm['X_scdiva'] = embeddings

# ============================================================================
# 4. 细胞类型注释
# ============================================================================

print("\n[步骤 4] 细胞类型注释...")

with torch.no_grad():
    all_predictions = []
    
    for i in range(0, len(gene_expression), batch_size):
        batch = gene_expression[i:i+batch_size]
        logits = model.predict(batch, task="annotation")
        predictions = torch.argmax(logits, dim=1)
        all_predictions.append(predictions.cpu().numpy())

# 合并预测结果
predicted_labels = np.hstack(all_predictions)
adata.obs['predicted_cell_type'] = predicted_labels

print(f"  ✓ 注释完成！")
print(f"  - 发现 {len(np.unique(predicted_labels))} 种细胞类型")

# ============================================================================
# 5. 下游分析和可视化
# ============================================================================

print("\n[步骤 5] 下游分析...")

# 使用 ScDiVa 嵌入进行聚类
print("  - 计算邻居图...")
sc.pp.neighbors(adata, use_rep='X_scdiva')

print("  - Leiden 聚类...")
sc.tl.leiden(adata)

print("  - UMAP 降维...")
sc.tl.umap(adata)

# 可视化
print("\n  - 生成可视化...")
sc.pl.umap(adata, color=['leiden', 'predicted_cell_type'], 
           save='_scdiva_results.png')

print("  ✓ 可视化保存为: figures/umap_scdiva_results.png")

# ============================================================================
# 6. 保存结果
# ============================================================================

print("\n[步骤 6] 保存结果...")

# 保存带有 ScDiVa 结果的 AnnData
adata.write_h5ad("scdiva_analysis_results.h5ad")
print("  ✓ 结果已保存: scdiva_analysis_results.h5ad")

# 导出细胞类型注释
adata.obs[['predicted_cell_type', 'leiden']].to_csv("cell_annotations.csv")
print("  ✓ 注释已保存: cell_annotations.csv")

# ============================================================================
# 完成
# ============================================================================

print("\n" + "=" * 60)
print("✓ 分析完成！")
print("=" * 60)
print("\n下一步:")
print("  1. 查看 scdiva_analysis_results.h5ad 中的完整结果")
print("  2. 检查 figures/ 文件夹中的可视化")
print("  3. 根据需要进行进一步分析")
print("\n更多信息:")
print("  - 文档: docs/inference.md")
print("  - 示例: examples/")
print("  - 问题: https://github.com/your-org/ScDiVa/issues")
print()
