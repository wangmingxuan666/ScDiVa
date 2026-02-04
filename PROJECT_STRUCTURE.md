# ScDiVa 项目完整结构

## 📂 目录树状图

```
ScDiVa/
├── 📄 README.md                          # 项目主文档（包含使用说明、结果展示）
├── 📄 LICENSE                            # Apache 2.0 开源协议
├── 📄 .gitignore                         # Git 忽略配置
├── 📄 requirements.txt                   # Python 依赖包列表
├── 📄 PROJECT_STRUCTURE.md               # 本文档：项目结构说明
│
├── 🐍 modeling_scdiva.py                 # ⭐ 模型架构定义（开源）
├── 🐍 scdiva_inference.py                # 推理 SDK（占位，待发布）
│
├── 📁 assets/                            # 图片和可视化素材
│   ├── scDiVa.pdf                        # 模型架构图
│   ├── batch_immune.pdf                  # 批次整合结果图
│   ├── Anno.pdf                          # 细胞注释结果图
│   ├── Multi.pdf                         # 多任务/多模态结果图
│   ├── 0.pdf                            # UMAP 可视化
│   ├── 2.pdf                            # 细胞轨迹可视化
│   └── workflow.png                      # 项目工作流程图
│
├── 📁 docs/                              # 详细文档
│   ├── model_architecture.md             # 模型架构详细说明
│   ├── inference.md                      # 推理使用教程
│   ├── benchmarks.md                     # 性能基准测试报告
│   └── faq.md                           # 常见问题解答
│
├── 📁 weights/                           # 模型权重（外部下载）
│   ├── README.md                         # 权重下载说明
│   ├── base-pretrain/                    # ScDiVa-Base 预训练权重
│   │   └── (从 HuggingFace/ModelScope 下载)
│   ├── large-pretrain/                   # ScDiVa-Large 预训练权重
│   │   └── (从 HuggingFace/ModelScope 下载)
│   └── finetuned/                        # 微调权重
│       ├── batch-integration/
│       ├── annotation/
│       └── multitask/
│
├── 📁 datasets/                          # 下游任务数据集（外部下载）
│   ├── README.md                         # 数据集下载说明
│   ├── batch_integration/
│   │   ├── immune_atlas.h5ad
│   │   ├── pbmc.h5ad
│   │   └── covid19.h5ad
│   ├── annotation/
│   │   ├── panglao_train.h5ad
│   │   └── panglao_test.h5ad
│   └── multi_task/
│       └── combined_benchmark.h5ad
│
└── 📁 examples/                          # 示例代码（可选）
    ├── quick_start.ipynb                 # 快速入门 Jupyter Notebook
    ├── batch_integration_demo.py         # 批次整合示例
    └── annotation_demo.py                # 细胞注释示例
```

---

## 🎯 核心文件说明

### 1️⃣ 用户首先查看的文件

| 文件 | 作用 | 重要性 |
|------|------|--------|
| **README.md** | 项目总览、快速开始、结果展示 | ⭐⭐⭐⭐⭐ |
| **modeling_scdiva.py** | 模型架构定义（可供研究和扩展） | ⭐⭐⭐⭐⭐ |
| **assets/** | 论文中的图表、可视化结果 | ⭐⭐⭐⭐ |

### 2️⃣ 深入使用的文件

| 文件 | 作用 | 重要性 |
|------|------|--------|
| **docs/inference.md** | 详细的推理使用教程 | ⭐⭐⭐⭐ |
| **docs/benchmarks.md** | 完整的性能测试报告 | ⭐⭐⭐ |
| **scdiva_inference.py** | 简化版推理接口（待发布） | ⭐⭐⭐⭐ |

### 3️⃣ 下载资源

| 资源 | 来源 | 说明 |
|------|------|------|
| **weights/** | HuggingFace/ModelScope | 预训练和微调权重 |
| **datasets/** | HuggingFace Datasets | 下游任务数据集 |

---

## 🔄 项目工作流程

```
用户访问 GitHub
    ↓
阅读 README.md（查看结果、了解能力）
    ↓
查看 assets/ 中的可视化结果
    ↓
阅读 modeling_scdiva.py（了解模型架构）
    ↓
下载 weights/（从 HuggingFace）
    ↓
使用 scdiva_inference.py（进行推理）
    ↓
参考 docs/inference.md（深入使用）
    ↓
（可选）下载 datasets/ 进行训练/微调
```

---

## 📊 文件类型统计

| 类型 | 数量 | 用途 |
|------|------|------|
| 📄 Markdown 文档 | 7 | 项目说明、教程、文档 |
| 🐍 Python 代码 | 2 | 模型定义、推理接口 |
| 📊 PDF/图片 | 6+ | 结果展示、架构图 |
| 📦 配置文件 | 3 | 依赖、Git、协议 |

---

## ⚠️ 隐私保护说明

### ✅ 已开源的内容
- ✅ 模型架构定义 (`modeling_scdiva.py`)
- ✅ 模型权重（预训练 + 微调）
- ✅ 下游任务数据集
- ✅ 完整的文档和使用教程

### ❌ 未开源的内容
- ❌ 模型训练代码 (`train.py`)
- ❌ 数据预处理流程
- ❌ 词表文件（如适用）
- ❌ 内部实验脚本

### 🔒 隐私保护策略
根据项目隐私限制，我们采用了 **"开放架构 + 开放权重"** 的策略：
- 用户可以使用预训练权重进行推理
- 用户可以研究和扩展模型架构
- 用户可以在下游任务上微调模型
- 但训练细节和私有数据处理逻辑不公开

---

## 🚀 快速开始

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-org/ScDiVa.git
   cd ScDiVa
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **下载权重**
   ```bash
   # 查看 weights/README.md 获取下载链接
   ```

4. **运行推理**
   ```python
   from modeling_scdiva import ScDiVaModel
   model = ScDiVaModel.from_pretrained("ScDiVa/base-pretrain")
   ```

---

## 📞 联系方式

- 📧 Email: contact@scdiva.ai
- 💬 Issues: [GitHub Issues](https://github.com/your-org/ScDiVa/issues)
- 📖 Docs: [完整文档](./docs/)

---

**最后更新**: 2026-02-03
