import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse, csr_matrix
from tqdm import tqdm, trange

# ==========================================
# 1. NLP Metrics (依赖检测与安全实现)
# ==========================================

# 尝试导入快速库，如果没有则使用 Python 实现
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

def levenshtein_distance(seq1_list, seq2_list):
    """
    计算两组序列之间的平均 Levenshtein 编辑距离。
    支持输入为 Token ID 列表 (List[List[int]]) 或 Numpy 数组。
    """
    scores = []
    
    # 简单的 Python 实现 fallback (用于未安装 python-Levenshtein 库的情况)
    def _levenshtein_fallback(s1, s2):
        if len(s1) < len(s2):
            return _levenshtein_fallback(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    for s1, s2 in zip(seq1_list, seq2_list):
        # 转换为 list
        t1 = s1.tolist() if isinstance(s1, np.ndarray) else list(s1)
        t2 = s2.tolist() if isinstance(s2, np.ndarray) else list(s2)
        
        if HAS_LEVENSHTEIN:
            # 库通常处理字符串，我们将 token ID 映射为 unicode 字符以利用库的 C++ 速度
            # 注意：这只是为了计算差异，字符本身没有意义
            str1 = "".join([chr(x % 65535) for x in t1]) 
            str2 = "".join([chr(x % 65535) for x in t2])
            scores.append(Levenshtein.distance(str1, str2))
        else:
            scores.append(_levenshtein_fallback(t1, t2))
            
    return np.mean(scores)

def bleu_score(references, hypotheses, n_gram=4):
    """
    计算平均 BLEU 分数。
    """
    if not HAS_NLTK:
        print("Warning: NLTK not installed. Returning 0.0 for BLEU score.")
        return 0.0

    scores = []
    smoothing = SmoothingFunction().method1
    weights = tuple([1. / n_gram] * n_gram)
    
    for ref, hyp in zip(references, hypotheses):
        # NLTK 期望 references 是 [ref1, ref2...] 的列表
        r = ref.tolist() if isinstance(ref, np.ndarray) else list(ref)
        h = hyp.tolist() if isinstance(hyp, np.ndarray) else list(hyp)
        
        # 过滤掉 padding (假设 0 或负数为 padding，或者不处理由调用者保证)
        # 这里直接计算
        try:
            score = sentence_bleu([r], h, weights=weights, smoothing_function=smoothing)
        except Exception:
            score = 0.0
        scores.append(score)
        
    return np.mean(scores)

# ==========================================
# 2. Data Processing Utils
# ==========================================

def permute_genes_by_expression(adata, tokenizer, symbol2id=None, dataset_name="default"):
    """
    根据表达量对基因进行排序，并转换为 Token ID。
    """
    mapped_gene_ids = []
    
    # 1. 基因名映射逻辑
    for gene_name in adata.var.index:
        final_name = gene_name
        # 如果有 symbol2id 映射表
        if symbol2id and gene_name in symbol2id:
            final_name = symbol2id[gene_name]
        
        # 获取 Token ID
        # 兼容不同的 Tokenizer 接口 (HuggingFace 或 简单 Dict)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            tid = tokenizer.convert_tokens_to_ids(final_name)
        elif hasattr(tokenizer, "encode"):
            tid = tokenizer.encode(final_name)[0]
        else:
            # 简单的字典回退
            tid = tokenizer.vocab.get(final_name, 0) # 0 as unk
            
        mapped_gene_ids.append(tid)
        
    gene_ids_vec = np.array(mapped_gene_ids)
    permuted_gene_ids = []
    
    # 2. 对每个细胞按表达量排序
    # 检测稀疏矩阵
    is_sparse = issparse(adata.X)
    X = adata.X
    
    for i in trange(X.shape[0], desc="Permuting genes"):
        if is_sparse:
            row_data = X[i].toarray().flatten()
        else:
            row_data = X[i].flatten()
            
        # argsort 从小到大，取反变为从大到小
        # 这一步获得的是“原本在 gene_ids_vec 中的索引”
        sorted_indices = np.argsort(-row_data)
        
        # 根据索引取 Token ID
        permuted_gene_ids.append(gene_ids_vec[sorted_indices])
        
    return np.array(permuted_gene_ids)

def load_g2v(file_path):
    """加载 Gene2Vec 嵌入文件"""
    g2v = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            # 假设第一列是基因名，后面是向量
            if len(line) > 1:
                g2v[line[0]] = np.array([float(i) for i in line[1:]])
    return g2v

# ==========================================
# 3. Statistical Metrics
# ==========================================

def pearson_correlation(x, y, abs_val=False):
    """计算 Pearson 相关系数，支持 Tensor"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
        
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    corr, _ = pearsonr(x_flat, y_flat)
    
    if abs_val:
        return abs(corr)
    return corr

def kl_divergence(P, Q):
    """Kullback-Leibler 散度"""
    # 避免除以 0 和 log(0)
    epsilon = 1e-10
    P = np.asarray(P, dtype=np.float64) + epsilon
    Q = np.asarray(Q, dtype=np.float64) + epsilon
    
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    return np.sum(P * np.log(P / Q))

def jensen_shannon_divergence(P, Q):
    """Jensen-Shannon 散度"""
    P = np.asarray(P)
    Q = np.asarray(Q)
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

# ==========================================
# 4. Misc / Dataset
# ==========================================

class CellDataset(Dataset):
    """
    基础 Dataset 类 (适配 List of Dicts)
    """
    def __init__(self, cells, tokenizer=None):
        self.cells = cells
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        item = self.cells[idx]
        # 确保去掉多余的 batch 维度 (1, L) -> (L,)
        processed_item = {}
        for key, val in item.items():
            if isinstance(val, torch.Tensor):
                processed_item[key] = val.squeeze(0) if val.dim() > 1 and val.size(0) == 1 else val
            else:
                processed_item[key] = val
        return processed_item

def get_last_checkpoint(output_dir):
    """获取最新检查点路径"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for root, dirs, files in os.walk(output_dir):
        for d in dirs:
            if d.startswith("checkpoint"):
                checkpoints.append(os.path.join(root, d))
                
    if not checkpoints:
        return None
        
    # 尝试按数字排序
    try:
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    except:
        pass # 如果格式不对就按字母序
        
    return checkpoints[-1]