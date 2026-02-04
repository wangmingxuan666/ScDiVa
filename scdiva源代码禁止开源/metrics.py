# scdiva/metrics.py
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def _iter_tensors(obj, out_list):
    """递归收集 obj 内部所有 torch.Tensor（兼容 tuple/list/dict/ModelOutput）"""
    if obj is None:
        return

    if torch.is_tensor(obj):
        out_list.append(obj)
        return

    # HF ModelOutput / dataclass-like 常见接口
    if hasattr(obj, "to_tuple") and callable(getattr(obj, "to_tuple")):
        try:
            tup = obj.to_tuple()
            _iter_tensors(tup, out_list)
            return
        except Exception:
            pass

    if isinstance(obj, dict):
        for v in obj.values():
            _iter_tensors(v, out_list)
        return

    if isinstance(obj, (tuple, list)):
        for it in obj:
            _iter_tensors(it, out_list)
        return

    # 其他类型忽略


def _is_mask_tensor(t: torch.Tensor) -> bool:
    """判断是否像 masked_indices：(B,L) 且 bool 或 0/1 整数"""
    if not torch.is_tensor(t) or t.dim() != 2:
        return False
    if t.dtype == torch.bool:
        return True
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        # 常见 mask 是 0/1
        # 避免全量 unique 太贵：抽样判断
        sample = t.flatten()
        if sample.numel() > 4096:
            sample = sample[torch.randint(0, sample.numel(), (4096,), device=sample.device)]
        uniq = torch.unique(sample)
        return uniq.numel() <= 4 and torch.all((uniq == 0) | (uniq == 1))
    return False


def preprocess_logits_for_metrics(logits, labels):
    """
    ✅ 终极鲁棒版：不假设 logits 的 tuple 结构
    - 收集所有 tensor
    - 自动识别 gene_logits / pred_values / masked_indices
    - DDP-safe：不在这里做 mask 过滤
    """
    tensors = []
    _iter_tensors(logits, tensors)

    if len(tensors) == 0:
        raise TypeError(f"preprocess_logits_for_metrics: no tensor found in logits, type={type(logits)}")

    # 1) gene_logits: 3D (B,L,V)，一般 V 最大
    gene_logits_candidates = [t for t in tensors if t.dim() == 3 and t.is_floating_point()]
    if len(gene_logits_candidates) == 0:
        # 有些版本 gene_logits 可能是 bf16/float，都算 floating；若都没找到，报更清晰
        shapes = [(tuple(t.shape), str(t.dtype)) for t in tensors]
        raise TypeError(f"Cannot find 3D floating gene_logits in logits tensors: {shapes}")

    gene_logits = max(gene_logits_candidates, key=lambda x: x.shape[-1])  # V 最大的那个

    # 2) pred_values: 2D float (B,L)，且形状要和 gene_logits 的 (B,L) 匹配
    BL = gene_logits.shape[:2]
    pred_candidates = [
        t for t in tensors
        if t.dim() == 2 and t.is_floating_point() and tuple(t.shape) == tuple(BL)
    ]
    if len(pred_candidates) == 0:
        shapes = [(tuple(t.shape), str(t.dtype)) for t in tensors]
        raise TypeError(f"Cannot find pred_values (B,L) float tensor. Expected BL={BL}, got tensors={shapes}")
    # 如果有多个，取一个（一般只有一个）
    pred_values = pred_candidates[0]

    # 3) masked_indices: 2D mask (B,L)，形状同 BL
    mask_candidates = [
        t for t in tensors
        if t.dim() == 2 and tuple(t.shape) == tuple(BL) and _is_mask_tensor(t)
    ]
    if len(mask_candidates) == 0:
        shapes = [(tuple(t.shape), str(t.dtype)) for t in tensors]
        raise TypeError(f"Cannot find masked_indices (B,L) mask tensor. Expected BL={BL}, got tensors={shapes}")
    masked_indices = mask_candidates[0]

    # gene_preds: (B,L)
    gene_preds = torch.argmax(gene_logits, dim=-1)
    return (gene_preds, pred_values, masked_indices)


def compute_metrics(eval_preds):
    preds_tuple = eval_preds.predictions
    labels_tuple = eval_preds.label_ids

    m_gene_preds_full = preds_tuple[0]
    m_val_preds_full = preds_tuple[1]
    masked_indices_full = preds_tuple[2]

    target_values_full = labels_tuple[0]
    gene_ids_full = labels_tuple[1]

    mask_flat = masked_indices_full.flatten().astype(bool)
    if not mask_flat.any():
        return {"accuracy": 0.0, "mse": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}

    final_gene_preds = m_gene_preds_full.flatten()[mask_flat]
    final_gene_targets = gene_ids_full.flatten()[mask_flat]
    final_val_preds = m_val_preds_full.flatten()[mask_flat]
    final_val_targets = target_values_full.flatten()[mask_flat]

    acc = float((final_gene_preds == final_gene_targets).mean())
    mse = float(((final_val_preds - final_val_targets) ** 2).mean())
    rmse = float(np.sqrt(mse))

    pearson_val = 0.0
    spearman_val = 0.0
    if len(final_val_preds) > 10:
        try:
            pv, _ = pearsonr(final_val_preds, final_val_targets)
            pearson_val = float(pv) if not np.isnan(pv) else 0.0
            sv, _ = spearmanr(final_val_preds, final_val_targets)
            spearman_val = float(sv) if not np.isnan(sv) else 0.0
        except Exception:
            pass

    return {
        "accuracy": acc,
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson_val,
        "spearman": spearman_val,
    }
