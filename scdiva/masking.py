# scdiva/masking.py
import torch
from .config import Config

def mask_gene_id_and_value(gene_ids, attention_mask, eps=1e-6):
    """
    只生成 mask：
    - masked_indices: 计算loss的位置
    - replace_mask:   实际替换输入embedding的位置（90% replace，10% keep）
    ✅ 不 mask 41815/41816/41817
    """
    B, L = gene_ids.shape
    device = gene_ids.device
    cfg = Config()

    t = torch.rand(B, device=device).clamp(min=eps, max=1.0)
    rand_mat = torch.rand(B, L, device=device)

    is_special = (
        (gene_ids == cfg.PAD_FILL_GENE_ID) |
        (gene_ids == cfg.BOS_GENE_ID) |
        (gene_ids == cfg.EOS_GENE_ID)
    )

    masked_indices = (rand_mat < t[:, None]) & (attention_mask == 1) & (~is_special)

    rand_action = torch.rand(B, L, device=device)
    replace_mask = (rand_action < 0.9) & masked_indices

    return masked_indices, replace_mask, t
