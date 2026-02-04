# scdiva/loss.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def scdiva_dual_loss(
    gene_logits: torch.Tensor,      # (B,L,V)
    gene_ids: torch.Tensor,         # (B,L)
    pred_values: torch.Tensor,      # (B,L)
    target_values: torch.Tensor,    # (B,L)
    masked_indices: torch.Tensor,   # (B,L) bool
    mse_weight: float = 10.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    loss_total = loss_ce + mse_weight * loss_mse
    只在 masked_indices 上计算（与你原版一致）
    """
    if masked_indices is None or not masked_indices.any():
        return None, None, None

    loss_mse = F.mse_loss(pred_values[masked_indices], target_values[masked_indices])
    loss_ce = F.cross_entropy(gene_logits[masked_indices], gene_ids[masked_indices])
    loss = loss_ce + mse_weight * loss_mse
    return loss, loss_ce, loss_mse
