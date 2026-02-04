# scdiva/modeling_scdiva.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput as HFModelOutput

from .config import Config
from .masking import mask_gene_id_and_value
from .loss import scdiva_dual_loss
from .blocks import ScDiVaBlock, RMSNorm

class ModuleType(str):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"

def init_weights(module: nn.Module, cfg: Config, type_of_module: Optional[str] = None):
    init_std = 0.02
    if isinstance(module, (nn.Linear, nn.Embedding)):
        std = init_std
        if type_of_module == ModuleType.out_module:
            std = init_std / math.sqrt(2.0 * cfg.N_LAYERS)
        elif type_of_module == ModuleType.final_out:
            std = cfg.HIDDEN_DIM ** -0.5

        cutoff = 3
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff * std, b=cutoff * std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

@dataclass
class ScDiVaOutput(HFModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    gene_logits: torch.Tensor = None
    pred_values: torch.Tensor = None
    masked_indices: torch.Tensor = None

    loss_ce: Optional[torch.Tensor] = None
    loss_mse: Optional[torch.Tensor] = None

class LatentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        return self.mlp(mean_embedding)

class ScDiVaModel(nn.Module):
    """
    scDiVa: 从头开始预训练
    - gene_embedding: 随机初始化 & 可训练
    - value_encoder: 随机初始化 & 可训练
    - 12层 block: 选项A（RoPE+SDPA + RMSNorm + SwiGLU）
    - 双任务双loss：CE + 10*MSE
    """
    def __init__(self, cfg: Config, init_params: bool = True):
        super().__init__()
        self.cfg = cfg

        # 从头开始：embedding/value_encoder 都不加载外部权重
        self.gene_embedding = nn.Embedding(cfg.VOCAB_SIZE, cfg.SCGPT_DIM, padding_idx=cfg.PAD_TOKEN_ID)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, cfg.SCGPT_DIM),
            nn.ReLU(),
            nn.Linear(cfg.SCGPT_DIM, cfg.SCGPT_DIM),
        )

        # mask embedding
        self.value_mask_emb = nn.Parameter(torch.zeros(cfg.SCGPT_DIM))
        self.gene_mask_emb = nn.Parameter(torch.zeros(cfg.SCGPT_DIM))

        self.input_proj = nn.Linear(cfg.SCGPT_DIM, cfg.HIDDEN_DIM, bias=False)

        self.latent_encoder = LatentEncoder(cfg.SCGPT_DIM, cfg.HIDDEN_DIM)

        self.layers = nn.ModuleList([
            ScDiVaBlock(
                d_model=cfg.HIDDEN_DIM,
                nhead=cfg.N_HEADS,
                d_hid=cfg.D_HID,
                dropout=cfg.DROPOUT,
                max_len=cfg.ROPE_MAX_LEN,
                rope_theta=cfg.ROPE_THETA,
            )
            for _ in range(cfg.N_LAYERS)
        ])
        self.final_norm = RMSNorm(cfg.HIDDEN_DIM)

        self.gene_head = nn.Linear(cfg.HIDDEN_DIM, cfg.VOCAB_SIZE, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        if init_params:
            self.reset_parameters()

    def reset_parameters(self):
        cfg = self.cfg
        init_weights(self.gene_embedding, cfg, type_of_module=ModuleType.emb)
        init_weights(self.input_proj, cfg, type_of_module=ModuleType.in_module)
        init_weights(self.gene_head, cfg, type_of_module=ModuleType.emb)

        for m in list(self.value_encoder.modules()) + list(self.latent_encoder.modules()) + list(self.value_head.modules()):
            if isinstance(m, nn.Linear):
                init_weights(m, cfg, type_of_module=ModuleType.in_module)

        for layer in self.layers:
            init_weights(layer.attn.q_proj, cfg, type_of_module=ModuleType.in_module)
            init_weights(layer.attn.k_proj, cfg, type_of_module=ModuleType.in_module)
            init_weights(layer.attn.v_proj, cfg, type_of_module=ModuleType.in_module)
            init_weights(layer.attn.o_proj, cfg, type_of_module=ModuleType.out_module)

            init_weights(layer.mlp.gate_proj, cfg, type_of_module=ModuleType.in_module)
            init_weights(layer.mlp.up_proj, cfg, type_of_module=ModuleType.in_module)
            init_weights(layer.mlp.down_proj, cfg, type_of_module=ModuleType.out_module)

    # ✅ 修复：加入 inference_mask 参数
    def forward(self, gene_ids, target_values, attention_mask=None, inference_mask=None, **kwargs):
        B, L = gene_ids.shape
        cfg = self.cfg
        device = gene_ids.device

        if gene_ids.max() >= cfg.VOCAB_SIZE:
            raise ValueError(f"Gene ID out of vocabulary: max={int(gene_ids.max())}, vocab={cfg.VOCAB_SIZE}")

        if attention_mask is None:
            attention_mask = (gene_ids != cfg.PAD_TOKEN_ID).long()
            attention_mask = attention_mask * (gene_ids != cfg.PAD_FILL_GENE_ID).long()

        # clean embedding
        g_emb = self.gene_embedding(gene_ids)
        v_emb = self.value_encoder(target_values.unsqueeze(-1).to(g_emb.dtype))
        x_clean = g_emb + v_emb

        # latent token
        z_latent = self.latent_encoder(x_clean, attention_mask)
        z_token = z_latent.unsqueeze(1)

        # 🔥🔥🔥 MASKING LOGIC 🔥🔥🔥
        if inference_mask is not None:
            # 【推理模式】：严格执行外部传入的 Mask
            replace_mask = inference_mask.to(device)
            masked_indices = replace_mask 
            mask = replace_mask.unsqueeze(-1).to(g_emb.dtype)
        else:
            # 【训练模式】：内部生成随机 Mask
            masked_indices, replace_mask, _ = mask_gene_id_and_value(gene_ids, attention_mask)
            mask = replace_mask.unsqueeze(-1).to(g_emb.dtype)

        # 应用 Mask
        gene_mask = self.gene_mask_emb.view(1, 1, -1).to(g_emb)
        value_mask = self.value_mask_emb.view(1, 1, -1).to(v_emb)

        g_emb_masked = g_emb * (1.0 - mask) + gene_mask * mask
        v_emb_masked = v_emb * (1.0 - mask) + value_mask * mask
        x_final_512 = g_emb_masked + v_emb_masked

        x_input = self.input_proj(x_final_512)
        x_seq = torch.cat([z_token, x_input], dim=1)  # (B,1+L,H)

        # additive attention mask
        latent_mask = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([latent_mask, attention_mask], dim=1)  # (B,1+L)

        min_dtype = torch.finfo(x_seq.dtype).min
        attn_mask_4d = extended_mask[:, None, None, :].to(dtype=x_seq.dtype)
        attn_mask_4d = (1.0 - attn_mask_4d) * min_dtype  # (B,1,1,1+L)

        h = x_seq
        for layer in self.layers:
            h = layer(h, attn_mask_4d=attn_mask_4d)

        h = self.final_norm(h)
        gene_output = h[:, 1:, :]  # (B,L,H)

        gene_logits = self.gene_head(gene_output)
        pred_values = self.value_head(gene_output).squeeze(-1)

        # 如果有被 Mask 的位置，计算 Loss (为了推理方便)
        loss = None
        loss_ce = None
        loss_mse = None
        
        if masked_indices.any():
            loss, loss_ce, loss_mse = scdiva_dual_loss(
                gene_logits=gene_logits,
                gene_ids=gene_ids,
                pred_values=pred_values,
                target_values=target_values,
                masked_indices=masked_indices,
                mse_weight=10.0,
            )
            
            # Detach for safe keeping
            loss_ce = loss_ce.detach() if loss_ce is not None else None
            loss_mse = loss_mse.detach() if loss_mse is not None else None

        return ScDiVaOutput(
            loss=loss,
            logits=(gene_logits, pred_values, masked_indices),
            gene_logits=gene_logits,
            pred_values=pred_values,
            masked_indices=masked_indices,
            loss_ce=loss_ce,
            loss_mse=loss_mse,
        )