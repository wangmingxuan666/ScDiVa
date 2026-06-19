"""
ScDiVa: Masked Discrete Diffusion for Joint Modeling of Single-Cell Identity and Expression
Official model architecture. This file matches the weights in model.safetensors.
"""

import math
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import ModelOutput as HFModelOutput

# =============================================================================
# Config
# =============================================================================

class Config:
    SCGPT_DIM = 512
    HIDDEN_DIM = 512
    N_LAYERS = 12
    N_HEADS = 8
    DROPOUT = 0.1
    D_HID = 2048
    VOCAB_SIZE = 41818
    PAD_TOKEN_ID = 0
    MASK_TOKEN_ID = 1
    PAD_FILL_GENE_ID = 41815
    BOS_GENE_ID = 41816
    EOS_GENE_ID = 41817
    MAX_GENE_LEN = 1200
    ROPE_THETA = 10000.0
    ROPE_MAX_LEN = 41819

    @classmethod
    def from_dict(cls, d):
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# =============================================================================
# Blocks (RMSNorm, SwiGLU, RoPE, Attention)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos[:seq_len, :], self.sin[:seq_len, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RoPESDPAAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, max_len: int, rope_theta: float):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_len, base=rope_theta)

    def forward(self, x, attn_mask_4d=None):
        B, L, H = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        return self.o_proj(out)


class ScDiVaBlock(nn.Module):
    def __init__(self, d_model, nhead, d_hid, dropout, max_len, rope_theta):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPESDPAAttention(d_model, nhead, dropout, max_len=max_len, rope_theta=rope_theta)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask_4d=None):
        h = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask_4d=attn_mask_4d)
        x = h + self.drop(x)
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + self.drop(x)
        return x


# =============================================================================
# Masking
# =============================================================================

def mask_gene_id_and_value(gene_ids, attention_mask, cfg=None):
    if cfg is None:
        cfg = Config()
    B, L = gene_ids.shape
    device = gene_ids.device
    t = torch.rand(B, device=device).clamp(min=1e-6, max=1.0)
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


# =============================================================================
# Loss
# =============================================================================

def scdiva_dual_loss(gene_logits, gene_ids, pred_values, target_values, masked_indices, mse_weight=10.0):
    if masked_indices is None or not masked_indices.any():
        return None, None, None
    loss_mse = F.mse_loss(pred_values[masked_indices], target_values[masked_indices])
    loss_ce = F.cross_entropy(gene_logits[masked_indices], gene_ids[masked_indices])
    loss = loss_ce + mse_weight * loss_mse
    return loss, loss_ce, loss_mse


# =============================================================================
# LatentEncoder
# =============================================================================

class LatentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        return self.mlp(mean_embedding)


# =============================================================================
# Main Model
# =============================================================================

@dataclass
class ScDiVaOutput(HFModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    gene_logits: torch.Tensor = None
    pred_values: torch.Tensor = None
    masked_indices: torch.Tensor = None
    loss_ce: Optional[torch.Tensor] = None
    loss_mse: Optional[torch.Tensor] = None


class ScDiVaModel(nn.Module):
    """
    ScDiVa: Masked Discrete Diffusion for Single-Cell Modeling.
    Architecture matches the pre-trained weights in model.safetensors.
    """
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = Config()
        self.cfg = cfg

        self.gene_embedding = nn.Embedding(cfg.VOCAB_SIZE, cfg.SCGPT_DIM, padding_idx=cfg.PAD_TOKEN_ID)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, cfg.SCGPT_DIM),
            nn.ReLU(),
            nn.Linear(cfg.SCGPT_DIM, cfg.SCGPT_DIM),
        )
        self.value_mask_emb = nn.Parameter(torch.zeros(cfg.SCGPT_DIM))
        self.gene_mask_emb = nn.Parameter(torch.zeros(cfg.SCGPT_DIM))
        self.input_proj = nn.Linear(cfg.SCGPT_DIM, cfg.HIDDEN_DIM, bias=False)
        self.latent_encoder = LatentEncoder(cfg.SCGPT_DIM, cfg.HIDDEN_DIM)
        self.layers = nn.ModuleList([
            ScDiVaBlock(cfg.HIDDEN_DIM, cfg.N_HEADS, cfg.D_HID, cfg.DROPOUT, cfg.ROPE_MAX_LEN, cfg.ROPE_THETA)
            for _ in range(cfg.N_LAYERS)
        ])
        self.final_norm = RMSNorm(cfg.HIDDEN_DIM)
        self.gene_head = nn.Linear(cfg.HIDDEN_DIM, cfg.VOCAB_SIZE, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, gene_ids, target_values, attention_mask=None, inference_mask=None, **kwargs):
        B, L = gene_ids.shape
        cfg = self.cfg
        device = gene_ids.device

        if attention_mask is None:
            attention_mask = (gene_ids != cfg.PAD_TOKEN_ID).long()
            attention_mask = attention_mask * (gene_ids != cfg.PAD_FILL_GENE_ID).long()

        # Embeddings
        g_emb = self.gene_embedding(gene_ids)
        v_emb = self.value_encoder(target_values.unsqueeze(-1).to(g_emb.dtype))
        x_clean = g_emb + v_emb

        # Latent token
        z_latent = self.latent_encoder(x_clean, attention_mask)
        z_token = z_latent.unsqueeze(1)

        # Masking
        if inference_mask is not None:
            replace_mask = inference_mask.to(device)
            masked_indices = replace_mask
            mask = replace_mask.unsqueeze(-1).to(g_emb.dtype)
        else:
            masked_indices, replace_mask, _ = mask_gene_id_and_value(gene_ids, attention_mask, cfg)
            mask = replace_mask.unsqueeze(-1).to(g_emb.dtype)

        gene_mask = self.gene_mask_emb.view(1, 1, -1).to(g_emb)
        value_mask = self.value_mask_emb.view(1, 1, -1).to(v_emb)
        g_emb_masked = g_emb * (1.0 - mask) + gene_mask * mask
        v_emb_masked = v_emb * (1.0 - mask) + value_mask * mask
        x_final = g_emb_masked + v_emb_masked
        x_input = self.input_proj(x_final)
        x_seq = torch.cat([z_token, x_input], dim=1)

        # Attention mask
        latent_mask = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([latent_mask, attention_mask], dim=1)
        min_dtype = torch.finfo(x_seq.dtype).min
        attn_mask_4d = extended_mask[:, None, None, :].to(dtype=x_seq.dtype)
        attn_mask_4d = (1.0 - attn_mask_4d) * min_dtype

        # Transformer
        h = x_seq
        for layer in self.layers:
            h = layer(h, attn_mask_4d=attn_mask_4d)
        h = self.final_norm(h)
        gene_output = h[:, 1:, :]

        gene_logits = self.gene_head(gene_output)
        pred_values = self.value_head(gene_output).squeeze(-1)

        loss = loss_ce = loss_mse = None
        if masked_indices.any():
            loss, loss_ce, loss_mse = scdiva_dual_loss(
                gene_logits, gene_ids, pred_values, target_values, masked_indices, mse_weight=10.0,
            )

        return ScDiVaOutput(
            loss=loss,
            logits=(gene_logits, pred_values, masked_indices),
            gene_logits=gene_logits,
            pred_values=pred_values,
            masked_indices=masked_indices,
            loss_ce=loss_ce,
            loss_mse=loss_mse,
        )

    def encode(self, gene_expression):
        """Encode gene expression into cell embeddings. Simplified inference API."""
        # Pad/truncate to vocab size (41818)
        B = gene_expression.shape[0]
        device = gene_expression.device
        pad_len = self.cfg.VOCAB_SIZE - gene_expression.shape[1]
        if pad_len > 0:
            padded = F.pad(gene_expression, (0, pad_len))
        else:
            padded = gene_expression[:, :self.cfg.VOCAB_SIZE]

        # Build dummy gene IDs (0..N-1) and values
        gene_ids = torch.arange(self.cfg.VOCAB_SIZE, device=device).unsqueeze(0).expand(B, -1).clone()
        # Mask special tokens
        attention_mask = torch.ones_like(gene_ids)
        attention_mask[:, :self.cfg.MAX_GENE_LEN] = 1
        attention_mask[:, self.cfg.MAX_GENE_LEN:] = 0

        # Run full model with high mask ratio to get latent
        inference_mask = torch.ones_like(gene_ids, dtype=torch.bool)
        inference_mask[:, :1] = False  # Keep first gene visible

        with torch.no_grad():
            outputs = self.forward(gene_ids, padded, attention_mask=attention_mask, inference_mask=inference_mask)
            # Get latent from encoder output
            h = self._get_hidden(gene_ids, padded, attention_mask)
            latent = h[:, 0, :]  # [LAT] token
        return {"latent": latent, "mu": latent, "logvar": torch.zeros_like(latent)}

    def _get_hidden(self, gene_ids, target_values, attention_mask=None):
        B, L = gene_ids.shape
        cfg = self.cfg
        device = gene_ids.device
        if attention_mask is None:
            attention_mask = (gene_ids != cfg.PAD_TOKEN_ID).long()
        g_emb = self.gene_embedding(gene_ids)
        v_emb = self.value_encoder(target_values.unsqueeze(-1).to(g_emb.dtype))
        x_clean = g_emb + v_emb
        z_latent = self.latent_encoder(x_clean, attention_mask)
        z_token = z_latent.unsqueeze(1)
        inference_mask = torch.ones_like(gene_ids, dtype=torch.bool)
        inference_mask[:, :1] = False
        mask = inference_mask.unsqueeze(-1).to(g_emb.dtype)
        gene_mask = self.gene_mask_emb.view(1, 1, -1)
        value_mask = self.value_mask_emb.view(1, 1, -1)
        g_emb_masked = g_emb * (1.0 - mask) + gene_mask.to(g_emb.device) * mask
        v_emb_masked = v_emb * (1.0 - mask) + value_mask.to(v_emb.device) * mask
        x_final = g_emb_masked + v_emb_masked
        x_input = self.input_proj(x_final)
        x_seq = torch.cat([z_token, x_input], dim=1)
        latent_mask = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([latent_mask, attention_mask], dim=1)
        min_dtype = torch.finfo(x_seq.dtype).min
        attn_mask_4d = extended_mask[:, None, None, :].to(dtype=x_seq.dtype)
        attn_mask_4d = (1.0 - attn_mask_4d) * min_dtype
        h = x_seq
        for layer in self.layers:
            h = layer(h, attn_mask_4d=attn_mask_4d)
        h = self.final_norm(h)
        return h

    def predict(self, gene_expression, task="annotation", **kwargs):
        """Inference API for compatibility."""
        encoding = self.encode(gene_expression)
        return encoding["latent"]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        map_location: Optional[str] = None,
        strict: bool = True,
        use_auth_token: Optional[str] = None,
    ) -> "ScDiVaModel":
        """
        Load ScDiVa from local path or Hugging Face Hub.
        Supports both .safetensors and .bin / .pt / .pth formats.
        """
        if map_location is None:
            map_location = "cpu"

        # Try loading config
        cfg = Config()
        try:
            config_path = None
            if os.path.isdir(model_name_or_path):
                config_path = os.path.join(model_name_or_path, "config.json")
            elif os.path.isfile(model_name_or_path):
                config_path = os.path.join(os.path.dirname(model_name_or_path), "config.json")

            if config_path and os.path.exists(config_path):
                with open(config_path) as f:
                    cfg_dict = json.load(f)
                cfg = Config.from_dict(cfg_dict)
        except Exception as e:
            print(f"[ScDiVa] Warning: config loading failed ({e}), using defaults.")

        model = cls(cfg)

        # Find checkpoint
        ckpt_path = None
        if os.path.exists(model_name_or_path):
            if os.path.isfile(model_name_or_path):
                ckpt_path = model_name_or_path
            elif os.path.isdir(model_name_or_path):
                for name in ["model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"]:
                    p = os.path.join(model_name_or_path, name)
                    if os.path.exists(p):
                        ckpt_path = p
                        break

        # Try HuggingFace Hub
        if ckpt_path is None:
            try:
                from huggingface_hub import hf_hub_download
                print(f"[ScDiVa] Downloading from HF: {model_name_or_path}")
                try:
                    ckpt_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename="model.safetensors",
                        token=use_auth_token,
                    )
                except Exception:
                    ckpt_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename="pytorch_model.bin",
                        token=use_auth_token,
                    )
            except ImportError:
                print("[ScDiVa] huggingface_hub not installed. Cannot download from HF.")
            except Exception as e:
                print(f"[ScDiVa] HF download failed: {e}")

        if ckpt_path is None:
            print("[ScDiVa] ERROR: No weights found. Cannot initialize model.")
            raise FileNotFoundError(
                f"No checkpoint found at {model_name_or_path}. "
                "Please provide a valid path or HF repo ID."
            )

        print(f"[ScDiVa] Loading weights from {ckpt_path}...")
        try:
            # Safetensors format
            if ckpt_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file as safe_load
                    state_dict = safe_load(ckpt_path, device=map_location)
                except ImportError:
                    raise ImportError(
                        "safetensors package is required to load .safetensors files. "
                        "Install it with: pip install safetensors"
                    )
            else:
                # PyTorch format
                state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state_dict = state["state_dict"]
                elif isinstance(state, dict) and "model_state_dict" in state:
                    state_dict = state["model_state_dict"]
                else:
                    state_dict = state

            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            if missing:
                print(f"[ScDiVa] Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"[ScDiVa] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            if not missing and not unexpected:
                print(f"[ScDiVa] ✅ All weights loaded successfully!")

        except Exception as e:
            print(f"[ScDiVa] Error loading weights: {e}")
            raise

        return model
