"""
ScDiVa: A Foundation Model for Single-cell Genomics
Model Architecture Definition

This file contains the core architecture definition of ScDiVa.
It integrates SwiGLU, RoPE, and RMSNorm as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
import math
import os

class ScDiVaConfig:
    def __init__(
        self,
        num_genes: int = 41818,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1200,
        layer_norm_eps: float = 1e-5,
        latent_dim: int = 128,
        num_cell_types: int = 100,
        use_variational: bool = True,
        rope_theta: float = 10000.0,
        **kwargs
    ):
        self.num_genes = num_genes
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.latent_dim = latent_dim
        self.num_cell_types = num_cell_types
        self.use_variational = use_variational
        self.rope_theta = rope_theta

# =============================================================================
# Core Blocks (Adapted from blocks.py to match Paper)
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
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    # Helper to apply rotation
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    # Reshape cos/sin for broadcasting: [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoPESDPAAttention(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.nhead = config.num_attention_heads
        self.head_dim = config.hidden_size // self.nhead
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta)
        self.dropout = config.attention_probs_dropout_prob

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Use PyTorch's efficient SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, L, config.hidden_size)
        return self.o_proj(out)

class ScDiVaBlock(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = RoPESDPAAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        h = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask=attn_mask)
        x = h + self.drop(x)
        
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + self.drop(x)
        return x

# =============================================================================
# Outer Model Architecture
# =============================================================================

class GeneEmbedding(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.gene_projection = nn.Linear(config.num_genes, config.hidden_size)
        # Updated to RMSNorm to match paper consistency
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        embeddings = self.gene_projection(gene_expression)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerEncoder(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            ScDiVaBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class VariationalLayer(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.mu_projection = nn.Linear(config.hidden_size, config.latent_dim)
        self.logvar_projection = nn.Linear(config.hidden_size, config.latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu_projection(hidden_states)
        logvar = self.logvar_projection(hidden_states)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class AnnotationHead(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.dense = nn.Linear(config.latent_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_cell_types)
        
    def forward(self, latent_representation: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.dense(latent_representation))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits

class BatchIntegrationHead(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.dense = nn.Linear(config.latent_dim, config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.num_genes)
        
    def forward(self, latent_representation: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.dense(latent_representation))
        reconstructed = self.decoder(hidden)
        return reconstructed

class ScDiVaModel(nn.Module):
    """
    ScDiVa: Single-cell Deep Variational Analysis Model
    """
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.config = config
        self.gene_embedding = GeneEmbedding(config)
        self.encoder = TransformerEncoder(config)
        self.variational_layer = VariationalLayer(config)
        self.annotation_head = AnnotationHead(config)
        self.batch_integration_head = BatchIntegrationHead(config)
        
    def encode(self, gene_expression: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embeddings = self.gene_embedding(gene_expression)
        # Add sequence dimension for Transformer [Batch, SeqLen=1, Dim]
        # Note: If input is token sequence, normalization should happen before calling encode
        embeddings = embeddings.unsqueeze(1) 
        
        encoded = self.encoder(embeddings, attention_mask)
        encoded = encoded.squeeze(1)
        z, mu, logvar = self.variational_layer(encoded)
        return {"latent": z, "mu": mu, "logvar": logvar}
    
    def predict(self, gene_expression: torch.Tensor, task: str = "annotation", attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoding = self.encode(gene_expression, attention_mask)
        latent = encoding["latent"]
        if task == "annotation":
            return self.annotation_head(latent)
        elif task == "batch_integration":
            return self.batch_integration_head(latent)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        map_location: Optional[str] = None,
        strict: bool = True,
        use_auth_token: Optional[str] = None,
    ) -> "ScDiVaModel":
        config = ScDiVaConfig()
        model = cls(config)
        if map_location is None:
            map_location = "cpu"
        
        ckpt_path: Optional[str] = None
        
        # 1. Try Local
        if os.path.exists(model_name_or_path):
            if os.path.isfile(model_name_or_path):
                ckpt_path = model_name_or_path
            elif os.path.isdir(model_name_or_path):
                for name in ["pytorch_model.bin", "model.safetensors", "model.pt"]:
                    p = os.path.join(model_name_or_path, name)
                    if os.path.exists(p):
                        ckpt_path = p
                        break
        
        # 2. Try Hugging Face
        if ckpt_path is None:
            try:
                from huggingface_hub import hf_hub_download
                print(f"[ScDiVa] Downloading weights from HF: {model_name_or_path}")
                try:
                    ckpt_path = hf_hub_download(repo_id=model_name_or_path, filename="model.safetensors", token=use_auth_token)
                except:
                    ckpt_path = hf_hub_download(repo_id=model_name_or_path, filename="pytorch_model.bin", token=use_auth_token)
            except ImportError:
                pass
            except Exception as e:
                print(f"[ScDiVa] Warning: HF download failed: {e}")

        # 3. Load or Fallback
        if ckpt_path is None:
            print(f"[ScDiVa] Warning: No weights found. Using random initialization (DEMO MODE).")
            return model

        print(f"[ScDiVa] Loading weights from {ckpt_path}...")
        try:
            state = torch.load(ckpt_path, map_location=map_location)
            state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            if missing: print(f"Missing keys: {len(missing)}")
        except Exception as e:
            print(f"[ScDiVa] Error loading weights: {e}. Using random init.")
            
        return model
