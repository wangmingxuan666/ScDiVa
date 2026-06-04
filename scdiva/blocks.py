# scdiva/blocks.py
import torch
from torch import nn
from torch.nn import functional as F
import math

# 你要求：float32 RMSNorm + eps=1e-5
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).type_as(x)

# 你要求：SwiGLU 使用 gate_proj / up_proj / down_proj 命名
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
    # cos/sin: (L, head_dim) -> (1,1,L,head_dim) 广播
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class RoPESDPAAttention(nn.Module):
    """
    RoPE + (SDPA fast path) + (explicit attn path for analysis)
    """
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

        # ✅ 用于 CellFM Fig5 分析
        self.last_attn = None  # (B, head, L, L)

    def forward(self, x: torch.Tensor, attn_mask_4d: torch.Tensor, capture_attn: bool = False):
        # x: (B, L, H)
        B, L, H = x.shape

        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B,head,L,hd)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ✅ 默认：走 SDPA（快、省显存）
        if not capture_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_4d,  # (B,1,1,L) additive mask 可广播
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # (B,head,L,hd)

            out = out.transpose(1, 2).contiguous().view(B, L, H)
            return self.o_proj(out)

        # ✅ 分析模式：显式算 attention prob 并保存
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,head,L,L)

        if attn_mask_4d is not None:
            # attn_mask_4d: (B,1,1,L) additive mask，直接广播加到最后一维
            scores = scores + attn_mask_4d.to(dtype=scores.dtype)

        # softmax 用 float32 更稳
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)  # (B,head,L,L)

        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)  # (B,head,L,hd)
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.o_proj(out)

        self.last_attn = attn.detach()  # ✅ 保存给外部用（CellFM connectivity）
        return out

class ScDiVaBlock(nn.Module):
    """
    Pre-Norm Residual:
    - RMSNorm
    - RoPE+SDPA Attention
    - RMSNorm
    - SwiGLU(gate/up/down)
    """
    def __init__(self, d_model: int, nhead: int, d_hid: int, dropout: float, max_len: int, rope_theta: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPESDPAAttention(d_model, nhead, dropout, max_len=max_len, rope_theta=rope_theta)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask_4d: torch.Tensor, capture_attn: bool = False):
        h = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask_4d=attn_mask_4d, capture_attn=capture_attn)
        x = h + self.drop(x)

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + self.drop(x)
        return x
