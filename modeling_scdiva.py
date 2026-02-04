"""
ScDiVa: A Foundation Model for Single-cell Genomics
Model Architecture Definition

This file contains the core architecture definition of ScDiVa.
It allows loading pre-trained weights for inference.
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
        num_genes: int = 41818,  # Updated to match paper (Table 4)
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

class GeneEmbedding(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.gene_projection = nn.Linear(config.num_genes, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        embeddings = self.gene_projection(gene_expression)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        return attention_output

class FeedForward(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        layer_output = self.feed_forward(attention_output)
        return layer_output

class TransformerEncoder(nn.Module):
    def __init__(self, config: ScDiVaConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
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
        """
        Input Shape: (batch_size, num_genes)
        Returns: Dict containing latent, mu, logvar
        """
        embeddings = self.gene_embedding(gene_expression)
        embeddings = embeddings.unsqueeze(1)  # (B, 1, H)
        encoded = self.encoder(embeddings, attention_mask)  # (B, 1, H)
        encoded = encoded.squeeze(1)  # (B, H)
        z, mu, logvar = self.variational_layer(encoded)
        return {"latent": z, "mu": mu, "logvar": logvar}
    
    def predict(self, gene_expression: torch.Tensor, task: str = "annotation", attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference interface:
        - task="annotation": returns classification logits
        - task="batch_integration": returns reconstructed expression
        """
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
        """
        Load pre-trained model from local path or Hugging Face Hub.
        Supports directly loading from 'warming666/ScDiVa'.
        """
        config = ScDiVaConfig()
        model = cls(config)

        if map_location is None:
            map_location = "cpu"

        ckpt_path: Optional[str] = None

        # 1. Try Local File
        if os.path.exists(model_name_or_path):
            if os.path.isfile(model_name_or_path):
                ckpt_path = model_name_or_path
            elif os.path.isdir(model_name_or_path):
                # Search for typical weights file
                for name in ["pytorch_model.bin", "model.safetensors", "model.pt"]:
                    p = os.path.join(model_name_or_path, name)
                    if os.path.exists(p):
                        ckpt_path = p
                        break
        
        # 2. Try Hugging Face Hub Download
        if ckpt_path is None:
            try:
                from huggingface_hub import hf_hub_download
                print(f"[ScDiVa] Attempting to download weights from HF: {model_name_or_path}")
                # Try safetensors first, then bin
                try:
                    ckpt_path = hf_hub_download(repo_id=model_name_or_path, filename="model.safetensors", token=use_auth_token)
                except:
                    # Fallback to pytorch_model.bin
                    try:
                        ckpt_path = hf_hub_download(repo_id=model_name_or_path, filename="pytorch_model.bin", token=use_auth_token)
                    except:
                        pass
            except ImportError:
                print("[ScDiVa] Warning: `huggingface_hub` not installed. Cannot download from HF.")
            except Exception as e:
                print(f"[ScDiVa] Warning: HF download error (check network/repo ID): {e}")

        # 3. Load or Fallback to Random Init (Demo Mode)
        if ckpt_path is None:
            print(f"[ScDiVa] Warning: No weights found at '{model_name_or_path}'. Using random initialization (DEMO MODE).")
            return model

        print(f"[ScDiVa] Loading weights from {ckpt_path}...")
        try:
            state = torch.load(ckpt_path, map_location=map_location)
            # Support both raw state_dict and dictionary containing state_dict
            state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
            
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            if missing:
                print(f"[ScDiVa] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[ScDiVa] Unexpected keys: {len(unexpected)}")
            print("✅ Model weights loaded successfully.")
                
        except Exception as e:
            print(f"[ScDiVa] Error loading weights: {e}")
            print("[ScDiVa] Model structure initialized with random weights.")

        return model
