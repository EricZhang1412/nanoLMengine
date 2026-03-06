import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from .linear_attn_layer import LinearAttention
from .norm import RMSNorm, LayerNorm
from .mlp import GatedMLP

class LinearAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024, # input hidden state dimension
        num_heads: int = 8, # number of attention heads
        intermediate_size: int = 4096, # intermediate dimension of MLP
        attn_expand_k: float = 1.0,
        attn_expand_v: float = 1.0,
        attn_feature_map: str = "identity",
        attn_output_norm: str = "rmsnorm",
        attn_norm_q: bool = False,
        attn_norm_k: bool = False,
        attn_norm_eps: float = 1e-5,
        mlp_reduce_output: bool = True,
        mlp_act_fun: str = "swiglu",
        mlp_norm_type: str = "rmsnorm",
        mlp_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.attn_output_norm = attn_output_norm
        if self.attn_output_norm == "rmsnorm":
            self.attn_norm = RMSNorm(hidden_size, eps=attn_norm_eps)
        elif self.attn_output_norm == "layernorm":
            self.attn_norm = LayerNorm(hidden_size, eps=attn_norm_eps)
        else:
            raise ValueError(f"Unsupported attn_output_norm: {self.attn_output_norm}")

        self.mlp_norm_type = mlp_norm_type
        if self.mlp_norm_type == "rmsnorm":
            self.mlp_norm = RMSNorm(hidden_size, eps=mlp_norm_eps)
        elif self.mlp_norm_type == "layernorm":
            self.mlp_norm = LayerNorm(hidden_size, eps=mlp_norm_eps)
        else:
            raise ValueError(f"Unsupported mlp_norm_type: {self.mlp_norm_type}")

        self.attn = LinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expand_k=attn_expand_k,
            expand_v=attn_expand_v,
            feature_map=attn_feature_map,
            output_norm=attn_output_norm,
            norm_q=attn_norm_q,
            norm_k=attn_norm_k,
            norm_eps=attn_norm_eps,
        )
        self.mlp = GatedMLP(
            dim=hidden_size,
            inter_dim=intermediate_size,
            reduce_output=mlp_reduce_output,
            act_fun=mlp_act_fun,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # attention block
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states

        # mlp block
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LinearAttentionLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        ctx_len: int = 1024,
        d_model: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        attn_expand_k: float = 1.0,
        attn_expand_v: float = 1.0,
        attn_feature_map: str = "identity",
        attn_output_norm: str = "rmsnorm",
        attn_norm_q: bool = False,
        attn_norm_k: bool = False,
        attn_norm_eps: float = 1e-5,
        mlp_intermediate_size: int = None,
        mlp_reduce_output: bool = True,
        mlp_act_fun: str = "swiglu",
        mlp_norm_type: str = "rmsnorm",
        mlp_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.emb = nn.Embedding(vocab_size, d_model)

        intermediate_size = 4*d_model if mlp_intermediate_size is None else mlp_intermediate_size
        self.blocks = nn.ModuleList([LinearAttentionBlock(
            hidden_size=d_model,
            num_heads=n_head,
            intermediate_size=intermediate_size,    
            attn_expand_k=attn_expand_k,
            attn_expand_v=attn_expand_v,
            attn_feature_map=attn_feature_map,
            attn_output_norm=attn_output_norm,
            attn_norm_q=attn_norm_q,
            attn_norm_k=attn_norm_k,
            attn_norm_eps=attn_norm_eps,
            mlp_reduce_output=mlp_reduce_output,
            mlp_act_fun=mlp_act_fun,
            mlp_norm_type=mlp_norm_type,
            mlp_norm_eps=mlp_norm_eps,
        ) for _ in range(n_layer)])
        self.ln_out = nn.RMSNorm(d_model, eps=float(mlp_norm_eps))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lm_head.weight = self.emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.ctx_len, f"input_ids length {T} exceeds context length {self.ctx_len}"

        x = self.emb(input_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        logits = self.lm_head(x)
        return logits