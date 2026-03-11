import math
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
        # ── Pre-norm (attention) ──────────────────────────────
        self.attn_output_norm = attn_output_norm
        if self.attn_output_norm == "rmsnorm":
            self.attn_norm = RMSNorm(hidden_size, eps=attn_norm_eps)
        elif self.attn_output_norm == "layernorm":
            self.attn_norm = LayerNorm(hidden_size, eps=attn_norm_eps)
        else:
            raise ValueError(f"Unsupported attn_output_norm: {self.attn_output_norm}")

        # ── Pre-norm (MLP) ────────────────────────────────────
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
        vocab_size: int = 65530,
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

        self.apply(self._init_weights) # basic initialization
        self._apply_residual_scaling(n_layer)
        self.lm_head.weight = self.emb.weight # [FIX] Should be after basic initialization


    def _init_weights(self, module):
        """Linear / Embedding / Norm。"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, RMSNorm, LayerNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_residual_scaling(self, n_layer: int) -> None:
        """[FIX-1] 对残差分支的输出投影按 1/√(2·N) 缩放。

        依据：GPT-2 论文及 FLA 实现。若每个残差分支的输出 std≈0.02，
        则经过 N 层叠加后残差通道的方差会增长 N 倍。
        在输出投影处预先缩放可使叠加后方差保持 O(1)。

        数学上：Var[Σ_i x_i] = N · Var[x_i]（独立同分布假设）
        缩放后：σ_scaled = 0.02 / √(2·N)，使得：
            Var[残差和] ≈ 2N · σ_scaled² = 2N · (0.02)²/(2N) = (0.02)²  ✓
        （factor 2 来自每个 block 含 attention + MLP 两条残差分支）
        """
        std = 0.02 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            # attention 输出投影
            nn.init.trunc_normal_(block.attn.o_proj.weight, std=std)

            # MLP 输出投影（SwiGLU 标准命名为 down_proj，也可能是 out_proj / fc2 / w2）
            _mlp_out_candidates = ("down_proj", "out_proj", "fc2", "w2")
            for attr_name in _mlp_out_candidates:
                proj = getattr(block.mlp, attr_name, None)
                if isinstance(proj, nn.Linear):
                    nn.init.trunc_normal_(proj.weight, std=std)
                    break
            else:
                # 如果上述名字都不匹配，对 MLP 内最后一个 Linear 做缩放
                mlp_linears = [
                    m for m in block.mlp.modules() if isinstance(m, nn.Linear)
                ]
                if mlp_linears:
                    nn.init.trunc_normal_(mlp_linears[-1].weight, std=std)

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