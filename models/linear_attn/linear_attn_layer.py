
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .linear_attn_fwd import tl_fused_chunk_fwd
from .linear_attn_bwd import tl_fused_chunk_bwd
from .norm import RMSNorm

class FusedChunkLinearAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        """
        q, k, v: [B, S, H, D]
        returns:
            o: [B, S, H, D]
        """
        o, final_state = tl_fused_chunk_fwd(q, k, v)
        ctx.save_for_backward(q, k, v)

        # 【TODO】Cache and Recurrent...
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, do):
        """
        do: [B, S, H, D]
        returns:
            dq, dk, dv
        """
        q, k, v = ctx.saved_tensors

        dq, dk, dv = tl_fused_chunk_bwd(
            q,
            k,
            v,
            do.contiguous().to(q.dtype),
        )

        return dq, dk, dv

def fused_chunk_linear_attn(q, k, v):
    return FusedChunkLinearAttentionFn.apply(q, k, v)

def _compute_denominator(q: torch.Tensor, k: torch.Tensor, eps: float) -> torch.Tensor:
    # z_t = cumsum(k, dim=1): [B, S, H, D]
    z = torch.cumsum(k, dim=1)
    # dot product per (token, head): [B, S, H]
    qz = (q * z).sum(dim=-1)           # [B, S, H]
    # 下界：max(|qz|, eps)，保持符号以允许梯度流过（实际值为正）
    denom = qz.abs().clamp(min=eps)    # [B, S, H]

    return denom.unsqueeze(-1)         # [B, S, H, 1]

class LinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        feature_map: str = "identity",
        output_norm: str = "rmsnorm",
        norm_q: bool = False,
        norm_k: bool = False,
        norm_eps: float = 1e-5,
        denom_eps: float = 1.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.denom_eps = denom_eps

        assert self.key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        assert self.value_dim % num_heads == 0, "value_dim must be divisible by num_heads"
        assert self.key_dim == self.value_dim, "current fused TileLang kernel requires key_dim == value_dim"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        assert self.head_k_dim % 64 == 0, f"head_k_dim must be divisible by 64, got {self.head_k_dim}"
        assert self.head_v_dim % 64 == 0, f"head_v_dim must be divisible by 64, got {self.head_v_dim}"

        self.feature_map = feature_map
        self.norm_q = norm_q
        self.norm_k = norm_k

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if self.norm_q:
            self.q_norm = RMSNorm(self.head_k_dim, eps=norm_eps, elementwise_affine=True)
        if self.norm_k:
            self.k_norm = RMSNorm(self.head_k_dim, eps=norm_eps, elementwise_affine=True)

        if output_norm == "rmsnorm":
            self.norm = RMSNorm(self.head_v_dim, eps=norm_eps, elementwise_affine=True)
        elif output_norm == "identity":
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Unsupported output_norm: {output_norm}")

    def _feature_map_q(self, q: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "identity":
            return q
        if self.feature_map == "relu":
            return F.relu(q)
        if self.feature_map == "elu":
            return F.elu(q) + 1.0
        raise NotImplementedError(f"Unsupported feature_map: {self.feature_map}")

    def _feature_map_k(self, k: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "identity":
            return k
        if self.feature_map == "relu":
            return F.relu(k)
        if self.feature_map == "elu":
            return F.elu(k) + 1.0
        raise NotImplementedError(f"Unsupported feature_map: {self.feature_map}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        assert S % 64 == 0, f"sequence length must be divisible by 64, got {S}"

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        q = self._feature_map_q(q)
        k = self._feature_map_k(k)

        if self.norm_q:
            q = self.q_norm(q)
        if self.norm_k:
            k = self.k_norm(k)


        o = fused_chunk_linear_attn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
        )

        # ── [FIX-GRAD] Denominator 归一化 ─────────────────────
        denom = _compute_denominator(q, k, eps=self.denom_eps)
        o = o / denom 

        o = self.norm(o)
        o = rearrange(o, "b s h d -> b s (h d)")
        o = self.o_proj(o)
        return o

def test():
    B, S, H, D = 8, 1024, 64, 64
    hidden_state = torch.randn(B, S, H*D, device="cuda").to(torch.bfloat16)

    attn = LinearAttention(hidden_size=H*D, num_heads=H).cuda().to(torch.bfloat16)
    o = attn(hidden_state)
    print(o.shape)

if __name__ == "__main__":
    test()