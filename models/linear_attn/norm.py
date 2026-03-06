from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): feature dimension
        eps (float): numerical stability
        elementwise_affine (bool): whether to learn scale parameter
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ):
        dtype = x.dtype
        if residual is not None:
            x = x + residual
        x_fp32 = x.float()
        # faster than pow(2)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(var + float(self.eps))

        if self.weight is not None:
            x_norm = x_norm * self.weight

        x_norm = x_norm.to(dtype)

        if residual is None:
            return x_norm
        else:
            return x_norm, x


class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x,
            (x.shape[-1],),
            self.weight,
            self.bias,
            float(self.eps),
        )