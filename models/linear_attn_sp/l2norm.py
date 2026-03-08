import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse

from einops import rearrange
from typing import Optional, Tuple

def tl_l2norm_fwd_kernel(BD, out_dtype="float16"):
    Tdim = T.dynamic("Tdim", "int32")
    Ddim = T.dynamic("Ddim", "int32")

    @T.prim_func
    def kernel(
        X: T.Tensor((Tdim, Ddim), "float16"),
        Y: T.Tensor((Tdim, Ddim), out_dtype),
        RSTD: T.Tensor((Tdim,), "float32"),
        eps: T.float32,
    ):
        with T.Kernel(Tdim, threads=128) as (bx):
            row = bx
            tx = T.get_thread_binding(0)

            x_frag = T.alloc_fragment((BD,), "float32")
            sq_frag = T.alloc_fragment((BD,), "float32")
            sum_frag = T.alloc_fragment((1,), "float32")
            T.clear(x_frag)
            T.clear(sq_frag)
            T.clear(sum_frag)

            for j in T.Parallel(BD):
                if j < Ddim:
                    x_frag[j] = T.Cast("float32", X[row, j])
                    sq_frag[j] = x_frag[j] * x_frag[j]
                else:
                    x_frag[j] = T.float32(0)
                    sq_frag[j] = T.float32(0)

            T.reduce_sum(sq_frag, sum_frag, dim=0)
            rstd = T.rsqrt(sum_frag[0] + eps)

            if tx == 0:
                RSTD[row] = rstd

            for j in T.Parallel(BD):
                if j < Ddim:
                    Y[row, j] = T.Cast(out_dtype, x_frag[j] * rstd)

    return kernel

def tl_l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x2 = x.view(-1, x.shape[-1]).contiguous()
    T_, D = x2.shape

    if output_dtype is None:
        y = torch.empty_like(x2)
        out_dtype_str = str(x2.dtype).replace("torch.", "")
    else:
        y = torch.empty((T_, D), device=x2.device, dtype=output_dtype)
        out_dtype_str = str(output_dtype).replace("torch.", "")

    rstd = torch.empty((T_,), device=x2.device, dtype=torch.float32)

    BD = 1
    while BD < D:
        BD *= 2
    BD = min(BD, 1024)

    if D > BD:
        raise RuntimeError(f"D={D} exceeds BD={BD}")

    prog = tl_l2norm_fwd_kernel(BD=BD, out_dtype=out_dtype_str)
    mod = tilelang.compile(prog, out_idx=[1, 2])
    y, rstd = mod(x2, eps)

    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])