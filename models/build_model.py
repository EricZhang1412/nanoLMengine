# nanoLMengine/models/build_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from .transformer.model import TransformerLM
from .rwkv7.model import RWKV7Model, LitRWKV, RWKV7Block
from .linear_attn.model import LinearAttentionLM
from .linear_attn.norm import RMSNorm, LayerNorm

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

# [FIX-3] 不施加 weight decay 的模块类型
_NO_DECAY_MODULE_TYPES = (
    nn.Embedding,
    nn.LayerNorm,
    nn.RMSNorm,
    RMSNorm,    # 自定义 RMSNorm
    LayerNorm,  # 自定义 LayerNorm
)

def make_optimizer_groups(
    model: nn.Module,
    weight_decay: float,
    verbose: bool = True,
) -> list[dict]:
    """[FIX-3] 将模型参数分成两组：

    - no_decay：Embedding / Norm / bias / 1-D 标量参数
    - decay   ：其余权重矩阵

    遵循 RWKV-LM / FLA / NanoGPT 的最佳实践。

    返回：可直接传给 AdamW 的 param_groups 列表。
    """
    decay_ids: set[int] = set()
    no_decay_ids: set[int] = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            # 按 id 去重（shared / tied weights 只计一次）
            pid = id(param)
            if pid in decay_ids or pid in no_decay_ids:
                continue

            is_no_decay = (
                isinstance(module, _NO_DECAY_MODULE_TYPES)
                or param_name == "bias"
                or param.ndim < 2          # 1-D: norm scale、bias 等
            )

            if is_no_decay:
                no_decay_ids.add(pid)
            else:
                decay_ids.add(pid)

    # 反向映射 id → 参数对象（named_parameters 已解引用 tied weights）
    id2param: dict[int, torch.Tensor] = {
        id(p): p for p in model.parameters()
    }

    decay_params   = [id2param[i] for i in decay_ids   if i in id2param]
    no_decay_params = [id2param[i] for i in no_decay_ids if i in id2param]

    if verbose:
        rank_zero_info(
            f"[Optimizer] decay params: {sum(p.numel() for p in decay_params):,}  "
            f"no_decay params: {sum(p.numel() for p in no_decay_params):,}"
        )

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

class LitLM(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_config: Any,
        train_config: Any,
        tokenizer_config: Any = None,
        tokenizer: Any = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.tokenizer_config = tokenizer_config
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        # [NEW] skip_nan_inf related states
        self._skip_optimizer_step = False
        self._last_grad_norm = None
        self.skipped_steps = 0

        # [NEW] tokens_per_sec
        self._step_start_time = None
        self.tokens_per_step = None


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)  # [B,T,V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="mean",
        )
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/lr", lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        
        # return loss
        return L2Wrap.apply(loss, logits)

    def on_train_start(self):
        world_size = self.trainer.world_size
        micro_bsz = self.train_config.batch_size_per_gpu
        accumulate = self.trainer.accumulate_grad_batches
        seq_len = self.tokenizer_config.max_seq_len if hasattr(self.tokenizer_config, "max_seq_len") else None

        if seq_len is None:
            seq_len = getattr(self.model, "ctx_len", None)

        self.tokens_per_step = (
            world_size * micro_bsz * seq_len * accumulate
        )

    def on_train_batch_start(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            self._step_start_time = time.perf_counter()

    def on_before_optimizer_step(self, optimizer):
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return

        grad_norm = torch.norm(
            torch.stack([p.grad.detach().float().norm(2) for p in params]),
            2,
        )
        param_norm = torch.norm(
            torch.stack([p.detach().float().norm(2) for p in params]),
            2,
        )
        grad_param_ratio = grad_norm / (param_norm + 1e-12)

        self._last_grad_norm = grad_norm
        skip_nan_inf = bool(getattr(self.train_config, "skip_nan_inf", False))
        invalid_grad = torch.isnan(grad_norm) or torch.isinf(grad_norm)
        if skip_nan_inf and invalid_grad:
            self._skip_optimizer_step = True
            self.skipped_steps += 1
            rank_zero_info(
                f"[WARN] Skipping optimizer step due to invalid grad_norm={grad_norm.item()}"
            )
        else:
            self._skip_optimizer_step = False

        self.log("train/grad_norm", grad_norm, on_step=True, sync_dist=True)
        self.log("train/param_norm", param_norm, on_step=True, sync_dist=True)
        self.log("train/grad_param_ratio", grad_param_ratio, on_step=True, sync_dist=True)
        self.log("train/skipped_steps", float(self.skipped_steps), on_step=True, sync_dist=False)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self._skip_optimizer_step:
            optimizer.zero_grad(set_to_none=True)
            return
        start = self._step_start_time
        optimizer.step(closure=optimizer_closure)

        if start is not None and self.trainer.is_global_zero:
            step_time = time.perf_counter() - start
            tokens_per_sec = self.tokens_per_step / step_time
            self.log(
                "train/tokens_per_sec",
                tokens_per_sec,
                on_step=True,
                prog_bar=True,
                sync_dist=False,
            )
            self.log(
                "train/step_time",
                step_time,
                on_step=True,
                sync_dist=False,
            )

    def configure_optimizers(self):
        lr = float(getattr(self.optimizer_config, "lr", 3e-4))
        b1, b2 = float(getattr(self.optimizer_config, "beta1", 0.9)), float(getattr(self.optimizer_config, "beta2", 0.95))
        betas = (b1, b2)
        weight_decay = float(getattr(self.optimizer_config, "weight_decay", 0.1))

        optim_groups = make_optimizer_groups(
            self.model, weight_decay=weight_decay, verbose=True
        )

        # optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        sched_name = getattr(self.optimizer_config, "scheduler", None)
        if not sched_name:
            return optimizer

        if sched_name == "cosine":
            max_steps = int(getattr(self.train_config, "max_steps", 0))
            warmup_steps = int(getattr(self.optimizer_config, "warmup_steps", 0))
            min_lr_ratio = float(getattr(self.optimizer_config, "min_lr_ratio", 0.1))
            if max_steps <= 0:
                rank_zero_info("scheduler=cosine but max_steps not set; return optimizer only.")
                return optimizer

            warmup_steps = int(getattr(self.optimizer_config, "warmup_steps", 0) or 0)

            def lr_lambda(step: int):
                # ----- warmup -----
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                # ----- cosine decay -----
                progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        rank_zero_info(f"Unknown scheduler={sched_name}; return optimizer only.")
        return optimizer

def build_model(
    model_config: Any,
    optimizer_config: Any,
    train_config: Any,
    tokenizer_config: Any = None,
    tokenizer: Any = None,
) -> LitLM:
    if tokenizer is None:
        raise ValueError("tokenizer must be provided to build_model (for vocab_size/eos/pad).")

    vocab_size = int(getattr(tokenizer, "vocab_size", None) or 0)
    if vocab_size <= 0:
        raise ValueError(f"Invalid tokenizer vocab_size={vocab_size}")

    d_model = int(getattr(model_config, "d_model", 768))

    if model_config.name == "transformer": 
        model = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            ctx_len=tokenizer_config.max_seq_len,
            d_model=model_config.d_model,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            dropout=model_config.dropout,
        )
        rank_zero_info(f"Model core: {model.__class__.__name__}")
        rank_zero_info(f"vocab_size={vocab_size}, d_model={d_model}")

        lit_model = LitLM(
            model=model,
            optimizer_config=optimizer_config,
            train_config=train_config,
            tokenizer=tokenizer,
        )
        return lit_model

    elif model_config.name == "rwkv7": 
        args = model_config
        args.vocab_size = tokenizer.vocab_size
        args.ctx_len = tokenizer_config.max_seq_len
        base_model = RWKV7Model(args=args, BlockCls=RWKV7Block)
        rank_zero_info(f"Base Model: {base_model.__class__.__name__}")
        base_model.init_from_rwkv_scheme_(verbose=True, strict=True)

        lit = LitRWKV(
            core=base_model,
            args=args,
            optimizer_config=optimizer_config,
            train_config=train_config,
        )
        return lit

    elif model_config.name == "linear_attn_naive": 
        model = LinearAttentionLM(
            vocab_size=tokenizer.vocab_size,
            ctx_len=tokenizer_config.max_seq_len,
            d_model=model_config.d_model,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            dropout=model_config.dropout,
            attn_expand_k=model_config.attn_expand_k,
            attn_expand_v=model_config.attn_expand_v,
            attn_feature_map=model_config.attn_feature_map,
            attn_output_norm=model_config.attn_output_norm,
            attn_norm_q=model_config.attn_norm_q,
            attn_norm_k=model_config.attn_norm_k,
            attn_norm_eps=model_config.attn_norm_eps,
            mlp_intermediate_size=model_config.mlp_intermediate_size,
            mlp_reduce_output=model_config.mlp_reduce_output,
            mlp_act_fun=model_config.mlp_act_fun,
            mlp_norm_type=model_config.mlp_norm_type,
            mlp_norm_eps=model_config.mlp_norm_eps,
        )
        rank_zero_info(f"Model core: {model.__class__.__name__}")
        rank_zero_info(f"vocab_size={vocab_size}, d_model={d_model}")

        lit_model = LitLM(
            model=model,
            optimizer_config=optimizer_config,
            train_config=train_config,
            tokenizer=tokenizer,
        )
        return lit_model
    
    else: 
        raise ValueError(f"Unknown model: {model_config.name}")