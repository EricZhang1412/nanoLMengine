# nanoLMengine/models/build_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from .transformer.model import TransformerLM
# from .rwkv7.model import RWKV7Model, LitRWKV, RWKV7Block
from .linear_attn.model import LinearAttentionLM

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

class LitLM(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_config: Any,
        train_config: Any,
        tokenizer: Any = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=["model", "tokenizer"])

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
        return loss
        # return L2Wrap.apply(loss, logits)

    def configure_optimizers(self):
        lr = float(getattr(self.optimizer_config, "lr", 3e-4))
        b1, b2 = float(getattr(self.optimizer_config, "beta1", 0.9)), float(getattr(self.optimizer_config, "beta2", 0.95))
        betas = (b1, b2)
        weight_decay = float(getattr(self.optimizer_config, "weight_decay", 0.1))

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

    # elif model_config.name == "rwkv7": 
    #     args = model_config
    #     args.vocab_size = tokenizer.vocab_size
    #     args.ctx_len = tokenizer_config.max_seq_len
    #     base_model = RWKV7Model(args=args, BlockCls=RWKV7Block)
    #     rank_zero_info(f"Base Model: {base_model.__class__.__name__}")
    #     base_model.init_from_rwkv_scheme_(verbose=True, strict=True)

    #     lit = LitRWKV(
    #         core=base_model,
    #         args=args,
    #         optimizer_config=optimizer_config,
    #         train_config=train_config,
    #     )
    #     return lit

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