import os
import datetime
import numpy as np
import torch
import functools
import lightning as L
from pathlib import Path
from types import SimpleNamespace
from argparse import Namespace
from omegaconf import DictConfig, ListConfig
import torch.serialization
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, OnExceptionCheckpoint
from aim.pytorch_lightning import AimLogger
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from utils.load_config import load_config
from utils.hf_dataset import load_project_dataset, HFDataset  # 你原来的函数（里面调用 load_dataset）
from utils.tokenizer.base import build_tokenizer
from utils.count_token import TokenCountCallback
from utils.resume import resolve_resume_ckpt, load_aim_run_hash, save_aim_run_hash
from models.build_model import build_model
from models.transformer.model import TransformerBlock
from models.linear_attn.model import LinearAttentionBlock



def parser_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_config", type=str, required=True)
    parser.add_argument("--tokenizer_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--optimizer_config", type=str, required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="auto / none / /path/to/xxx.ckpt"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="checkpoint save dir; default: <project_config.output_dir or ./outputs>/checkpoints/<exp_name>"
    )
    return parser.parse_args()

class ProjectDataModule(L.LightningDataModule):
    def __init__(self, project_config, train_config, tokenizer_config, tokenizer):
        super().__init__()
        self.project_config = project_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config

        self.train_dataset = None

    def setup(self, stage=None):
        world_size = getattr(self.trainer, "world_size", 1) or 1
        is_ddp = world_size > 1

        def barrier():
            self.trainer.strategy.barrier()

        if is_ddp:
            if self.trainer.is_global_zero:
                base_ds = load_project_dataset(self.project_config)
            barrier()
            if not self.trainer.is_global_zero:
                base_ds = load_project_dataset(self.project_config)
            barrier()
        else:
            base_ds = load_project_dataset(self.project_config)

        self.train_dataset = HFDataset(
            dataset=base_ds,
            tokenizer=self.tokenizer,
            ctx_len=self.tokenizer_config.max_seq_len,
            text_column=self.project_config.text_column,
        )

        if self.trainer.is_global_zero:
            rank_zero_info("Train dataset is ready (tokenized iterable).")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size_per_gpu,
            shuffle=False,
            num_workers=getattr(self.train_config, "num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

def is_global_zero_env():
    return int(os.environ.get("RANK", "0")) == 0

def train(args):
    rank_zero_info("########## training in progress ##########")

    torch.serialization.add_safe_globals([
        SimpleNamespace,
        Namespace,
        DictConfig,
        ListConfig,
    ])
    project_config = load_config(args.project_config)
    tokenizer_config = load_config(args.tokenizer_config)
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    optimizer_config = load_config(args.optimizer_config)

    exp_name = (
        f"{model_config.name}_{tokenizer_config.name}"
        f"_seqlen.{tokenizer_config.max_seq_len}"
        f"_bsz.{train_config.batch_size_per_gpu}"
        f"_lr.{optimizer_config.lr}"
        f"_schedule.{optimizer_config.scheduler}"
        f"_warmup.{optimizer_config.warmup_steps}"
    )

    # seed
    if train_config.random_seed >= 0:
        rank_zero_info(
            f"########## WARNING: GLOBAL SEED {train_config.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########"
        )
        seed_everything(train_config.random_seed, workers=True)

    np.set_printoptions(precision=8, suppress=True, linewidth=200)

    timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    gpus_per_node = int(os.environ.get("GPU_PER_NODE", "1"))
    num_nodes = int(os.environ.get("N_NODE", "1"))
    world_size = gpus_per_node * num_nodes
    micro_bsz_per_gpu = int(train_config.batch_size_per_gpu)
    accumulate = int(getattr(train_config.trainer, "accumulate_grad_batches", 1))
    effective_bsz_global = world_size * micro_bsz_per_gpu * accumulate
    # --- token stats --- # 
    seq_len = int(tokenizer_config.max_seq_len)
    tokens_per_update = effective_bsz_global * seq_len
    max_updates = int(getattr(train_config, "max_steps", 0) or 0)
    tokens_total_planned = tokens_per_update * max_updates if max_updates > 0 else None
    rank_zero_info(f"run_id: {timestamp}")
    rank_zero_info(f"world_size: {world_size}")
    rank_zero_info(f"effective_bsz_global: {effective_bsz_global}")
    rank_zero_info(f"tokens_per_update: {tokens_per_update}")
    rank_zero_info(f"max_updates: {max_updates}")
    rank_zero_info(f"tokens_total_planned: {tokens_total_planned}")

    # cudnn / tf32
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if train_config.trainer.precision in ("32", "32-true", "fp32"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    trainer_strategy = train_config.trainer.strategy
    if isinstance(trainer_strategy, str) and trainer_strategy.lower() == "fsdp":
        # [FIX-5] 根据实际模型选择 wrap class
        _model_name = model_config.name
        if _model_name == "linear_attn_naive":
            _wrap_cls = {LinearAttentionBlock}
        else:
            _wrap_cls = {TransformerBlock}

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=_wrap_cls,
        )
        trainer_strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=next(iter(_wrap_cls)),
            use_orig_params=True,
        )

    
    default_root_dir = getattr(project_config, "output_dir", "./outputs")
    ckpt_dir = args.ckpt_dir or os.path.join(default_root_dir, "checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if is_global_zero_env():
        aim_run_hash = load_aim_run_hash(ckpt_dir) if args.resume != "none" else None
        aim_logger = AimLogger(
            repo=project_config.aim_log_dir,
            experiment=exp_name,
            run_hash=aim_run_hash,
        )
    else:
        aim_logger = None
    # Trainer kwargs（Lightning 2.6.1）
    trainer_kwargs = dict(
        strategy=trainer_strategy,     # "deepspeed_stage_2" / "ddp" 
        precision=train_config.trainer.precision,   # "bf16-mixed"/"16-mixed"/"32-true" 
        devices=int(os.environ.get("GPU_PER_NODE", "1")),
        num_nodes=int(os.environ.get("N_NODE", "1")),
        max_epochs=train_config.trainer.max_epochs,
        gradient_clip_val=train_config.trainer.gradient_clip_val,
        gradient_clip_algorithm=train_config.trainer.gradient_clip_algorithm,
        log_every_n_steps=train_config.trainer.log_every_n_steps,
        check_val_every_n_epoch=train_config.trainer.check_val_every_n_epoch,
        val_check_interval=train_config.trainer.val_check_interval,
        enable_checkpointing=train_config.trainer.enable_checkpointing,
        accumulate_grad_batches=train_config.trainer.accumulate_grad_batches,
        limit_train_batches=train_config.limit_train_batches,
        logger=aim_logger,
    )
    save_every_n_train_steps = int(getattr(train_config.trainer, "save_every_n_train_steps", 1000))


    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TokenCountCallback(log_every_n_updates=train_config.trainer.log_every_n_steps),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{step:09d}",
            save_top_k=-1, 
            save_last=True, 
            every_n_train_steps=save_every_n_train_steps, 
            save_on_train_epoch_end=False
        ),
    ]

    tokenizer = build_tokenizer(tokenizer_config)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    datamodule = ProjectDataModule(
        project_config=project_config, 
        train_config=train_config, 
        tokenizer_config=tokenizer_config, 
        tokenizer=tokenizer
    )

    if model_config.name == "rwkv7" and hasattr(model_config, "jit_on"):
        os.environ["RWKV_JIT_ON"] = "1" if model_config.jit_on else "0"
    if model_config.name == "rwkv7" and hasattr(model_config, "head_size"):
        os.environ["RWKV_HEAD_SIZE"] = str(model_config.head_size)

    model = build_model(
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
        tokenizer_config=tokenizer_config,
        tokenizer=tokenizer,
    )
    rank_zero_info(model)
    # print total params
    total_params = sum(p.numel() for p in model.parameters())
    rank_zero_info(f"Total params: {total_params}")

    rank_zero_info(f"tokenizer vocab_size: {tokenizer.vocab_size}")
    rank_zero_info(f"tokenizer eos_token_id: {tokenizer.eos_token_id}")
    max_id = max(tokenizer.trie_tokenizer.idx2token.keys())
    rank_zero_info(f"max_id in vocab: {max_id}")

    # if use multiple gpus, use tensorboard logger
    if trainer_kwargs["devices"] > 1:
        trainer_kwargs["logger"] = TensorBoardLogger(
            save_dir=os.path.join(default_root_dir, "tensorboard"),
            name=exp_name,
        )
    trainer = Trainer(**trainer_kwargs, callbacks=callbacks)

    try:
        run_hash = load_aim_run_hash(aim_logger)
        if run_hash is not None:
            save_aim_run_hash(ckpt_dir, run_hash)
        else:
            rank_zero_info("[WARN] Aim run hash unavailable, skip saving.")
    except Exception as e:
        rank_zero_info(f"[WARN] Failed to save Aim run hash: {e}")

    ckpt_path = resolve_resume_ckpt(args.resume, ckpt_dir)
    if ckpt_path is not None:
        rank_zero_info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        rank_zero_info("Training from scratch (no checkpoint found).")
    

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    args = parser_args()
    train(args)
