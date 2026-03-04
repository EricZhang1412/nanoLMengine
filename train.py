import os
import datetime
import numpy as np
import torch

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from torch.utils.data import DataLoader

from utils.load_config import load_config
from utils.hf_dataset import load_project_dataset, HFDataset  # 你原来的函数（里面调用 load_dataset）
from utils.tokenizer.base import build_tokenizer
from models.build_model import build_model



def parser_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_config", type=str, required=True)
    parser.add_argument("--tokenizer_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--optimizer_config", type=str, required=True)
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

def train(args):
    rank_zero_info("########## training in progress ##########")

    project_config = load_config(args.project_config)
    tokenizer_config = load_config(args.tokenizer_config)
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    optimizer_config = load_config(args.optimizer_config)

    # seed
    if train_config.random_seed >= 0:
        rank_zero_info(
            f"########## WARNING: GLOBAL SEED {train_config.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########"
        )
        seed_everything(train_config.random_seed, workers=True)

    np.set_printoptions(precision=8, suppress=True, linewidth=200)

    timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    real_bsz = int(os.environ.get("GPU_PER_NODE", "1")) * int(os.environ.get("N_NODE", "1")) * train_config.batch_size_per_gpu
    samples_per_epoch = train_config.epoch_steps * real_bsz
    tokens_per_epoch = samples_per_epoch * tokenizer_config.max_seq_len
    rank_zero_info(f"run_id: {timestamp}")
    rank_zero_info(f"real_bsz: {real_bsz}")
    rank_zero_info(f"samples_per_epoch: {samples_per_epoch}")
    rank_zero_info(f"tokens_per_epoch: {tokens_per_epoch}")

    # cudnn / tf32
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if train_config.trainer.precision in ("32", "32-true", "fp32"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Trainer kwargs（Lightning 2.6.1）
    trainer_kwargs = dict(
        strategy=train_config.trainer.strategy,     # "deepspeed_stage_2" / "ddp" 
        precision=train_config.trainer.precision,   # "bf16-mixed"/"16-mixed"/"32-true" 
        devices=int(os.environ.get("GPU_PER_NODE", "1")),
        num_nodes=int(os.environ.get("N_NODE", "1")),
        max_epochs=train_config.trainer.max_epochs,
        gradient_clip_val=train_config.trainer.gradient_clip_val,
        log_every_n_steps=train_config.trainer.log_every_n_steps,
        check_val_every_n_epoch=train_config.trainer.check_val_every_n_epoch,
        val_check_interval=train_config.trainer.val_check_interval,
        enable_checkpointing=train_config.trainer.enable_checkpointing,
    )

    tokenizer = build_tokenizer(tokenizer_config)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    datamodule = ProjectDataModule(
        project_config=project_config, 
        train_config=train_config, 
        tokenizer_config=tokenizer_config, 
        tokenizer=tokenizer
    )

    # TODO: 这里换成你真实的 LightningModule
    # from your_package.model import MyLitModule
    # model = MyLitModule(model_config, optimizer_config, tokenizer_config, train_config)
    # model = None  # <- 你必须替换掉
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

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    args = parser_args()
    train(args)
