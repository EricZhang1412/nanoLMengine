import logging
logging.basicConfig(level=logging.INFO)
import sys

from argparse import ArgumentParser

import pytorch_lightning as pl
import os, warnings, math, datetime, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
import deepspeed
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning import seed_everything
from utils.load_config import load_config

def parser_args():
    parser = ArgumentParser()
    

    parser.add_argument("--project_config", type=str)
    parser.add_argument("--tokenizer_config", type=str)
    parser.add_argument("--train_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--optimizer_config", type=str)
    return parser.parse_args()

def train(args):
    rank_zero_info("########## training in progress ##########")

    project_config = load_config(args.project_config)
    tokenizer_config = load_config(args.tokenizer_config)
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    optimizer_config = load_config(args.optimizer_config)

    trainer_kwargs = dict(
        strategy=train_config.trainer.strategy,
        precision=train_config.trainer.precision,
        devices=int(os.environ["GPU_PER_NODE"]),
        num_nodes=int(os.environ["N_NODE"]),
        max_epochs=train_config.trainer.max_epochs,
        gradient_clip_val=train_config.trainer.gradient_clip_val,
        log_every_n_steps=train_config.trainer.log_every_n_steps,
        check_val_every_n_epoch=train_config.trainer.check_val_every_n_epoch,
        val_check_interval=train_config.trainer.val_check_interval,
        enable_checkpointing=train_config.trainer.enable_checkpointing,
    )
    if train_config.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {train_config.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n")
        seed_everything(train_config.random_seed)

    np.set_printoptions(precision=8, suppress=True, linewidth=200)

    timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    real_bsz = int(os.environ["GPU_PER_NODE"]) * int(os.environ["N_NODE"]) * train_config.batch_size_per_gpu
    samples_per_epoch = train_config.epoch_steps * real_bsz
    tokens_per_epoch = samples_per_epoch * tokenizer_config.max_seq_len
    print(f"real_bsz: {real_bsz}")
    print(f"samples_per_epoch: {samples_per_epoch}")
    print(f"tokens_per_epoch: {tokens_per_epoch}")

    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
    f"""
    ############################################################################
        # Found torch {torch.__version__}, recommend latest torch
        # Found deepspeed {deepspeed_version}, recommend latest deepspeed
        # Found pytorch_lightning {pl.__version__}, recommend latest pytorch_lightning
    ############################################################################
    """
    )
    rank_zero_info(str(vars(args)) + "\n")

    model_pretrainer = Trainer(**trainer_kwargs)
if __name__ == "__main__":  
    args = parser_args()
    train(args)
