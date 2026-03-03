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
    # parser = add_trainer_args(parser)
    group = parser.add_argument_group("Trainer")
    group.add_argument("--strategy", type=str, default="ddp")
    group.add_argument("--precision", type=str, default="bf16")
    group.add_argument("--gpus", type=int, default=1)
    group.add_argument("--num_nodes", type=int, default=1)
    group.add_argument("--max_epochs", type=int, default=10)
    group.add_argument("--gradient_clip_val", type=float, default=0.0)
    group.add_argument("--log_every_n_steps", type=int, default=100)
    group.add_argument("--check_val_every_n_epoch", type=int, default=1)
    group.add_argument("--val_check_interval", type=float, default=0.5)
    group.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()

def train(args):
    rank_zero_info("########## training in progress ##########")
    project_config = load_config(args.project_config)
    tokenizer_config = load_config(args.tokenizer_config)
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    optimizer_config = load_config(args.optimizer_config)

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
if __name__ == "__main__":  
    args = parser_args()
    train(args)
