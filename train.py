import logging
logging.basicConfig(level=logging.INFO)
import sys

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import pytorch_lightning as pl

from utils.load_config import load_config

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--project_config", type=str, required=True)
    parser.add_argument("--tokenizer_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--optimizer_config", type=str, required=True)
    return parser.parse_args()

def train(args):
    rank_zero_info("########## training in progress ##########")
    project_config = load_config(args.project_config)
    tokenizer_config = load_config(args.tokenizer_config)
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    optimizer_config = load_config(args.optimizer_config)

if __name__ == "__main__":  
    args = parser_args()
    train(args)
