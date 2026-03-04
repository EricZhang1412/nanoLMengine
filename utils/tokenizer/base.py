import os
from transformers import AutoTokenizer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from .rwkv_tokenizer.rwkv_trie_tokenizer import RWKVTrieTokenizerForTraining

def build_tokenizer(tokenizer_config):
    """
    Build tokenizer from config.

    tokenizer_config example:
    {
        name: "gpt2"
        padding_side: "right"
        use_fast: true
    }
    """

    name = tokenizer_config.name
    vocab_file = getattr(tokenizer_config, "vocab_file", None)
    if vocab_file is not None and os.path.isfile(vocab_file) and vocab_file.endswith(".txt"):
        tokenizer = RWKVTrieTokenizerForTraining(vocab_file=vocab_file)
        tokenizer.padding_side = getattr(tokenizer_config, "padding_side", "right")

        rank_zero_info(f"Tokenizer loaded: rwkv_trie from {vocab_file}")
        rank_zero_info(f"vocab_size: {tokenizer.vocab_size}")
        rank_zero_info(f"eos_token_id: {tokenizer.eos_token_id}")
        return tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        use_fast=getattr(tokenizer_config, "use_fast", True),
        trust_remote_code=True,
        truncation=getattr(tokenizer_config, "truncation", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = getattr(tokenizer_config, "padding_side", "right")

    rank_zero_info(f"Tokenizer loaded: {name}")
    rank_zero_info(f"vocab_size: {tokenizer.vocab_size}")
    rank_zero_info(f"eos_token_id: {tokenizer.eos_token_id}")

    return tokenizer

def get_vocab_size(tokenizer):
    return tokenizer.vocab_size

def get_eos_id(tokenizer):
    return tokenizer.eos_token_id