import glob
import os
import torch
import itertools
from torch.utils.data import IterableDataset
from datasets import load_dataset


def load_project_dataset(project_config):
    data_path = project_config.path
    if getattr(project_config, "sample", None):
        data_path = os.path.join(data_path, "sample", str(project_config.sample))

    if os.path.isdir(data_path):
        parquet_files = glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under: {data_path}")
        return load_dataset("parquet", data_files={"train": parquet_files}, split="train", streaming=True)

    return load_dataset(project_config.data_name, split="train", streaming=True)


class HFDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, ctx_len, text_column):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.text_column = text_column

    def __iter__(self):
        token_buffer = []

        # ---- distributed slicing (rank/world_size) ----
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # ---- dataloader worker slicing ----
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        stride = world_size * num_workers
        offset = rank * num_workers + worker_id

        iterable = itertools.islice(self.dataset, offset, None, stride)

        for sample in iterable:
            try:
                text = sample[self.text_column]
                if not isinstance(text, str) or len(text) == 0:
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(tokens)
                token_buffer.append(self.tokenizer.eos_token_id)

            except Exception as e:
                if rank == 0 and worker_id == 0:
                    print(f"warning: {e}")
                continue

            while len(token_buffer) >= self.ctx_len + 1:
                chunk = token_buffer[: self.ctx_len + 1]
                token_buffer = token_buffer[self.ctx_len + 1 :]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

        if worker_id == 0:
            print(f"[rank {rank}] using offset={offset}, stride={stride}")
