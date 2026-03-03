import torch
import itertools
from torch.utils.data import IterableDataset
from datasets import load_dataset


class HFDataset(IterableDataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.ctx_len = args.ctx_len
        self.text_column = args.text_column

        # 加载流式数据集
        self.dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True)

        self.vocab_size = self.tokenizer.vocab_size

    def __iter__(self):
        token_buffer = []

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iterable = self.dataset
        else:
            iterable = itertools.islice(self.dataset, worker_info.id, None, worker_info.num_workers)

        # 迭代处理数据集中的每个样本
        for sample in iterable:
            try:
                # 获取文本并进行分词
                tokens = self.tokenizer.encode(sample[self.text_column], add_special_tokens=False)
                token_buffer.extend(tokens)

                token_buffer.append(self.tokenizer.eos_token_id)

            except Exception as e:
                print(f"warning: {e}")
                continue

            while len(token_buffer) >= self.ctx_len + 1:

                chunk = token_buffer[:self.ctx_len + 1]
                token_buffer = token_buffer[self.ctx_len + 1:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)

                yield x, y