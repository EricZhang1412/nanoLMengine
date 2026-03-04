# utils/tokenizer/rwkv_trie_tokenizer.py
import os

class TRIE:
    __slots__ = ("ch", "to", "values", "front")
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            self.values.add(val if val is not None else key)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx + 1, val)

    def find_longest(self, key: bytes, idx: int = 0):
        u = self
        ret = None
        ch = key[idx]
        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = (idx, u, u.values)
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class RWKV_TOKENIZER:
    def __init__(self, file_name: str):
        self.idx2token = {}
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}

        self.root = TRIE()
        for t, i in self.token2idx.items():
            self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx = 0
        out = []
        while idx < len(src):
            found = self.root.find_longest(src, idx)
            if found is None:
                # 理论不会发生；兜底按单字节
                out.append(self.token2idx.get(src[idx:idx+1], 0))
                idx += 1
                continue
            idx, _, values = found
            _, token_id = next(iter(values))
            out.append(token_id)
        return out


class RWKVTrieTokenizerForTraining:
    """
    Minimal tokenizer for your HFDataset:
      - encode(text, add_special_tokens=False) -> List[int]
      - eos_token_id (int)
      - vocab_size (int)
    """
    def __init__(self, vocab_file: str, eos_token_id: int = None):
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"vocab_file not found: {vocab_file}")

        self.vocab_file = vocab_file
        self.trie_tokenizer = RWKV_TOKENIZER(vocab_file)

        self._max_id = max(self.trie_tokenizer.idx2token.keys())  # 你说是 65529
        self._vocab_size = self._max_id + 1                      # 65530

        # ✅ 默认 eos = max_id（65529），不需要传字符串 eos_token
        self._eos_token_id = int(self._max_id if eos_token_id is None else eos_token_id)

        if not (0 <= self._eos_token_id < self._vocab_size):
            raise ValueError(f"eos_token_id out of range: {self._eos_token_id}, vocab_size={self._vocab_size}")

        # 兼容你训练脚本
        self.add_bos_token = False
        self.add_eos_token = False
        self.padding_side = "right"

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        if not isinstance(text, str):
            raise TypeError(f"encode expects str, got {type(text)}")
        return self.trie_tokenizer.encodeBytes(text.encode("utf-8"))
