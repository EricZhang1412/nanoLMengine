import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_head, dropout=0.1, attn_type="naive"):
        super().__init__()
        self.attn_type = attn_type
        assert d_model % n_head == 0

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self._flash_attn = None

    def _get_flash_attn(self):
        if self._flash_attn is None:
            try:
                from flash_attn import flash_attn_func
            except Exception as e:
                raise ImportError(
                    "flash-attn is not available. Please install flash-attn to use attn_type='flash_attn'."
                ) from e
            self._flash_attn = flash_attn_func
        return self._flash_attn

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.attn_type == "naive":
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        elif self.attn_type == "sdpa_torch":
            # PyTorch 2.x: flash/mem-efficient
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        elif self.attn_type == "flash_attn":
            flash_attn_func = self._get_flash_attn()
            q_ = q.transpose(1, 2)  # (B, T, nh, hd)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)

            y_ = flash_attn_func(
                q_, k_, v_,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                softmax_scale=None,
                causal=True,
            )  # (B, T, nh, hd)  :contentReference[oaicite:1]{index=1}
            y = y_.transpose(1, 2).contiguous().view(B, T, C)
        else:
            raise ValueError(f"Unknown attn_type: {self.attn_type}")

        y = self.resid_dropout(self.proj(y))

        return y


class MLP(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_head, dropout=0.1, attn_type="naive"):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, attn_type)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        ctx_len,
        d_model=768,
        n_layer=12,
        n_head=12,
        dropout=0.1,
        attn_type="naive",
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_head, dropout, attn_type)
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.ctx_len

        pos = torch.arange(0, T, device=input_ids.device)
        tok = self.token_emb(input_ids)
        pos = self.pos_emb(pos)
        x = tok + pos

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
