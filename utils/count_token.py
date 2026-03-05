import torch
from lightning.pytorch.callbacks import Callback

class TokenCountCallback(Callback):
    def __init__(self, log_every_n_updates: int = 1):
        self.log_every_n_updates = log_every_n_updates
        self._update_tokens_accum = 0.0
        self._tokens_total = 0.0

    def _all_reduce_sum(self, value_local: float, device):
        t = torch.tensor(value_local, device=device, dtype=torch.float32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return float(t.item())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        x, y = batch
        tokens_local = float(x.numel())  # B*T
        tokens_global = self._all_reduce_sum(tokens_local, x.device)
        self._update_tokens_accum += tokens_global
        self._tokens_total += tokens_global

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # 每次更新前触发：此时 _update_tokens_accum 已累积了 A 个 micro-batch 的 tokens
        if trainer.is_global_zero and (trainer.global_step % self.log_every_n_updates == 0):
            pl_module.log("train/tokens_update_global", self._update_tokens_accum,
                          on_step=True, logger=True, sync_dist=False)
            pl_module.log("train/tokens_total_global", self._tokens_total,
                          on_step=True, logger=True, sync_dist=False)
        self._update_tokens_accum = 0.0
