########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib, sys
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.strategies import DeepSpeedStrategy
try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
except Exception:
    deepspeed = None
    DeepSpeedCPUAdam = None
    FusedAdam = None

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ.get("RWKV_JIT_ON", "1") == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load
HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE", "64"))
CHUNK_LEN = 16

cusparse_inc = None
try:
    import nvidia.cusparse
    cusparse_inc = os.path.join(os.path.dirname(nvidia.cusparse.__file__), "include")
except ImportError:
    pass

if not cusparse_inc:
    # Check current environment (uv/venv)
    _python_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _path = os.path.join(sys.prefix, "lib", _python_ver, "site-packages", "nvidia", "cusparse", "include")
    if os.path.exists(_path):
        cusparse_inc = _path
    elif "CONDA_PREFIX" in os.environ:
        # Fallback to CONDA_PREFIX
        _path = os.path.join(os.environ["CONDA_PREFIX"], "lib", _python_ver, "site-packages", "nvidia", "cusparse", "include")
        if os.path.exists(_path):
            cusparse_inc = _path

if not cusparse_inc:
    rank_zero_info("WARNING: cusparse include path not found, using default.")
    cusparse_inc = "/usr/local/cuda/include"

flags = [
    '-res-usage', 
    f'-D_C_={HEAD_SIZE}', 
    f"-D_CHUNK_LEN_={CHUNK_LEN}", 
    "--use_fast_math", 
    "-O3", 
    "-Xptxas -O3", 
    "--extra-device-vectorization"
]
_curr_dir = os.path.dirname(os.path.abspath(__file__))
load(
    name="wind_backstepping", 
    sources=[os.path.join(_curr_dir, 'cuda', 'wkv7_cuda.cu'), os.path.join(_curr_dir, 'cuda', 'wkv7_op.cpp')],
    extra_include_paths=["/usr/local/cuda/include", cusparse_inc],
    is_python_module=False, 
    verbose=True, 
    extra_cuda_cflags=flags
)
class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db
def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    # cast + contiguous
    q,w,k,v,a,b = [i.to(torch.bfloat16).contiguous() for i in [q,w,k,v,a,b]]
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)


class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embed

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    
########################################################################################################

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embed)
            for i in range(args.n_embed):
                ddd[0, 0, i] = i / args.n_embed
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embed, args.n_embed * 4, bias=False)
        self.value = nn.Linear(args.n_embed * 4, args.n_embed, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embed**0.5), 0.5/(args.n_embed**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)

class RWKV7Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embed)
        self.ln2 = nn.LayerNorm(args.n_embed)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embed)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV7Model(nn.Module):
    def __init__(self, args, BlockCls):
        super().__init__()
        self.args = args

        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embed
        if not hasattr(args, "dim_ffn"):
            args.dim_ffn = int((args.n_embed * 3.5) // 32 * 32)

        assert args.n_embed % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embed)
        self.blocks = nn.ModuleList([BlockCls(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embed)
        self.head = nn.Linear(args.n_embed, args.vocab_size, bias=False)

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if getattr(args, "grad_cp", 0) == 1 and deepspeed is not None:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def generate_init_weight(self, verbose: bool = True) -> dict:
        
        rank_zero_info("""Init model weight (slow for large models)...""")

        def _get_layer_id_from_name(n: str) -> int | None:
            m = re.search(r"\bblocks\.(\d+)\b", n)
            return int(m.group(1)) if m else None

        m = {}
        n_params = 0
        sd = self.state_dict()

        float_mode = os.environ.get("RWKV_FLOAT_MODE", "bf16").lower()  # "fp16" / "bf16" / ""(fp32)
        use_cuda_alloc = (os.environ.get("RWKV_ACCELERATOR", "CPU").upper() == "GPU") and torch.cuda.is_available()
        for n, p in sd.items():
            shape = p.shape
            if verbose and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
                s0 = str(shape[0]) if len(shape) > 0 else ""
                s1 = str(shape[1]) if len(shape) > 1 else ""
                s2 = str(shape[2]) if len(shape) > 2 else ""
                s3 = str(shape[3]) if len(shape) > 3 else ""
                print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")
            scale = 1.0

            if (
                ("ln_" in n) or (".ln" in n) or ("time_" in n) or ("_mask" in n)
                or ("pos_emb" in n) or (".mask." in n)
                or n.endswith("_w") or n.endswith("_w1") or n.endswith("_w2") or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    # layer_scale = (1 + layer_id) / n_layer
                    layer_id = _get_layer_id_from_name(n)
                    if layer_id is None:
                        m[n] = (p * 0.0) + 1.0
                    else:
                        layer_scale = (1 + layer_id) / float(self.args.n_layer)
                        m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                if verbose:
                    print()
            elif n == "emb.weight":
                m[n] = p.clone()
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                if verbose:
                    print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p.clone()
                if self.args.vocab_size > self.args.n_embed:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embed)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                if verbose:
                    print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight"), f"Unexpected param name (not .weight): {n}"

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]
                for kk in zero:
                    if kk in n:
                        scale = 0.0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                if verbose:
                    rank_zero_info(f" [scale {scale}]")

                if len(shape) != 2:
                    m[n] = p.clone()
                else:
                    if use_cuda_alloc:
                        w = torch.empty((shape[0], shape[1]), device="cuda", dtype=torch.float32)
                    else:
                        w = torch.empty((shape[0], shape[1]), device="cpu", dtype=torch.float32)

                    if scale == 0.0:
                        nn.init.zeros_(w)
                    elif scale < 0:
                        nn.init.uniform_(w, a=scale, b=-scale)
                    else:
                        nn.init.orthogonal_(w, gain=scale)

                    m[n] = w

            m[n] = m[n].detach().cpu()
            if float_mode == "fp16":
                m[n] = m[n].half()
            elif float_mode == "bf16":
                m[n] = m[n].bfloat16()

            n_params += m[n].numel()

        
        rank_zero_info(f"model params {n_params}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return m

    @torch.no_grad()
    def init_from_rwkv_scheme_(self, verbose: bool = True, strict: bool = True):
        init_sd = self.generate_init_weight(verbose=verbose)
        missing, unexpected = self.load_state_dict(init_sd, strict=strict)
        if verbose and (missing or unexpected):
            print("missing keys:", missing)
            print("unexpected keys:", unexpected)
        return self

class LitRWKV(L.LightningModule):
    def __init__(self, core: nn.Module, args, optimizer_config=None, train_config=None):
        super().__init__()
        self.core = core
        self.args = args
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.save_hyperparameters(ignore=["core"])
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.core(idx)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = getattr(self.trainer, "strategy", None)
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config.get("zero_optimization", {})
            return bool(cfg.get("offload_optimizer") or cfg.get("offload_param"))
        return False

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)  # [B,T,V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # add other monitors 
        # lr
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return L2Wrap.apply(loss, logits)

    # grad_norm
    def on_before_optimizer_step(self, optimizer):
        total_norm = torch.norm(
            torch.stack([
                p.grad.detach().norm()
                for p in self.parameters()
                if p.grad is not None
            ])
        )
        self.log("train/grad_norm", total_norm, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        args = self.args
        lr = float(getattr(self.optimizer_config, "lr", getattr(args, "lr_init", 3e-4)))
        b1, b2 = float(getattr(self.optimizer_config, "beta1", getattr(args, "beta1", 0.9))), float(getattr(self.optimizer_config, "beta2", getattr(args, "beta2", 0.95)))
        betas = (b1, b2)
        adam_eps = float(getattr(self.optimizer_config, "adam_eps", getattr(args, "adam_eps", 1e-8)))
        weight_decay = float(getattr(self.optimizer_config, "weight_decay", getattr(args, "weight_decay", 0.0)))

        lr_decay, lr_1x, lr_2x = set(), set(), set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(lr_decay)
        lr_1x = sorted(lr_1x)
        lr_2x = sorted(lr_2x)

        if self.trainer.is_global_zero:
            rank_zero_info(f"decay: {lr_decay}")
            rank_zero_info(f"1x: {lr_1x}")
            rank_zero_info(f"2x: {lr_2x}")

        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]
        if weight_decay > 0:
            optim_groups.append({"params": [param_dict[n] for n in lr_decay], "weight_decay": weight_decay, "my_lr_scale": 1.0})

        use_deepspeed_adam = (
            (deepspeed is not None)
            and (FusedAdam is not None)
            and isinstance(getattr(self.trainer, "strategy", None), DeepSpeedStrategy)
        )

        if use_deepspeed_adam:
            if self.deepspeed_offload and DeepSpeedCPUAdam is not None:
                optimizer = DeepSpeedCPUAdam(
                    optim_groups, lr=lr, betas=betas, eps=adam_eps,
                    bias_correction=True,
                    adamw_mode=(weight_decay > 0),
                    amsgrad=False,
                )
            else:
                optimizer = FusedAdam(
                    optim_groups, lr=lr, betas=betas, eps=adam_eps,
                    bias_correction=True,
                    adam_w_mode=(weight_decay > 0),
                    amsgrad=False,
                )
        else:
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=lr,
                betas=betas,
                eps=adam_eps,
                weight_decay=0.0,
            )

        sched_name = getattr(self.optimizer_config, "scheduler", None) if self.optimizer_config is not None else None
        if not sched_name:
            return optimizer

        if sched_name == "cosine":
            max_steps = int(getattr(self.train_config, "max_steps", 0) or 0) if self.train_config is not None else 0
            warmup_steps = int(getattr(self.optimizer_config, "warmup_steps", 0) or 0) if self.optimizer_config is not None else 0
            if max_steps <= 0:
                rank_zero_info("scheduler=cosine but max_steps not set; return optimizer only.")
                return optimizer

            def lr_lambda(step: int):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(progress * math.pi))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        rank_zero_info(f"Unknown scheduler={sched_name}; return optimizer only.")
        return optimizer