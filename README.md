# nanoLMengine

This is a customed engine repo for pretraining LMs.

# Usage

## Environment setup

### 1. uv installation

The whole project is based on uv. Make sure you have uv installed.
``` shell
pip install uv
```
Create 3rdparty folder and clone the [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention.git),  [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness.git) and [`flame`](https://github.com/fla-org/flame.git) repository.
``` shell
mkdir 3rdparty
cd 3rdparty
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
```
Then, you can install the dependencies by:
``` shell
uv sync
```
For more usage of uv, you can refer to [uv](https://docs.astral.sh/uv/).

> **Note**:
> If the env setup is not complete, you can denote `"flash-attn==2.7.3",` in the `pyproject.toml` to skip the installation of flash-attn. Then, after pytorch installation, you can install flash-attn again.

### 2. Run the training script

``` shell
uv run bash train.sh
```