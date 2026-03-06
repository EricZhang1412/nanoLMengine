# nanoLMengine

This is a customed engine repo for pretraining LMs.

# TODO List
- [x] Add multi-gpu training script
- [x] Add logging based on aimstack engine
- [x] Add resume training scheme
- [ ] Fix the cosine scheduler issue of epoch/steps issue
- [ ] Add evaluation script based on lm-evaluation-harness

# Usage

## Environment setup

### 1. uv installation

The whole project is based on uv. Make sure you have uv installed.
``` shell
pip install uv
```
Create 3rdparty folder and clone the [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness.git) repository.
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
uv run bash train.sh # single gpu
uv run bash multigpu_train.sh # multi gpu
```

### 3. logging (based on aimstack engine)
The training process will be logged in the `./aim_log` folder. You can use `aim up` to visualize the logging.
``` shell
cd ./aim_log
aim up
```
Outputs are like:
``` shell
Running Aim UI on repo `<Repo#-{HASH_ID} path={AIM_PATH} read_only=None>`
Open http://127.0.0.1:43800
Press Ctrl+C to exit
```
> Note: You can manually set the `aim_log_dir` in the `default_project_configs.yaml` file.