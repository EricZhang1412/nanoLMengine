HOSTNAME=$(hostname)

#### Multi-Nodes Only
# if [[ "$HOSTNAME" == *"master-0"* ]]; then
#   NODE_RANK=0
# else
#   # 提取 worker 后面的数字，比如 worker-0 ==> 0
#   WORKER_ID=$(echo "$HOSTNAME" | sed -E 's/.*worker-([0-9]+)$/\1/')
#   NODE_RANK=$((WORKER_ID + 1))
# fi

# export NODE_RANK

NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29502}
echo "NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

N_NODE=${N_NODE:-1}
GPU_PER_NODE=${GPU_PER_NODE:-8}

export N_NODE=${N_NODE}
export GPU_PER_NODE=${GPU_PER_NODE}
echo "N_NODE: ${N_NODE}, GPU_PER_NODE: ${GPU_PER_NODE}"

export HF_HOME=${SLURM_TMPDIR:-/tmp}/hf_cache_${USER}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE"
#### Domestic mirrors
export HF_ENDPOINT=https://hf-mirror.com

#### Special PATH vars for RWKV7
export RWKV_JIT_ON=1
export MAX_JOBS=${MAX_JOBS:-8} # ninja workers for jit
# export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_LAUNCH_BLOCKING=1

torchrun \
  --nnodes=${N_NODE}:${N_NODE} \
  --nproc_per_node=${GPU_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
    train.py \
    --project_config configs/default_project_configs.yaml \
    --tokenizer_config configs/tokenizer_configs/rwkv.yaml \
    --train_config configs/train_configs/default.yaml \
    --model_config configs/model_configs/rwkv7_base.yaml \
    --optimizer_config configs/optimizer_configs/lr_3e-4_adam_cosine_warmp1000.yaml \
    --resume auto