#!/bin/bash
# =============================================================================
# Baseline GRPO Training Script
# 标准 GRPO 训练启动脚本，与 Author-Reviewer 方法对齐以进行公平对比
# =============================================================================
#
# 使用方式：
#   单节点: bash /pfs/qcy/video_rl/rlsd_verl/examples/visual_rl/grpo_baseline_train_MMFineReason.sh
#   多节点:
#     Node 0 (Head): WORLD_SIZE=2 RANK=0 MASTER_ADDR=<head_ip> bash /pfs/qcy/video_rl/rlsd_verl/examples/visual_rl/grpo_baseline_train_MMFineReason.sh
#     Node 1 (Worker): WORLD_SIZE=2 RANK=1 MASTER_ADDR=<head_ip> bash /pfs/qcy/video_rl/rlsd_verl/examples/visual_rl/grpo_baseline_train_MMFineReason.sh
#
# 环境变量：
#   MODEL_PATH, TRAIN_DATA, VAL_DATA, OUTPUT_PATH, EXPERIMENT_NAME, NPROC_PER_NODE
set -euo pipefail
set -x

export PATH="/pfs/siqingyi/miniconda3/bin:$PATH"
source activate rlsd

# =============================================================================
# 系统资源上限
# =============================================================================
ulimit -n 65536 2>/dev/null || ulimit -n 32768 2>/dev/null || ulimit -n 16384 2>/dev/null || echo "Warning: Could not increase ulimit -n"
ulimit -u 65536 2>/dev/null || ulimit -u 32768 2>/dev/null || echo "Warning: Could not increase ulimit -u"

sysctl -w net.ipv4.ip_local_port_range="1024 65535" 2>/dev/null || true
sysctl -w net.ipv4.tcp_fin_timeout=15 2>/dev/null || true
sysctl -w net.ipv4.tcp_tw_reuse=1 2>/dev/null || true

id sandbox_runner &>/dev/null || useradd -M -s /bin/false sandbox_runner 2>/dev/null || echo "[WARN] Could not create sandbox_runner user"

export WANDB_API_KEY=e72fce795d7e86b88f22eb1218731ec7e748feab
export TOKENIZERS_PARALLELISM=false
export RAY_worker_num_grpc_internal_threads=1
export RAY_ADDRESS=""
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800000
export NCCL_DEBUG=WARN
export VERL_LOG_LEVEL=INFO

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-6379}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
AVAILABLE_CPUS=$(nproc 2>/dev/null || echo 1)
RAY_NUM_CPUS=${RAY_NUM_CPUS:-72}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

PROJECT_DIR=/pfs/qcy/video_rl/rlsd_verl
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_DIR}/examples/visual_rl/grpo_baseline_config.yaml"}

MODEL_PATH=${MODEL_PATH:-"/pfs/qcy/models/Qwen3-VL-8B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/pfs/siqingyi/vl_rl_data/final_data/MMFineReason_data_with_conclusion.json"}
VAL_DATA=${VAL_DATA:-"/pfs/qcy/RL_DATA/val_data/visual/vl_math_val_mini_data.json"}
FORMAT_PROMPT="$PROJECT_DIR/examples/visual_rl/format_prompt/unified.jinja"
REWARD_FUNCTION="${PROJECT_DIR}/examples/visual_rl/reward_function/unified.py:compute_score"

OUTPUT_PATH=${OUTPUT_PATH:-"${PROJECT_DIR}/checkpoints/grpo_baseline_MMFineReason"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_baseline_qwen3vl8b_MMFineReason"}

# 与 Author-Reviewer 对齐的超参数
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}
ROLLOUT_N=${ROLLOUT_N:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/grpo_baseline_rank${RANK}_${TIMESTAMP}.log"

echo "============================================================"
echo "  Baseline GRPO Training - Distributed Mode"
echo "============================================================"
echo "  节点总数: ${WORLD_SIZE}, 当前节点: ${RANK}"
echo "  主节点: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Available CPUs: ${AVAILABLE_CPUS}, Ray CPUs: ${RAY_NUM_CPUS}"
echo "  每节点 GPU: ${NPROC_PER_NODE}, 总 GPU: $((WORLD_SIZE * NPROC_PER_NODE))"
echo "------------------------------------------------------------"
echo "  项目目录: ${PROJECT_DIR}"
echo "  配置文件: ${CONFIG_PATH}"
echo "  模型路径: ${MODEL_PATH}"
echo "  训练数据: ${TRAIN_DATA}"
echo "  验证数据: ${VAL_DATA}"
echo "  输出路径: ${OUTPUT_PATH}"
echo "  实验名称: ${EXPERIMENT_NAME}"
echo "  日志文件: ${LOG_FILE}"
echo "------------------------------------------------------------"
echo "  GRPO 参数 (与 Author-Reviewer 对齐):"
echo "    rollout_batch_size: ${ROLLOUT_BATCH_SIZE}"
echo "    rollout.n: ${ROLLOUT_N}"
echo "    global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "------------------------------------------------------------"
echo "  Reward函数: ${REWARD_FUNCTION}"
echo "============================================================"

cleanup_ray() {
    echo "[INFO] Cleaning up Ray processes..."
    ray stop --force 2>/dev/null || true
    sleep 3
}

wait_for_head() {
    local max_attempts=60
    local attempt=0
    echo "[INFO] Waiting for Head node at ${MASTER_ADDR}:${MASTER_PORT}..."
    while [ $attempt -lt $max_attempts ]; do
        if ray status --address="${MASTER_ADDR}:${MASTER_PORT}" &>/dev/null; then
            echo "[INFO] Head node is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "  等待 Head 节点... ($attempt/$max_attempts)"
        sleep 5
    done
    echo "[ERROR] 等待 Head 节点超时"
    return 1
}

wait_for_workers() {
    local expected_nodes=$WORLD_SIZE
    local max_attempts=120
    local attempt=0

    echo "[INFO] Waiting for $expected_nodes nodes to connect..."
    while [ $attempt -lt $max_attempts ]; do
        local connected_nodes
        connected_nodes=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "  已连接节点: $connected_nodes / $expected_nodes (尝试 $attempt/$max_attempts)"

        if [ "$connected_nodes" -ge "$expected_nodes" ]; then
            echo "[INFO] 所有节点已连接!"
            ray status
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 10
    done

    echo "[ERROR] 未能等到所有节点连接"
    ray status
    return 1
}

SAVE_CHECKPOINT_PATH="${OUTPUT_PATH}/${EXPERIMENT_NAME}"
mkdir -p "${SAVE_CHECKPOINT_PATH}"
export TENSORBOARD_DIR="${SAVE_CHECKPOINT_PATH}/tensorboard_log"
mkdir -p "${TENSORBOARD_DIR}"

if [ "$RANK" == "0" ]; then
    cleanup_ray

    echo "[HEAD] Starting Ray Head node..."
    ray start --head \
        --port=${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${RAY_DASHBOARD_PORT} \
        --num-cpus=${RAY_NUM_CPUS} \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats

    if [ "$WORLD_SIZE" -gt 1 ]; then
        echo "[HEAD] Waiting for Worker nodes to connect..."
        if ! wait_for_workers; then
            echo "[ERROR] 集群未就绪，退出"
            cleanup_ray
            exit 1
        fi
    fi

    echo "[HEAD] Launching Baseline GRPO training..."
    python3 -m verl.trainer.main \
        config=${CONFIG_PATH} \
        data.train_files=${TRAIN_DATA} \
        data.val_files=${VAL_DATA} \
        data.format_prompt=${FORMAT_PROMPT} \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.reward.reward_function=${REWARD_FUNCTION} \
        algorithm.adv_estimator=grpo \
        algorithm.disable_kl=True \
        data.max_response_length=4096 \
        algorithm.use_kl_loss=False \
        data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
        worker.rollout.n=${ROLLOUT_N} \
        worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
        trainer.project_name=rlsd \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
        trainer.n_gpus_per_node=${NPROC_PER_NODE} \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=10 \
        trainer.save_limit=5 \
        trainer.val_before_train=True \
        trainer.val_generations_to_log=3 \
        2>&1 | tee -a "$LOG_FILE"

    echo "[HEAD] Training completed!"
    cleanup_ray
else
    cleanup_ray

    echo "[WORKER ${RANK}] Waiting for Head node..."
    if ! wait_for_head; then
        echo "[ERROR] 无法连接到 Head 节点，退出"
        exit 1
    fi

    sleep 20

    echo "[WORKER ${RANK}] Connecting to Ray cluster..."
    ray start \
        --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-cpus=${RAY_NUM_CPUS} \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats \
        2>&1 | tee -a "$LOG_FILE"

    ray status --address="${MASTER_ADDR}:${MASTER_PORT}"
    echo "[WORKER ${RANK}] Connected and waiting for tasks..."
    sleep inf
fi
