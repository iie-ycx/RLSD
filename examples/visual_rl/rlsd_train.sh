#!/bin/bash
# =============================================================================
# 最简 GRPO + RLSD 训练脚本
# Simple GRPO + RLSD (no Navigator)
# =============================================================================
set -euo pipefail
set -x

export PATH="/pfs/siqingyi/miniconda3/bin:$PATH"
source activate rlsd

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
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

PROJECT_DIR=/pfs/qcy/video_rl/rlsd_verl
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_DIR}/examples/visual_rl/rlsd_config.yaml"}

MODEL_PATH=${MODEL_PATH:-"/pfs/qcy/models/Qwen3-VL-8B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/pfs/siqingyi/vl_rl_data/final_data/MMFineReason_data_with_conclusion.json"}
VAL_DATA=${VAL_DATA:-"/pfs/qcy/RL_DATA/val_data/visual/vl_math_val_mini_data.json"}
FORMAT_PROMPT="$PROJECT_DIR/examples/visual_rl/format_prompt/unified.jinja"
TEACHER_TEMPLATE="$PROJECT_DIR/examples/visual_rl/format_prompt/teacher_hint_self_distill.jinja"
REWARD_FUNCTION="$PROJECT_DIR/examples/visual_rl/reward_function/unified.py:compute_score"

OUTPUT_PATH=${OUTPUT_PATH:-"${PROJECT_DIR}/checkpoints/grpo_rlsd_simple"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_rlsd_simple_runz_20_60"}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}
ROLLOUT_N=${ROLLOUT_N:-8}
RLSD_LAMBDA=${RLSD_LAMBDA:-0.5}
RLSD_REWEIGHT_CLIP_RANGE=${RLSD_REWEIGHT_CLIP_RANGE:-0.2}

LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/grpo_rlsd_rank${RANK}_${TIMESTAMP}.log"

echo "============================================================"
echo "  Simple GRPO + RLSD"
echo "============================================================"
echo "  节点总数: ${WORLD_SIZE}, 当前节点: ${RANK}"
echo "  主节点: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  每节点 GPU: ${NPROC_PER_NODE}, 总 GPU: $((WORLD_SIZE * NPROC_PER_NODE))"
echo "------------------------------------------------------------"
echo "  配置文件: ${CONFIG_PATH}"
echo "  模型路径: ${MODEL_PATH}"
echo "  训练数据: ${TRAIN_DATA}"
echo "  验证数据: ${VAL_DATA}"
echo "  输出路径: ${OUTPUT_PATH}"
echo "  实验名称: ${EXPERIMENT_NAME}"
echo "  rollout.n: ${ROLLOUT_N}"
echo "  rollout_batch_size: ${ROLLOUT_BATCH_SIZE}"
echo "  rlsd_lambda: ${RLSD_LAMBDA}"
echo "  rlsd_reweight_clip_range: ${RLSD_REWEIGHT_CLIP_RANGE}"
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

    echo "[HEAD] Launching Simple GRPO + RLSD training..."
    python3 -m verl.trainer.opsd_main \
        config=${CONFIG_PATH} \
        data.train_files=${TRAIN_DATA} \
        data.val_files=${VAL_DATA} \
        data.format_prompt=${FORMAT_PROMPT} \
        data.trajectory_key=answer \
        data.max_response_length=4096 \
        data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
        worker.rollout.n=${ROLLOUT_N} \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.actor.clip_ratio_low=0.2 \
        worker.actor.clip_ratio_high=0.28 \
        worker.reward.reward_function=${REWARD_FUNCTION} \
        algorithm.adv_estimator=grpo \
        algorithm.disable_kl=True \
        opsd.max_trajectory_length=12000 \
        opsd.rlsd_lambda=${RLSD_LAMBDA} \
        opsd.rlsd_lambda_warmup_steps=0 \
        opsd.rlsd_lambda_decay_steps=60 \
        opsd.rlsd_reweight_clip_range=${RLSD_REWEIGHT_CLIP_RANGE} \
        opsd.freeze_teacher_model=True \
        opsd.rlsd_teacher_sync_interval=20 \
        opsd.teacher_hint_template=${TEACHER_TEMPLATE} \
        trainer.project_name=rlsd \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
        trainer.n_gpus_per_node=${NPROC_PER_NODE} \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.val_before_train=True \
        trainer.save_freq=10 \
        trainer.save_limit=5 \
        trainer.val_generations_to_log=3 \
        2>&1 | tee -a "$LOG_FILE"

    echo "[HEAD] Simple GRPO + RLSD training completed!"
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
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats \
        2>&1 | tee -a "$LOG_FILE"

    ray status --address="${MASTER_ADDR}:${MASTER_PORT}"
    echo "[WORKER ${RANK}] Connected and waiting for tasks..."
    sleep inf
fi
