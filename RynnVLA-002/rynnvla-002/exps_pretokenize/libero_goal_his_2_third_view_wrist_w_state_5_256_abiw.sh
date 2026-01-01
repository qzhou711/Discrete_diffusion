#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# 显存优化说明：
# - 当前使用 FSDP FULL_SHARD 策略，将模型参数、梯度和优化器状态分片到多个GPU
# - 使用 bf16 混合精度训练
# - 使用梯度检查点（checkpointing）进一步减少显存
# - 使用 LoRA 减少可训练参数和优化器状态（已启用）
# - 如果24GB显存仍不足，可以：
#   1. 减小 max_seq_len（最重要！当前为4096，显存占用与序列长度平方相关）
#      * 序列长度对显存影响巨大：max_seq_len=4096 时，激活值占用非常大
#      * 建议值：2048（约 4-8 GB/GPU），1024（约 2-4 GB/GPU），768（约 1.5-3 GB/GPU），512（约 1-2 GB/GPU）
#   2. 增加GPU数量（更多GPU = 更少每个GPU的显存占用）
#   3. 减小 accum_iter（当前为1，已是最小值，无法再减小）
#      * 为什么 accum_iter=1 是最优的：
#        - 优化器状态（AdamW 的 momentum 和 variance）在 step() 时需要临时内存
#        - 更频繁的 step() 可以让内存管理器更及时地回收和重用内存
#        - 梯度累积期间，中间激活值可能保留更长时间，accum_iter=1 可以最及时释放
#   4. 注意：当前代码不支持 Pipeline Parallelism (PPO)
# - max_seq_len 约束说明：
#   * 最小约束：至少需要 time_horizon * action_dim = 5 * 7 = 35 个 tokens（仅 action）
#   * 实际最小：建议至少 256-512，以确保包含图像、文本和 action tokens
#   * 最大约束：理论上可达 4096，但受显存限制
#   * 显存影响：序列长度对显存影响巨大（注意力复杂度 O(n²)，激活值占用 O(n)）
#   * 当前设置：4096（显存占用非常大！建议减小到 2048 或更小，如果使用 LoRA 可以适当增大）
#   * OOM 问题：如果出现 OOM，优先减小 max_seq_len
# - num_workers: 数据加载的并行进程数（当前为8）
#   * 主要影响 CPU 内存，对 GPU 显存影响很小
#   * 减小可以节省 CPU 内存，但可能降低数据加载速度
#   * 如果显存不足，优先考虑减小 max_seq_len、batch_size 或增加 GPU 数量


# Environment Variables
ARG_WORLD_SIZE=${1:-1}
# Auto-detect available GPU count if not specified as second argument
if [ -z "$2" ]; then
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ $? -eq 0 ] && [ $AVAILABLE_GPUS -gt 0 ]; then
        ARG_NPROC_PER_NODE=$AVAILABLE_GPUS
        echo "Auto-detected $AVAILABLE_GPUS GPU(s), using NPROC_PER_NODE=$ARG_NPROC_PER_NODE"
    else
        ARG_NPROC_PER_NODE=5  # fallback to 5 if detection fails
        echo "Warning: Could not detect GPU count, using default NPROC_PER_NODE=$ARG_NPROC_PER_NODE"
    fi
else
    ARG_NPROC_PER_NODE=$2
fi
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "master_addr: $MASTER_ADDR"
echo "master_port: $MASTER_PORT"
echo "rank: $RANK"

lr=5e-6
wd=0.15
dropout=0.08
z_loss_weight=1e-5

# LoRA parameters
lora_r=8
lora_alpha=16
lora_dropout=0.05

# Wandb parameters (optional, set use_wandb to false or empty to disable)
use_wandb=true
wandb_project="rynnvla-libero-goal"
wandb_entity=""  # Leave empty to use default entity, or set to your wandb team/username
# Check if wandb is available when enabled
if [ "$use_wandb" = "true" ]; then
    if ! python -c "import wandb" 2>/dev/null; then
        echo "Warning: wandb is not installed. Install it with: pip install wandb"
        echo "Continuing without wandb logging..."
        use_wandb=false
    fi
fi

data_config_train=../configs/libero_goal/his_2_third_view_wrist_w_state_5_256_pretokenize.yaml
data_config_val_ind=../configs/libero_goal/his_2_third_view_wrist_w_state_5_256_pretokenize.yaml
data_config_val_ood=../configs/libero_goal/his_2_third_view_wrist_w_state_5_256_pretokenize.yaml
time_horizon=5

exp_name=his_2_third_view_wrist_w_state_5_256_abiw
output_dir=../outputs/libero_goal
mkdir -p "$output_dir"/"$exp_name"

# Set wandb environment variables if enabled
if [ "$use_wandb" = "true" ]; then
    export WANDB_PROJECT="${wandb_project}"
    export WANDB_NAME="${exp_name}"
    if [ -n "$wandb_entity" ]; then
        export WANDB_ENTITY="${wandb_entity}"
    fi
    export WANDB_DIR="${output_dir}/${exp_name}"
    echo "Wandb enabled: project=${wandb_project}, name=${exp_name}${wandb_entity:+, entity=${wandb_entity}}"
else
    # Clear wandb environment variables if disabled
    unset WANDB_PROJECT
    unset WANDB_NAME
    unset WANDB_ENTITY
    unset WANDB_DIR
fi

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=30001 ../pretrain_solver_awm_w_ck_action_head.py \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK ../pretrain_solver_awm_w_ck_action_head.py \
--train_only True \
--disable_length_clustering \
--init_from ../ckpts/starting_point \
--tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
--ablation 0 \
--model_size 7B \
--batch_size 2 \
--accum_iter 4 \
--epochs 40 \
--warmup_epochs 0.01 \
--lr ${lr} \
--min_lr ${lr} \
--wd ${wd} \
--clip_grad 4 \
--action_dim 7 \
--time_horizon $time_horizon \
--data_config_train $data_config_train \
--data_config_val_ind $data_config_val_ind \
--data_config_val_ood $data_config_val_ood \
--num_workers 8 \
--output_dir "$output_dir"/"$exp_name" \
--checkpointing \
--max_seq_len 4096 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
--use_lora \
--lora_r ${lora_r} \
--lora_alpha ${lora_alpha} \
--lora_dropout ${lora_dropout} \
--ckpt_max_keep 0 \
2>&1 | tee -a "$output_dir"/"$exp_name"/output.log

echo "exp name: $exp_name" 