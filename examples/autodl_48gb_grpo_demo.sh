#!/usr/bin/env bash
# =============================================================================
# verl GRPO Demo - 针对 48GB vGPU 优化
# GRPO (Group Relative Policy Optimization) 是 verl 的核心 RL 算法
# =============================================================================

set -xeuo pipefail

echo "=========================================="
echo "verl GRPO Demo (48GB vGPU 优化版)"
echo "=========================================="

# ==================== 1. 环境检查 ====================
echo ""
echo ">>> 1. 检查环境"

if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到 nvidia-smi"
    exit 1
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPU 数量: ${NUM_GPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_mem / 1024**3:.1f} GB')
"

# ==================== 2. 安装 verl ====================
echo ""
echo ">>> 2. 安装 verl"

if python3 -c "import verl" 2>/dev/null; then
    echo "verl 已安装"
else
    echo "安装 verl..."
    pip install verl
fi

python3 -c "import verl; print(f'verl 版本: {verl.__version__}')"

# ==================== 3. 准备数据 ====================
echo ""
echo ">>> 3. 准备数据"

DATA_DIR="$HOME/data/gsm8k"
mkdir -p "${DATA_DIR}"

if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ]; then
    echo "下载 GSM8K 数据集..."
    
    # 获取脚本所在目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [ -f "${SCRIPT_DIR}/data_preprocess/gsm8k.py" ]; then
        python3 "${SCRIPT_DIR}/data_preprocess/gsm8k.py" --local_save_dir "${DATA_DIR}"
    else
        python3 -c "
import os
import re
import datasets

print('下载 openai/gsm8k...')
dataset = datasets.load_dataset('openai/gsm8k', 'main')

def extract_solution(solution_str):
    solution = re.search(r'#### (\-?[0-9\.\,]+)', solution_str)
    return solution.group(1).replace(',', '')

def process(example, idx):
    question = example['question'] + ' Let\'s think step by step and output the final answer after \"####\".'
    solution = extract_solution(example['answer'])
    return {
        'data_source': 'openai/gsm8k',
        'prompt': [{'role': 'user', 'content': question}],
        'ability': 'math',
        'reward_model': {'style': 'rule', 'ground_truth': solution},
    }

train = dataset['train'].map(process, with_indices=True)
test = dataset['test'].map(process, with_indices=True)

os.makedirs('${DATA_DIR}', exist_ok=True)
train.to_parquet('${DATA_DIR}/train.parquet')
test.to_parquet('${DATA_DIR}/test.parquet')
print(f'训练集: {len(train)} 条')
print(f'测试集: {len(test)} 条')
"
    fi
else
    echo "数据已存在"
fi

# ==================== 4. 选择模型 ====================
echo ""
echo ">>> 4. 选择模型"

# 48GB vGPU 运行 GRPO 的推荐配置
# GRPO 需要同时加载 actor 和 critic，所以显存需求更高
# 48GB 可以跑 7B-8B 模型

echo "GRPO 推荐模型:"
echo "  1) Qwen/Qwen2.5-3B-Instruct   (约6GB，最安全)"
echo "  2) Qwen/Qwen2.5-7B-Instruct   (约14GB，推荐)"
echo "  3) Qwen/Qwen3-8B               (约16GB，最新)"

# 默认使用 Qwen2.5-7B
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
echo "使用模型: ${MODEL_PATH}"

# ==================== 5. 运行 GRPO 训练 ====================
echo ""
echo ">>> 5. 运行 GRPO 训练"

SAVE_DIR="$HOME/checkpoints/verl_grpo_demo"
mkdir -p "${SAVE_DIR}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 48GB vGPU GRPO 优化配置
# - 减小 batch size 以适应显存
# - 使用 gradient checkpointing
# - 适当减小序列长度

python3 -m verl.trainer.main_ppo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=16384 \
    critic.fsdp.param_offload=False \
    critic.fsdp.optimizer_offload=False \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="autodl_grpo_demo_48gb" \
    trainer.experiment_name="grpo_$(date +%Y%m%d_%H%M)" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync

# ==================== 6. 完成 ====================
echo ""
echo "=========================================="
echo "GRPO Demo 完成！"
echo "=========================================="
echo ""
echo "检查点保存在: ${SAVE_DIR}"
echo ""
echo "查看检查点:"
ls -la "${SAVE_DIR}"
echo ""
echo "下一步可以尝试："
echo "1. 增加训练轮数"
echo "2. 调整学习率"
echo "3. 尝试不同的算法（如 PPO、DAPO）"
