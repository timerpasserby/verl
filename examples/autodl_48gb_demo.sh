#!/usr/bin/env bash
# =============================================================================
# verl AutoDL Demo - 针对 48GB vGPU 优化
# =============================================================================

set -xeuo pipefail

echo "=========================================="
echo "verl AutoDL Demo (48GB vGPU 优化版)"
echo "=========================================="

# ==================== 1. 环境检查 ====================
echo ""
echo ">>> 1. 检查环境"

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到 nvidia-smi，请确保在 GPU 服务器上运行"
    exit 1
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPU 数量: ${NUM_GPUS}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 检查 Python 和 PyTorch
python3 -c "
import torch
print(f'Python: {__import__(\"sys\").version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda}')
print(f'GPU 数量: {torch.cuda.device_count()}')
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

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "下载 GSM8K 数据集..."
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
    question = example['question'] + ' Let\'s think step by step.'
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
else
    echo "数据已存在"
fi

# ==================== 4. 选择模型 ====================
echo ""
echo ">>> 4. 选择模型"

# 48GB vGPU 可以跑 8B 模型，甚至 14B 模型（需要优化）
# 这里提供几个选项

echo "可用模型选项:"
echo "  1) Qwen/Qwen2.5-0.5B-Instruct  (最简单，<1GB)"
echo "  2) Qwen/Qwen2.5-3B-Instruct    (约6GB)"
echo "  3) Qwen/Qwen2.5-7B-Instruct    (约14GB) - 推荐"
echo "  4) Qwen/Qwen3-8B                (约16GB) - 最新模型"

# 默认使用 Qwen2.5-7B，适合 48GB vGPU
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
echo "使用模型: ${MODEL_PATH}"

# ==================== 5. 运行 SFT 训练 ====================
echo ""
echo ">>> 5. 运行 SFT 训练"

SAVE_DIR="$HOME/checkpoints/verl_demo"
mkdir -p "${SAVE_DIR}"

# 48GB vGPU 优化配置
# - micro_batch_size: 8 (48GB 可以处理更大的 batch)
# - max_length: 1024 (更长的序列)
# - gradient_checkpointing: 启用以节省显存

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=8 \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    optim.lr=1e-5 \
    engine=fsdp \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name="autodl_demo_48gb" \
    trainer.experiment_name="sft_$(date +%Y%m%d_%H%M)" \
    trainer.logger='["console"]' \
    trainer.total_epochs=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5

# ==================== 6. 完成 ====================
echo ""
echo "=========================================="
echo "Demo 完成！"
echo "=========================================="
echo ""
echo "检查点保存在: ${SAVE_DIR}"
echo ""
echo "查看检查点:"
ls -la "${SAVE_DIR}"
echo ""
echo "下一步可以尝试："
echo "1. 运行 GRPO 训练"
echo "2. 使用更大的模型（如 Qwen3-8B）"
echo "3. 调整超参数"
