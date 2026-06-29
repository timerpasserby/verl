#!/usr/bin/env bash
# =============================================================================
# verl AutoDL Demo Script
# 用于在 AutoDL GPU 服务器上快速运行 verl 训练示例
# =============================================================================

set -xeuo pipefail

# ==================== 配置区域 ====================
# AutoDL 服务器通常有 1-8 张 GPU
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
echo "检测到 ${NUM_GPUS} 张 GPU"

# 使用小模型以快速验证（0.5B 参数）
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}

# 使用小数据集
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k"}
SAVE_DIR=${SAVE_DIR:-"$HOME/checkpoints/verl_demo"}

# 训练参数（小规模快速验证）
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# 实验名称
PROJECT_NAME=${PROJECT_NAME:-"verl_autodl_demo"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"sft_demo_$(date +%Y%m%d_%H%M)"}

echo "=========================================="
echo "verl AutoDL Demo 配置"
echo "=========================================="
echo "GPU 数量: ${NUM_GPUS}"
echo "模型: ${MODEL_PATH}"
echo "数据目录: ${DATA_DIR}"
echo "保存目录: ${SAVE_DIR}"
echo "训练轮数: ${TOTAL_EPOCHS}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "=========================================="

# ==================== 步骤 1: 环境准备 ====================
echo ""
echo ">>> 步骤 1: 检查环境"

# 检查 Python 版本
python3 --version

# 检查 PyTorch 和 CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 检查 verl 是否已安装
if python3 -c "import verl" 2>/dev/null; then
    echo "verl 已安装"
    python3 -c "import verl; print(f'verl version: {verl.__version__}')"
else
    echo "verl 未安装，正在安装..."
    
    # 获取脚本所在目录（verl 仓库根目录）
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # 如果在 verl 仓库内，直接安装
    if [ -f "${SCRIPT_DIR}/setup.py" ] || [ -f "${SCRIPT_DIR}/pyproject.toml" ]; then
        echo "从本地安装 verl..."
        cd "${SCRIPT_DIR}"
        pip install -e ".[test]"
    else
        echo "从 PyPI 安装 verl..."
        pip install verl
    fi
fi

# ==================== 步骤 2: 数据准备 ====================
echo ""
echo ">>> 步骤 2: 准备数据"

mkdir -p "${DATA_DIR}"

# 检查数据是否已存在
if [ -f "${DATA_DIR}/train.parquet" ] && [ -f "${DATA_DIR}/test.parquet" ]; then
    echo "数据已存在，跳过下载"
    echo "训练集大小: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('${DATA_DIR}/train.parquet')))")"
    echo "测试集大小: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('${DATA_DIR}/test.parquet')))")"
else
    echo "下载并预处理 GSM8K 数据集..."
    
    # 获取 verl 仓库根目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # 检查是否在 verl 仓库内
    if [ -f "${SCRIPT_DIR}/examples/data_preprocess/gsm8k.py" ]; then
        python3 "${SCRIPT_DIR}/examples/data_preprocess/gsm8k.py" \
            --local_save_dir "${DATA_DIR}"
    else
        # 直接使用 verl 中的数据预处理
        python3 -c "
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs

data_source = 'openai/gsm8k'
print(f'下载 {data_source} 数据集...')
dataset = datasets.load_dataset(data_source, 'main')

import re

def extract_solution(solution_str):
    solution = re.search(r'#### (\-?[0-9\.\,]+)', solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop('question')
        question = question_raw + ' Let\\'s think step by step and output the final answer after \"####\".'
        answer_raw = example.pop('answer')
        solution = extract_solution(answer_raw)
        data = {
            'data_source': data_source,
            'prompt': [{'role': 'user', 'content': question}],
            'ability': 'math',
            'reward_model': {'style': 'rule', 'ground_truth': solution},
            'extra_info': {'split': split, 'index': idx, 'answer': answer_raw, 'question': question_raw},
        }
        return data
    return process_fn

train_dataset = dataset['train'].map(function=make_map_fn('train'), with_indices=True)
test_dataset = dataset['test'].map(function=make_map_fn('test'), with_indices=True)

os.makedirs('${DATA_DIR}', exist_ok=True)
train_dataset.to_parquet(os.path.join('${DATA_DIR}', 'train.parquet'))
test_dataset.to_parquet(os.path.join('${DATA_DIR}', 'test.parquet'))
print('数据准备完成！')
print(f'训练集大小: {len(train_dataset)}')
print(f'测试集大小: {len(test_dataset)}')
"
    fi
fi

# ==================== 步骤 3: 创建演示脚本 ====================
echo ""
echo ">>> 步骤 3: 创建 SFT 训练演示脚本"

mkdir -p "${SAVE_DIR}"

cat > "${SAVE_DIR}/run_sft_demo.sh" << 'EOF'
#!/usr/bin/env bash
# verl SFT 训练演示脚本
set -xeuo pipefail

# 配置参数
NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k"}
SAVE_DIR=${SAVE_DIR:-"$HOME/checkpoints/verl_demo"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
PROJECT_NAME=${PROJECT_NAME:-"verl_autodl_demo"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"sft_demo_$(date +%Y%m%d_%H%M)"}

echo "开始 SFT 训练..."
echo "模型: ${MODEL_PATH}"
echo "GPU 数量: ${NUM_GPUS}"

# 运行 SFT 训练
torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    optim.lr=1e-4 \
    engine=fsdp \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console"]' \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    "$@"

echo "训练完成！"
echo "检查点保存在: ${SAVE_DIR}"
EOF

chmod +x "${SAVE_DIR}/run_sft_demo.sh"
echo "演示脚本已创建: ${SAVE_DIR}/run_sft_demo.sh"

# ==================== 步骤 4: 运行训练 ====================
echo ""
echo ">>> 步骤 4: 运行 SFT 训练"
echo "开始训练（这可能需要几分钟到几十分钟，取决于 GPU 数量）..."
echo ""

# 设置环境变量
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

# 运行训练
bash "${SAVE_DIR}/run_sft_demo.sh"

# ==================== 步骤 5: 验证结果 ====================
echo ""
echo ">>> 步骤 5: 验证训练结果"

# 检查检查点
if [ -d "${SAVE_DIR}" ]; then
    echo "检查点目录内容:"
    ls -la "${SAVE_DIR}"
    
    # 查找最新的检查点
    LATEST_CKPT=$(ls -td "${SAVE_DIR}"/*/ 2>/dev/null | head -1)
    if [ -n "${LATEST_CKPT}" ]; then
        echo ""
        echo "最新检查点: ${LATEST_CKPT}"
        echo "检查点内容:"
        ls -la "${LATEST_CKPT}"
    fi
fi

echo ""
echo "=========================================="
echo "verl AutoDL Demo 完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 查看训练日志"
echo "2. 使用检查点进行推理或继续训练"
echo "3. 尝试不同的配置（模型、超参数等）"
echo ""
echo "常用命令："
echo "  # 查看 GPU 使用情况"
echo "  nvidia-smi"
echo ""
echo "  # 使用检查点进行推理"
echo "  python3 -c \"from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('${SAVE_DIR}/global_step_*/actor\")"
echo ""
echo "  # 继续训练"
echo "  bash ${SAVE_DIR}/run_sft_demo.sh trainer.resume_mode=resume_path trainer.resume_from_path=${SAVE_DIR}/global_step_*"
echo ""
