# verl AutoDL Demo

在 AutoDL GPU 服务器上快速运行 verl 训练示例。

## 快速开始

### 方法一：极简 Demo（推荐新手）

```bash
# 1. 克隆 verl 仓库
git clone https://github.com/verl-project/verl.git
cd verl

# 2. 运行极简 Demo
bash examples/autodl_quick_demo.sh
```

### 方法二：完整 Demo

```bash
# 1. 克隆 verl 仓库
git clone https://github.com/verl-project/verl.git
cd verl

# 2. 运行完整 Demo
bash examples/autodl_demo.sh
```

## 自定义配置

通过环境变量自定义训练参数：

```bash
# 使用更多 GPU
NUM_GPUS=4 bash examples/autodl_demo.sh

# 使用不同模型
MODEL_PATH="Qwen/Qwen3-8B" bash examples/autodl_demo.sh

# 调整训练参数
TOTAL_EPOCHS=3 TRAIN_BATCH_SIZE=64 bash examples/autodl_demo.sh
```

## 前置要求

- AutoDL GPU 服务器（推荐至少 1 张 GPU）
- Python 3.10+
- CUDA 11.8+

## 输出说明

训练完成后，检查点保存在 `~/checkpoints/verl_demo/` 目录下：

```
~/checkpoints/verl_demo/
├── global_step_10/
│   ├── actor/          # Actor 模型检查点
│   └── critic/         # Critic 模型检查点（如果使用 PPO）
└── config.yaml         # 训练配置
```

## 常见问题

### Q: 安装 verl 失败怎么办？

```bash
# 使用国内镜像
pip install verl -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或从源码安装
git clone https://github.com/verl-project/verl.git
cd verl
pip install -e ".[test]"
```

### Q: GPU 内存不足？

```bash
# 减小 batch size
MICRO_BATCH_SIZE=1 TRAIN_BATCH_SIZE=8 bash examples/autodl_demo.sh

# 使用更小的模型
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" bash examples/autodl_demo.sh
```

### Q: 如何查看训练日志？

```bash
# 实时查看
tail -f ~/checkpoints/verl_demo/*/trainer.log

# 或使用 tensorboard
tensorboard --logdir ~/checkpoints/verl_demo/
```

### Q: 如何使用训练好的模型？

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载检查点
model_path = "~/checkpoints/verl_demo/global_step_10/actor"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 推理
inputs = tokenizer("What is 2+2?", return_tensors="outputs")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## 进阶使用

### 运行 GRPO 训练

```bash
# 需要先准备数据
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# 运行 GRPO
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
```

### 运行 PPO 训练

```bash
# 运行 PPO
bash examples/ppo_trainer/run_qwen3_8b_fsdp.sh
```

## 相关资源

- [verl 文档](https://verl.readthedocs.io/)
- [GitHub 仓库](https://github.com/verl-project/verl)
- [Slack 社区](https://join.slack.com/t/verl-project/shared_invite/zt-3c6mc2khw-v0lo6NfDPuFP6OnkrZwfqw)

## 许可证

Apache License 2.0
