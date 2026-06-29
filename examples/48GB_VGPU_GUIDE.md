# verl 48GB vGPU 快速开始指南

针对 AutoDL 48GB vGPU 服务器优化的 verl 训练指南。

## 快速开始

### 方式一：SFT 训练（推荐先跑这个验证环境）

```bash
# 1. 克隆仓库
git clone https://github.com/verl-project/verl.git
cd verl

# 2. 运行 SFT Demo（约 10-20 分钟）
bash examples/autodl_48gb_demo.sh
```

### 方式二：GRPO 训练（RL 训练）

```bash
# 运行 GRPO Demo（约 30-60 分钟）
bash examples/autodl_48gb_grpo_demo.sh
```

## 48GB vGPU 推荐配置

### 模型选择

| 模型 | 显存占用 | 推荐度 | 说明 |
|------|----------|--------|------|
| Qwen2.5-0.5B-Instruct | <1GB | ⭐⭐⭐ | 最简单，快速验证 |
| Qwen2.5-3B-Instruct | ~6GB | ⭐⭐⭐ | 性价比高 |
| Qwen2.5-7B-Instruct | ~14GB | ⭐⭐⭐⭐⭐ | **推荐**，效果好 |
| Qwen3-8B | ~16GB | ⭐⭐⭐⭐ | 最新模型 |

### SFT 训练参数

```bash
# 基础配置
model.path="Qwen/Qwen2.5-7B-Instruct"
data.micro_batch_size_per_gpu=8
data.train_batch_size=32
data.max_prompt_length=512
data.max_response_length=512
optim.lr=1e-5
trainer.total_epochs=1
```

### GRPO 训练参数

```bash
# 基础配置
model.path="Qwen/Qwen2.5-7B-Instruct"
data.train_batch_size=64
data.max_prompt_length=512
data.max_response_length=512
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
algorithm.adv_estimator=grpo
trainer.total_epochs=1
```

## 显存优化技巧

### 1. 启用 Gradient Checkpointing
```bash
model.enable_gradient_checkpointing=True
```

### 2. 使用 Dynamic Batching
```bash
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
```

### 3. 调整 GPU 内存利用率
```bash
# vLLM 推理引擎的显存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # 40% 显存用于推理

# FSDP 参数卸载（如果显存不够）
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

### 4. 减小序列长度
```bash
data.max_prompt_length=256
data.max_response_length=256
```

## 常见问题

### Q: GPU 内存不足 (OOM)

```bash
# 方案 1: 减小 batch size
data.micro_batch_size_per_gpu=4
data.train_batch_size=16

# 方案 2: 使用更小的模型
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

# 方案 3: 启用参数卸载
actor_rollout_ref.ref.fsdp_config.param_offload=True
critic.fsdp.param_offload=True
```

### Q: 训练速度太慢

```bash
# 方案 1: 增大 batch size（如果显存允许）
data.micro_batch_size_per_gpu=16
data.train_batch_size=64

# 方案 2: 使用序列并行
SP_SIZE=2  # 需要多 GPU
```

### Q: 如何查看训练进度？

```bash
# 实时查看 GPU 使用
watch -n 1 nvidia-smi

# 查看训练日志
tail -f ~/checkpoints/verl_*_demo/*/trainer.log
```

### Q: 如何使用训练好的模型？

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载检查点
model_path = "~/checkpoints/verl_demo/global_step_10/actor"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 推理
messages = [{"role": "user", "content": "What is 2+2?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

## 进阶用法

### 1. 使用 LoRA 节省显存

```bash
# 启用 LoRA
python3 -m verl.trainer.sft_trainer \
    model.path="Qwen/Qwen2.5-7B-Instruct" \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    ...
```

### 2. 多 GPU 训练

```bash
# 如果有多个 GPU
NUM_GPUS=2 bash examples/autodl_48gb_demo.sh
```

### 3. 自定义奖励函数

对于 GRPO 训练，可以使用自定义奖励函数：

```python
# 在 verl/workers/reward_manager/ 中创建新文件
def custom_reward_function(responses, **kwargs):
    # 自定义奖励逻辑
    rewards = []
    for response in responses:
        # 计算奖励
        reward = compute_reward(response)
        rewards.append(reward)
    return rewards
```

## 相关资源

- [verl 文档](https://verl.readthedocs.io/)
- [GRPO 算法说明](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [性能调优指南](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

## 许可证

Apache License 2.0
