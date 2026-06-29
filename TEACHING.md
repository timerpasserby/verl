# verl 项目教学指南

## 1. 项目概述

**verl** (Volcano Engine Reinforcement Learning for LLMs) 是一个开源的强化学习训练库，专门用于大型语言模型(LLM)的后训练。由字节跳动 Seed 团队发起，实现了 HybridFlow 编程模型（已被 EuroSys 2025 接收）。

### 核心特点
- **灵活的 RL 算法支持**：PPO、GRPO、DAPO、GSPO、ReMax、REINFORCE++、RLOO、PRIME、DrGRPO 等
- **多后端训练**：FSDP、Megatron-LM、VeOmni、TorchTitan
- **多推理引擎**：vLLM、SGLang、HF Transformers
- **大规模扩展**：支持 671B 参数模型，数百个 GPU
- **多模态支持**：VLM（视觉语言模型）RL 训练

## 2. 核心架构

### 2.1 数据流架构
verl 采用 **HybridFlow** 编程模型，核心思想是将计算和数据依赖解耦：

```
┌─────────────────────────────────────────────────────────────┐
│                    Driver Process (Ray)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Actor     │  │   Critic    │  │   Reward    │        │
│  │   Worker    │  │   Worker    │  │   Worker    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │               │               │                   │
│         └───────────────┼───────────────┘                   │
│                         │                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              DataProto (TensorDict)                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

#### DataProto (`verl/protocol.py`)
- **核心数据结构**：基于 TensorDict 的数据容器
- **功能**：在不同 worker 之间传输数据
- **特点**：支持自动填充、序列打包、分布式通信

#### Engine Workers (`verl/workers/engine/`)
- **FSDP**：PyTorch 原生分布式训练
- **Megatron-LM**：大规模模型训练
- **VeOmni**：统一的模型引擎
- **TorchTitan**：PyTorch 原生分布式

#### Rollout Workers (`verl/workers/rollout/`)
- **vLLM**：高性能推理引擎
- **SGLang**：结构化生成语言
- **HF Transformers**：HuggingFace 原生推理

#### Trainer (`verl/trainer/`)
- **PPO Trainer**：主要训练逻辑
- **Hydra 配置**：灵活的配置管理

## 3. 代码结构详解

### 3.1 主要目录
```
verl/
├── __init__.py          # 包初始化，插件系统
├── protocol.py          # DataProto 核心数据结构
├── trainer/             # 训练逻辑
│   ├── config/          # Hydra 配置文件
│   ├── main_ppo.py      # PPO 训练入口
│   └── ppo/             # PPO 算法实现
├── workers/             # Worker 实现
│   ├── engine/          # 训练引擎 (FSDP, Megatron)
│   ├── rollout/         # 推理引擎 (vLLM, SGLang)
│   └── reward_manager/  # 奖励管理
├── models/              # 模型定义
├── experimental/        # 实验性功能
└── utils/               # 工具函数
```

### 3.2 训练流程

#### PPO 训练流程
```python
# 1. 初始化
config = load_config()  # Hydra 配置
ray.init()              # Ray 集群初始化

# 2. 创建 Worker
actor_worker = ActorWorker(config)
critic_worker = CriticWorker(config)
rollout_worker = RolloutWorker(config)

# 3. 训练循环
for epoch in range(total_epochs):
    # 3.1 Rollout：生成响应
    responses = rollout_worker.generate(prompts)
    
    # 3.2 计算奖励
    rewards = reward_manager.compute(responses)
    
    # 3.3 计算优势
    advantages = compute_advantages(rewards, values)
    
    # 3.4 更新 Actor
    actor_loss = actor_worker.update(advantages)
    
    # 3.5 更新 Critic
    critic_loss = critic_worker.update(rewards)
```

## 4. 使用指南

### 4.1 安装
```bash
# 基础安装
pip install -e .

# 带测试依赖
pip install -e ".[test]"

# 带 vLLM 后端
pip install -e ".[test,vllm]"

# 带 SGLang 后端
pip install -e ".[test,sglang]"
```

### 4.2 运行 PPO 训练
```bash
# 使用示例脚本
cd examples/ppo_trainer
bash run_qwen3_8b_fsdp.sh

# 自定义参数
python3 -m verl.trainer.main_ppo \
    data.train_files="['path/to/data']" \
    actor_rollout_ref.model.path="Qwen/Qwen3-8B" \
    trainer.total_epochs=10
```

### 4.3 运行 GRPO 训练
```bash
cd examples/grpo_trainer
bash run_qwen3_8b_fsdp.sh
```

## 5. 配置系统

### 5.1 Hydra 配置
verl 使用 Hydra 进行配置管理，配置文件位于 `verl/trainer/config/`。

#### 主要配置文件
- `ppo_trainer.yaml`：PPO 训练主配置
- `actor/`：Actor 模型配置
- `critic/`：Critic 模型配置
- `rollout/`：Rollout 配置
- `algorithm/`：算法配置

#### 配置覆盖
```bash
# 命令行覆盖
python3 -m verl.trainer.main_ppo \
    trainer.total_epochs=20 \
    actor_rollout_ref.actor.optim.lr=1e-5

# 环境变量
export VERL_LOGGING_LEVEL=DEBUG
```

### 5.2 生成配置
```bash
# 生成参考配置
scripts/generate_trainer_config.sh

# 打印当前配置
python3 scripts/print_cfg.py --cfg job
```

## 6. 开发工作流

### 6.1 环境设置
```bash
# 使用 uv（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

# 安装依赖
uv pip install pre-commit hydra-core
pre-commit install
```

### 6.2 代码质量
```bash
# 运行 pre-commit
pre-commit run --all-files

# 运行特定检查
pre-commit run --all-files ruff
pre-commit run --all-files mypy
```

### 6.3 测试
```bash
# CPU 测试
pytest tests/ -k "on_cpu"

# GPU 测试
pytest tests/ -k "not on_cpu"

# 特定测试文件
pytest tests/trainer/test_specific.py

# 健全性检查
pytest tests/special_sanity/
```

## 7. 常见模式

### 7.1 添加新算法
1. 在 `verl/trainer/ppo/` 中创建新算法文件
2. 在 `verl/trainer/config/algorithm/` 中添加配置
3. 在 `examples/` 中添加示例脚本
4. 更新 `verl/trainer/config/ppo_trainer.yaml` 中的默认配置

### 7.2 添加新模型引擎
1. 在 `verl/workers/engine/` 中创建新引擎目录
2. 实现 `base.py` 中定义的接口
3. 在 `verl/trainer/config/model_engine/` 中添加配置
4. 更新配置系统支持新引擎

### 7.3 添加新推理引擎
1. 在 `verl/workers/rollout/` 中创建新引擎目录
2. 实现 `base.py` 中定义的接口
3. 在 `verl/trainer/config/rollout/` 中添加配置

## 8. 调试技巧

### 8.1 日志级别
```bash
# 设置日志级别
export VERL_LOGGING_LEVEL=DEBUG

# 或在代码中
import logging
logging.getLogger(__name__).setLevel(logging.DEBUG)
```

### 8.2 性能分析
```bash
# 使用 nsys 进行性能分析
python3 -m verl.trainer.main_ppo \
    global_profiler.tool=nsys \
    global_profiler.steps=[0,1,2]

# 内存分析
python3 -m verl.trainer.main_ppo \
    global_profiler.tool=torch_memory
```

### 8.3 常见问题
- **OOM 错误**：减小 batch size 或启用 gradient checkpointing
- **NCCL 超时**：增加 `actor_rollout_ref.nccl_timeout` 值
- **配置错误**：使用 `python3 scripts/print_cfg.py` 验证配置

## 9. 扩展资源

### 9.1 文档
- [官方文档](https://verl.readthedocs.io/)
- [API 文档](https://verl.readthedocs.io/en/latest/)
- [示例代码](examples/)

### 9.2 社区
- [GitHub 仓库](https://github.com/verl-project/verl)
- [Slack 频道](https://join.slack.com/t/verl-project/shared_invite/zt-3c6mc2khw-v0lo6NfDPuFP6OnkrZwfqw)
- [微信群](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG)

### 9.3 相关论文
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [DAPO: An Open-Source LLM Reinforcement Learning System](https://dapo-sia.github.io/)

## 10. AutoDL GPU 服务器快速开始

### 10.1 极简 Demo（推荐新手）
```bash
# 1. 克隆仓库
git clone https://github.com/verl-project/verl.git
cd verl

# 2. 运行极简 Demo（一键完成）
bash examples/autodl_quick_demo.sh
```

### 10.2 完整 Demo
```bash
# 运行完整 Demo（包含更多配置选项）
bash examples/autodl_demo.sh
```

### 10.3 自定义配置
```bash
# 使用更多 GPU
NUM_GPUS=4 bash examples/autodl_demo.sh

# 使用不同模型
MODEL_PATH="Qwen/Qwen3-8B" bash examples/autodl_demo.sh

# 调整训练参数
TOTAL_EPOCHS=3 TRAIN_BATCH_SIZE=64 bash examples/autodl_demo.sh
```

### 10.4 输出说明
训练完成后，检查点保存在 `~/checkpoints/verl_demo/` 目录下。

## 11. 实践练习

### 练习 1：运行第一个训练
1. 安装 verl
2. 运行 AutoDL 极简 Demo
3. 观察训练日志和指标

### 练习 2：修改配置
1. 修改 `examples/autodl_demo.sh` 中的参数
2. 运行训练并观察变化
3. 尝试不同的超参数组合

### 练习 3：添加自定义奖励函数
1. 在 `verl/workers/reward_manager/` 中创建新文件
2. 实现自定义奖励函数
3. 在配置中使用新奖励函数

### 练习 4：调试训练问题
1. 使用 `VERL_LOGGING_LEVEL=DEBUG` 运行训练
2. 分析日志中的错误信息
3. 使用性能分析工具找出瓶颈

---

*本指南基于 verl 0.9.0.dev 版本编写，具体实现可能随版本更新而变化。*
