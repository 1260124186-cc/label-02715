# PPO强化学习算法项目

## How to Run

### 方式一：本地 Python 运行

```bash
# 1. 进入后端目录并安装依赖
cd backend
pip install -r requirements.txt

# 2. 直接运行PPO训练（命令行模式，默认500回合）
python ppo.py

# 3. 或启动API服务
python app.py
# 服务启动后访问 http://localhost:5001/
```

**本地API服务测试：**

```bash
# 健康检查
curl http://localhost:5001/health

# 开始训练（50回合快速测试）
curl -X POST http://localhost:5001/train \
  -H "Content-Type: application/json" \
  -d '{"max_episodes": 50}'

# 查看训练状态
curl http://localhost:5001/status

# 获取训练结果
curl http://localhost:5001/results
```

### 方式二：使用 Docker Compose 运行（推荐）

```bash
# 构建并启动服务
docker-compose up --build -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend

# 停止服务
docker-compose down
```

**Docker模式API测试：**

```bash
# 健康检查
curl http://localhost:8081/health

# 开始训练（默认300回合）
curl -X POST http://localhost:8081/train

# 自定义训练回合数
curl -X POST http://localhost:8081/train \
  -H "Content-Type: application/json" \
  -d '{"max_episodes": 500}'

# 查看训练状态
curl http://localhost:8081/status

# 获取训练结果JSON
curl http://localhost:8081/results

# 下载训练曲线图
curl http://localhost:8081/plot --output training_plot.png
```

### 验证镜像跨平台支持

```bash
# 验证ARM架构支持（Apple Silicon Mac）
docker pull --platform linux/arm64 python:3.11-slim

# 验证X86架构支持
docker pull --platform linux/amd64 python:3.11-slim
```

---

## Services

| 服务名 | 端口（Docker） | 端口（本地） | 描述 |
|--------|---------------|-------------|------|
| backend | 8081 | 5001 | PPO强化学习算法API服务 |

### API接口说明

| 接口 | 方法 | 描述 |
|------|------|------|
| `/` | GET | API说明和接口列表 |
| `/health` | GET | 健康检查 |
| `/status` | GET | 查看当前训练状态 |
| `/train` | POST | 开始训练（支持参数: max_episodes，默认300，最大10000） |
| `/results` | GET | 获取训练结果JSON |
| `/plot` | GET | 获取训练曲线图PNG |

---

## 测试账号

本项目为强化学习算法演示项目，无需登录账号。

---

## 题目内容

帮我生成一个强化学习PPO算法的案例，要求python代码

---

## 项目介绍

### 什么是PPO算法？

PPO（Proximal Policy Optimization，近端策略优化）是由OpenAI在2017年提出的一种策略梯度强化学习算法。它是目前最流行的强化学习算法之一，因其实现简单、性能稳定而被广泛应用。

### 核心原理

PPO的核心思想是限制策略更新的幅度，避免因为更新步长过大导致性能崩溃。它通过裁剪目标函数来实现这一点：

```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
```

其中：
- `r(θ) = π(a|s) / π_old(a|s)` 是新旧策略的概率比
- `A` 是优势函数
- `ε` 是裁剪参数（通常为0.2）

### 主要组件

1. **Actor-Critic网络**: 共享特征提取层，分别输出动作概率和状态价值
2. **GAE (广义优势估计)**: 平衡偏差和方差的优势函数估计方法
3. **裁剪目标函数**: 限制策略更新幅度，保证训练稳定性

### 环境说明

本案例使用 **CartPole-v1** 环境：
- **目标**: 通过左右移动小车来平衡杆子
- **状态空间**: 4维（位置、速度、角度、角速度）
- **动作空间**: 2个离散动作（左、右）
- **解决条件**: 连续100回合平均奖励 ≥ 475

### 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lr | 3e-4 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE参数 |
| clip_epsilon | 0.2 | PPO裁剪参数 |
| value_coef | 0.5 | 价值损失系数 |
| entropy_coef | 0.01 | 熵损失系数 |
| update_epochs | 10 | 每次更新的epoch数 |
| batch_size | 64 | 小批量大小 |

### 项目结构

```
.
├── backend/                 # 后端服务
│   ├── Dockerfile          # Docker构建文件（支持ARM和X86跨平台）
│   ├── requirements.txt    # Python依赖
│   ├── app.py              # Flask API服务（含结构化日志和错误处理）
│   └── ppo.py              # PPO算法核心实现
├── docker-compose.yml      # Docker Compose配置
├── .gitignore              # Git忽略文件
└── README.md               # 项目说明
```

### 预期结果

训练约200-300回合后，智能体应该能够稳定地平衡杆子，达到接近500的最大奖励。

### 日志输出示例

```
2026-02-12 23:09:39 - ppo - INFO - 开始PPO训练
2026-02-12 23:09:39 - ppo - INFO - 环境创建成功: CartPole-v1
2026-02-12 23:09:39 - ppo - INFO - 状态空间维度: 4
2026-02-12 23:09:39 - ppo - INFO - 动作空间维度: 2
2026-02-12 23:09:40 - ppo - INFO - 回合   10 | 奖励:   19.0 | 平均奖励(100):   18.8 | 总步数: 188
...
2026-02-12 23:09:40 - ppo - INFO - 训练完成! 总回合: 50, 最终平均奖励: 19.4
```
