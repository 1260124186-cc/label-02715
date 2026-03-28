"""
PPO (Proximal Policy Optimization) 强化学习算法实现
环境: CartPole-v1 (倒立摆)

PPO是一种策略梯度方法，通过限制策略更新的幅度来保证训练稳定性。
主要特点:
1. 使用裁剪的目标函数限制策略更新
2. 使用Actor-Critic架构
3. 支持多步采样和小批量更新
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime

# ==================== 日志配置 ====================

def get_logger(name='ppo'):
    """获取日志器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger('ppo')


# ==================== 神经网络模型 ====================

class ActorCritic(nn.Module):
    """
    Actor-Critic网络
    - Actor: 输出动作的概率分布
    - Critic: 估计状态价值函数V(s)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor头: 输出动作概率
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic头: 输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """前向传播"""
        features = self.shared(state)
        return features

    def get_action_probs(self, state):
        """获取动作概率分布"""
        features = self.forward(state)
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs

    def get_value(self, state):
        """获取状态价值"""
        features = self.forward(state)
        value = self.critic(features)
        return value

    def evaluate(self, state, action):
        """
        评估给定状态-动作对
        返回: 动作对数概率, 状态价值, 熵
        """
        features = self.forward(state)

        # 计算动作概率
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        # 计算对数概率
        action_log_probs = dist.log_prob(action)

        # 计算熵(用于鼓励探索)
        entropy = dist.entropy()

        # 计算状态价值
        value = self.critic(features)

        return action_log_probs, value.squeeze(-1), entropy


# ==================== 经验缓冲区 ====================

class RolloutBuffer:
    """
    存储采样轨迹的缓冲区
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        """存储一步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get_batch(self):
        """获取所有数据"""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.dones)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.values))
        )


# ==================== PPO算法 ====================

class PPO:
    """
    PPO算法实现
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64
    ):
        """
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵损失系数
            max_grad_norm: 梯度裁剪阈值
            update_epochs: 每次更新的epoch数
            batch_size: 小批量大小
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # 创建Actor-Critic网络
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 经验缓冲区
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        """
        根据当前策略选择动作
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy.get_action_probs(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.policy.get_value(state_tensor)

        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算广义优势估计(GAE)
        GAE平衡了偏差和方差的权衡
        """
        advantages = []
        gae = 0

        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD误差
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)

        # 计算回报(用于价值函数训练)
        returns = advantages + torch.FloatTensor(values)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, next_value):
        """
        使用收集的经验更新策略
        """
        # 获取缓冲区数据
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get_batch()

        # 计算GAE和回报
        advantages, returns = self.compute_gae(
            rewards.numpy(), values.numpy(), dones.numpy(), next_value
        )

        # 多个epoch更新
        dataset_size = len(states)

        for _ in range(self.update_epochs):
            # 随机打乱索引
            indices = np.random.permutation(dataset_size)

            # 小批量更新
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # 获取小批量数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 评估当前策略
                new_log_probs, new_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )

                # 计算比率 r(θ) = π(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages

                # 策略损失(取最小值)
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(new_values, batch_returns)

                # 熵损失(鼓励探索)
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

        # 清空缓冲区
        self.buffer.clear()


# ==================== 训练函数 ====================

def train_ppo(
    env_name='CartPole-v1',
    max_episodes=500,
    max_steps=500,
    update_interval=2048,
    print_interval=10,
    output_dir='/app/output'
):
    """
    训练PPO智能体

    参数:
        env_name: 环境名称
        max_episodes: 最大训练回合数
        max_steps: 每回合最大步数
        update_interval: 更新间隔(采样步数)
        print_interval: 打印间隔
        output_dir: 输出目录
    """
    logger.info("=" * 50)
    logger.info("开始PPO训练")
    logger.info("=" * 50)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 创建环境
    try:
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        logger.info(f"环境创建成功: {env_name}")
        logger.info(f"状态空间维度: {state_dim}")
        logger.info(f"动作空间维度: {action_dim}")
    except Exception as e:
        logger.error(f"环境创建失败: {env_name}, 错误: {str(e)}")
        raise

    # 创建PPO智能体
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=10,
        batch_size=64
    )

    # 训练记录
    episode_rewards = []
    avg_rewards = []
    total_steps = 0
    solved = False
    solved_episode = None

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.buffer.store(state, action, reward, done, log_prob, value)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # 达到更新间隔时更新策略
            if total_steps % update_interval == 0:
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    next_value = agent.policy.get_value(next_state_tensor).item()
                agent.update(next_value)

            if done:
                break

        # 记录奖励
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)

        # 打印训练信息
        if (episode + 1) % print_interval == 0:
            logger.info(f"回合 {episode + 1:4d} | "
                       f"奖励: {episode_reward:6.1f} | "
                       f"平均奖励(100): {avg_reward:6.1f} | "
                       f"总步数: {total_steps}")

        # 检查是否解决
        if avg_reward >= 475 and not solved:
            logger.info(f"环境已解决! 回合 {episode + 1}, 平均奖励: {avg_reward:.1f}")
            solved = True
            solved_episode = episode + 1

    # 处理训练结束后剩余的经验数据
    if len(agent.buffer.states) > 0:
        logger.info(f"处理训练结束后剩余的 {len(agent.buffer.states)} 步经验数据...")
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = agent.policy.get_value(next_state_tensor).item()
        agent.update(next_value)
        total_steps += len(agent.buffer.states)
        logger.info("剩余经验数据处理完成")

    env.close()
    logger.info("训练环境已关闭")

    # 保存训练结果
    results = {
        'env_name': env_name,
        'total_episodes': len(episode_rewards),
        'total_steps': total_steps,
        'final_avg_reward': float(avg_rewards[-1]) if avg_rewards else 0,
        'max_reward': float(max(episode_rewards)) if episode_rewards else 0,
        'solved': solved,
        'solved_episode': solved_episode,
        'timestamp': datetime.now().isoformat()
    }

    # 保存JSON结果
    results_path = os.path.join(output_dir, 'training_results.json')
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"训练结果已保存到: {results_path}")
    except IOError as e:
        logger.error(f"保存训练结果失败: {str(e)}")
        raise

    # 绘制并保存训练曲线
    try:
        plot_path = plot_training_results(episode_rewards, avg_rewards, output_dir)
        logger.info(f"训练曲线图已保存到: {plot_path}")
    except Exception as e:
        logger.error(f"保存训练曲线图失败: {str(e)}")

    logger.info("=" * 50)
    logger.info(f"训练完成! 总回合: {len(episode_rewards)}, 最终平均奖励: {results['final_avg_reward']:.1f}")
    logger.info("=" * 50)

    return agent, episode_rewards, avg_rewards, results


def plot_training_results(episode_rewards, avg_rewards, output_dir='/app/output'):
    """
    绘制训练结果
    """
    logger.debug("开始绘制训练曲线图...")

    plt.figure(figsize=(12, 5))

    # 绘制每回合奖励
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, label='回合奖励')
    plt.plot(avg_rewards, color='red', linewidth=2, label='平均奖励(100回合)')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.title('PPO训练曲线 - CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制平均奖励
    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards, color='red', linewidth=2)
    plt.axhline(y=475, color='green', linestyle='--', label='解决阈值 (475)')
    plt.xlabel('回合')
    plt.ylabel('平均奖励')
    plt.title('平均奖励曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'ppo_training_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


# ==================== 主程序 ====================

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("PPO (Proximal Policy Optimization) 强化学习算法")
    logger.info("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    logger.info("随机种子已设置: 42")

    # 确定输出目录：优先使用环境变量，否则使用当前目录下的output文件夹
    output_dir = os.environ.get('OUTPUT_DIR', './output')

    try:
        # 训练智能体
        agent, episode_rewards, avg_rewards, results = train_ppo(
            env_name='CartPole-v1',
            max_episodes=500,
            max_steps=500,
            update_interval=2048,
            print_interval=10,
            output_dir=output_dir
        )

        logger.info("训练完成!")
        logger.info(f"最终平均奖励: {results['final_avg_reward']:.1f}")
        logger.info(f"是否解决: {results['solved']}")

    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}")
        raise
