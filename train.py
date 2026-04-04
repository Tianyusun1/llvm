import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

# 导入我们的新版组件
from dataloader import DAGDataset
from env import DAGSchedulingEnv
from model import PPOActorCritic


def train():
    MAX_NODES = 500
    NUM_EPISODES = 10000
    UPDATE_FREQ = 8  # 稍微调大一点 Batch，让梯度更稳定

    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    PPO_EPOCHS = 4
    CLIP_EPS = 0.2

    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.1  # 前期稍微加大探索
    LEARNING_RATE = 3e-4

    print("正在初始化 ONNX 友好型数据集和物理环境...")
    dataset = DAGDataset()
    env = DAGSchedulingEnv(dataset, max_nodes=MAX_NODES)

    model = PPOActorCritic(node_feature_dim=6, hidden_dim=64, max_nodes=MAX_NODES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 不再使用绝对分数，改用滑动平均记录！
    best_avg_score = -float('inf')
    recent_scores = []
    reward_history = []

    print("\n🚀 PPO 纯净无痛版启动！(已移除错误诱导，引入 Mask 掩码)")
    print("-" * 60)

    batch_states, batch_actions, batch_logprobs = [], [], []
    batch_rewards, batch_values, batch_masks, batch_dones = [], [], [], []

    for episode in range(1, NUM_EPISODES + 1):
        obs_dict, info = env.reset()
        episode_reward = 0
        episode_stalls = 0  # 记录纯净的真实停顿惩罚

        terminated = False

        while not terminated:
            # 👑 修复1：把 mask 也转成 tensor 打包进状态字典
            x_tensor = torch.tensor(obs_dict["x"], dtype=torch.float32)
            adj_tensor = torch.tensor(obs_dict["adj"], dtype=torch.float32)
            mask_tensor = torch.tensor(obs_dict["mask"], dtype=torch.float32)

            state_tensor = {
                "x": x_tensor,
                "adj": adj_tensor,
                "mask": mask_tensor  # 传给神经网络过滤噪音
            }

            model.eval()
            with torch.no_grad():
                logits, value = model(state_tensor)

            # 动作掩码 (防止选到没就绪的指令)
            action_mask = torch.tensor(info["action_mask"], dtype=torch.bool)
            masked_logits = logits.masked_fill(~action_mask, float('-inf'))
            probs = torch.softmax(masked_logits, dim=-1)

            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_obs_dict, env_reward, terminated, truncated, info = env.step(action.item())

            # ==========================================
            # 👑 修复2：拔除错误的 Bonus 毒药！
            # 以前的 cp_bonus 会让每一局的总奖励变成一个常数，误导网络。
            # 现在我们直接用最纯粹的物理反馈：env_reward (包含 Stall 和 Spill 惩罚)
            # ==========================================
            shaped_reward = env_reward

            batch_states.append(state_tensor)
            batch_actions.append(action)
            batch_logprobs.append(log_prob)
            batch_rewards.append(shaped_reward)  # PPO 学习纯净的物理奖励
            batch_values.append(value)
            batch_masks.append(action_mask)  # 注意：这里存的是动作掩码
            batch_dones.append(terminated)

            obs_dict = next_obs_dict
            episode_reward += shaped_reward
            episode_stalls += env_reward  # 此时真实停顿就等于纯净奖励

        reward_history.append(episode_stalls)

        # 计算当前这张图“平均每个节点”承受了多少停顿惩罚
        node_avg_stall = episode_stalls / env.num_nodes
        recent_scores.append(node_avg_stall)
        if len(recent_scores) > 50:
            recent_scores.pop(0)

        if episode % UPDATE_FREQ == 0:
            model.train()
            rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
            values_t = torch.stack(batch_values).squeeze()
            dones_t = torch.tensor(batch_dones, dtype=torch.float32)
            old_logprobs_t = torch.stack(batch_logprobs).squeeze()

            advantages = torch.zeros_like(rewards_t)
            last_gae_lam = 0
            for t in reversed(range(len(rewards_t))):
                if t == len(rewards_t) - 1:
                    next_non_terminal = 1.0 - dones_t[t]
                    next_value = 0.0
                else:
                    next_non_terminal = 1.0 - dones_t[t]
                    next_value = values_t[t + 1]

                delta = rewards_t[t] + GAMMA * next_value * next_non_terminal - values_t[t]
                advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam

            returns = advantages + values_t
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 👑 修复3：在 Batch 训练时，也要把 mask 堆叠起来
            b_x = torch.stack([s["x"] for s in batch_states])
            b_adj = torch.stack([s["adj"] for s in batch_states])
            b_mask = torch.stack([s["mask"] for s in batch_states])

            b_states = {
                "x": b_x,
                "adj": b_adj,
                "mask": b_mask  # 喂给训练状态下的网络
            }

            b_actions = torch.stack(batch_actions)
            b_action_masks = torch.stack(batch_masks)

            for _ in range(PPO_EPOCHS):
                new_logits, new_values = model(b_states)
                new_values = new_values.squeeze()

                masked_new_logits = new_logits.masked_fill(~b_action_masks, float('-inf'))
                new_probs = torch.softmax(masked_new_logits, dim=-1)
                new_m = Categorical(new_probs)

                new_logprobs = new_m.log_prob(b_actions)
                entropy = new_m.entropy().mean()

                ratios = torch.exp(new_logprobs - old_logprobs_t)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(new_values, returns)

                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            batch_states, batch_actions, batch_logprobs = [], [], []
            batch_rewards, batch_values, batch_masks, batch_dones = [], [], [], []

        if episode % 20 == 0:
            avg_stall = np.mean(reward_history[-20:])
            ENTROPY_COEF = max(0.01, ENTROPY_COEF * 0.99)
            print(f"Episode {episode:4d}/{NUM_EPISODES} | "
                  f"近20局平均真实惩罚(越趋近0越好): {avg_stall:7.1f} | "
                  f"好奇心: {ENTROPY_COEF:.4f}")

        # 基于滑动平均分保存模型
        if episode > 100:
            current_rolling_avg = np.mean(recent_scores)
            if current_rolling_avg > best_avg_score:
                best_avg_score = current_rolling_avg
                torch.save(model.state_dict(), "model_gat.pth")

    # 强制把训练到最后、最成熟的模型保存一份，以防万一
    torch.save(model.state_dict(), "latest_model.pth")
    print("-" * 60)
    print("✅ PPO 训练结束！最终模型已保存。")


if __name__ == "__main__":
    train()