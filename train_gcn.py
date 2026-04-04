import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

# 👑 引入分离出来的 GCN 模型
from dataloader import DAGDataset
from env import DAGSchedulingEnv
from model_gcn import PPOActorCritic


def train_gcn():
    MAX_NODES = 500
    NUM_EPISODES = 10000
    UPDATE_FREQ = 8

    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    PPO_EPOCHS = 4
    CLIP_EPS = 0.2

    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.1
    LEARNING_RATE = 3e-4

    print("正在初始化 ONNX 友好型数据集和物理环境...")
    dataset = DAGDataset()
    env = DAGSchedulingEnv(dataset, max_nodes=MAX_NODES)

    # 初始化 GCN 网络
    model = PPOActorCritic(node_feature_dim=6, hidden_dim=64, max_nodes=MAX_NODES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_avg_score = -float('inf')
    recent_scores = []
    reward_history = []

    print("\n🚀 PPO-GCN 基线模型训练启动！")
    print("-" * 60)

    batch_states, batch_actions, batch_logprobs = [], [], []
    batch_rewards, batch_values, batch_masks, batch_dones = [], [], [], []

    for episode in range(1, NUM_EPISODES + 1):
        obs_dict, info = env.reset()
        episode_reward = 0
        episode_stalls = 0

        terminated = False

        while not terminated:
            x_tensor = torch.tensor(obs_dict["x"], dtype=torch.float32)
            adj_tensor = torch.tensor(obs_dict["adj"], dtype=torch.float32)
            mask_tensor = torch.tensor(obs_dict["mask"], dtype=torch.float32)

            state_tensor = {
                "x": x_tensor,
                "adj": adj_tensor,
                "mask": mask_tensor
            }

            model.eval()
            with torch.no_grad():
                logits, value = model(state_tensor)

            action_mask = torch.tensor(info["action_mask"], dtype=torch.bool)
            masked_logits = logits.masked_fill(~action_mask, float('-inf'))
            probs = torch.softmax(masked_logits, dim=-1)

            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_obs_dict, env_reward, terminated, truncated, info = env.step(action.item())

            shaped_reward = env_reward

            batch_states.append(state_tensor)
            batch_actions.append(action)
            batch_logprobs.append(log_prob)
            batch_rewards.append(shaped_reward)
            batch_values.append(value)
            batch_masks.append(action_mask)
            batch_dones.append(terminated)

            obs_dict = next_obs_dict
            episode_reward += shaped_reward
            episode_stalls += env_reward

        reward_history.append(episode_stalls)
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

            b_x = torch.stack([s["x"] for s in batch_states])
            b_adj = torch.stack([s["adj"] for s in batch_states])
            b_mask = torch.stack([s["mask"] for s in batch_states])

            b_states = {
                "x": b_x,
                "adj": b_adj,
                "mask": b_mask
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
                  f"近20局平均真实惩罚: {avg_stall:7.1f} | "
                  f"好奇心: {ENTROPY_COEF:.4f}")

        # 👑 专属命名，防止覆盖
        if episode > 100:
            current_rolling_avg = np.mean(recent_scores)
            if current_rolling_avg > best_avg_score:
                best_avg_score = current_rolling_avg
                torch.save(model.state_dict(), "best_model_gcn.pth")

    torch.save(model.state_dict(), "latest_model_gcn.pth")
    print("-" * 60)
    print("✅ PPO-GCN 训练结束！模型已保存为 best_model_gcn.pth")


if __name__ == "__main__":
    train_gcn()