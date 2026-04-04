import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from dataloader import DAGDataset
from env import DAGSchedulingEnv
from model_dqn import DuelingDQN

# ==========================================
# DQN 经验回放池
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return states, actions, rewards, next_states, dones, masks, next_masks

    def __len__(self):
        return len(self.buffer)

def train_dqn():
    MAX_NODES = 500
    NUM_EPISODES = 10000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LR = 3e-4
    TARGET_UPDATE = 10  # 每 10 局更新一次目标网络

    # Epsilon-Greedy 探索参数
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 2000

    dataset = DAGDataset()
    env = DAGSchedulingEnv(dataset, max_nodes=MAX_NODES)

    # 👑 Double DQN 需要两个网络：一个是主网络（训练），一个是目标网络（稳定 Q 值）
    policy_net = DuelingDQN(node_feature_dim=6, hidden_dim=64, max_nodes=MAX_NODES)
    target_net = DuelingDQN(node_feature_dim=6, hidden_dim=64, max_nodes=MAX_NODES)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(capacity=20000)

    best_avg_score = -float('inf')
    recent_scores = []
    reward_history = []
    global_step = 0

    print("\n🚀 Dueling Double DQN 训练启动！")
    print("-" * 60)

    for episode in range(1, NUM_EPISODES + 1):
        obs_dict, info = env.reset()
        episode_stalls = 0
        terminated = False

        state = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs_dict.items()}
        mask = torch.tensor(info["action_mask"], dtype=torch.bool)

        while not terminated:
            global_step += 1
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * global_step / EPS_DECAY)

            # Epsilon-Greedy 策略
            if random.random() > eps_threshold:
                policy_net.eval()
                with torch.no_grad():
                    q_values = policy_net(state)
                    # 屏蔽无效动作
                    masked_q = q_values.masked_fill(~mask, float('-inf'))
                    action = torch.argmax(masked_q).item()
            else:
                ready_nodes = np.where(mask.numpy() == True)[0]
                action = random.choice(ready_nodes)

            next_obs_dict, reward, terminated, _, next_info = env.step(action)
            episode_stalls += reward

            next_state = {k: torch.tensor(v, dtype=torch.float32) for k, v in next_obs_dict.items()}
            next_mask = torch.tensor(next_info["action_mask"], dtype=torch.bool)

            # 存入经验池
            memory.push(state, action, reward, next_state, terminated, mask, next_mask)
            state = next_state
            mask = next_mask

            # ==========================================
            # 网络训练环节
            # ==========================================
            if len(memory) > BATCH_SIZE:
                policy_net.train()
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_masks, b_next_masks = memory.sample(BATCH_SIZE)

                b_x = torch.stack([s["x"] for s in b_states])
                b_adj = torch.stack([s["adj"] for s in b_states])
                b_m = torch.stack([s["mask"] for s in b_states])
                dict_states = {"x": b_x, "adj": b_adj, "mask": b_m}

                b_nx = torch.stack([s["x"] for s in b_next_states])
                b_nadj = torch.stack([s["adj"] for s in b_next_states])
                b_nm = torch.stack([s["mask"] for s in b_next_states])
                dict_next_states = {"x": b_nx, "adj": b_nadj, "mask": b_nm}

                b_actions = torch.tensor(b_actions, dtype=torch.int64).unsqueeze(1)
                b_rewards = torch.tensor(b_rewards, dtype=torch.float32)
                b_dones = torch.tensor(b_dones, dtype=torch.float32)
                b_next_masks = torch.stack(b_next_masks)

                # 当前 Q 值
                curr_q = policy_net(dict_states).gather(1, b_actions).squeeze(1)

                # 👑 Double DQN 核心逻辑：用 Policy 网络选出最大动作，用 Target 网络评估该动作价值
                with torch.no_grad():
                    next_q_policy = policy_net(dict_next_states)
                    next_q_policy = next_q_policy.masked_fill(~b_next_masks, float('-inf'))
                    best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)

                    next_q_target = target_net(dict_next_states)
                    max_next_q = next_q_target.gather(1, best_next_actions).squeeze(1)

                expected_q = b_rewards + GAMMA * max_next_q * (1 - b_dones)

                loss = nn.MSELoss()(curr_q, expected_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        reward_history.append(episode_stalls)
        recent_scores.append(episode_stalls / env.num_nodes)
        if len(recent_scores) > 50:
            recent_scores.pop(0)

        if episode % 20 == 0:
            avg_stall = np.mean(reward_history[-20:])
            print(f"Episode {episode:4d}/{NUM_EPISODES} | 近20局平均回报: {avg_stall:7.1f} | 探索率 Eps: {eps_threshold:.3f}")

        if episode > 200:
            current_rolling_avg = np.mean(recent_scores)
            if current_rolling_avg > best_avg_score:
                best_avg_score = current_rolling_avg
                torch.save(policy_net.state_dict(), "best_model_dqn.pth")

    torch.save(policy_net.state_dict(), "latest_model_dqn.pth")
    print("✅ DQN 训练结束！模型已保存为 best_model_dqn.pth")

if __name__ == "__main__":
    train_dqn()