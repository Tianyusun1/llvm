import torch
import torch.nn as nn
import torch.nn.functional as F

class FastGATLayer(nn.Module):
    # 保持不变的优秀底层引擎
    def __init__(self, in_dim, out_dim):
        super(FastGATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj, mask):
        h = self.W(x)
        src_score = self.a_src(h)
        dst_score = self.a_dst(h)
        e = src_score + dst_score.transpose(-1, -2)
        e = self.leaky_relu(e)
        e = e.masked_fill(adj == 0, float('-inf'))
        e = e.masked_fill(mask.transpose(-1, -2) == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = torch.matmul(alpha, h)
        out = out + h
        return F.elu(out) * mask

class DuelingDQN(nn.Module):
    """
    Dueling DQN 架构：将 Q 值拆分为 状态价值 (V) 和 动作优势 (A)
    """
    def __init__(self, node_feature_dim=6, hidden_dim=64, max_nodes=500):
        super(DuelingDQN, self).__init__()
        self.max_nodes = max_nodes

        self.gat1 = FastGATLayer(node_feature_dim, hidden_dim)
        self.gat2 = FastGATLayer(hidden_dim, hidden_dim)

        # 👑 Dueling 拆分：优势流 (Advantage Stream) -> 评估每个节点的动作价值
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        # 👑 Dueling 拆分：价值流 (Value Stream) -> 评估当前图的整体状态价值
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs_dict):
        x = obs_dict["x"]
        adj = obs_dict["adj"]
        mask = obs_dict["mask"]

        is_unbatched = False
        if x.dim() == 2:
            is_unbatched = True
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            mask = mask.unsqueeze(0)

        h = self.gat1(x, adj, mask)
        h = self.gat2(h, adj, mask)

        # 1. 计算 Advantage (每个节点的相对优势) [Batch, N]
        advantage = self.advantage_head(h).squeeze(-1)
        # 用 mask 把无效节点的 advantage 归零
        advantage = advantage * mask.squeeze(-1)

        # 2. 计算 Value (全局状态价值) [Batch, 1]
        sum_h = h.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        global_h = sum_h / valid_counts
        value = self.value_head(global_h)

        # 3. 组合得到 Q 值: Q(s,a) = V(s) + (A(s,a) - mean(A))
        # 减去 mean 是 Dueling DQN 的标准操作，用于提高网络稳定性
        adv_mean = advantage.sum(dim=1, keepdim=True) / valid_counts
        q_values = value + (advantage - adv_mean)

        if is_unbatched:
            q_values = q_values.squeeze(0)

        return q_values