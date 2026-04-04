import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGATLayer(nn.Module):
    """
    ONNX 友好的原生图注意力层 (Graph Attention Network)
    替代之前的 GCN，让网络学会“重点关注”关键路径或高压力的依赖边
    """

    def __init__(self, in_dim, out_dim):
        super(FastGATLayer, self).__init__()
        # 特征变换矩阵
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # 注意力评分向量 (分别评估源节点和目标节点)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj, mask):
        # x 形状: [B, N, in_dim]
        # adj 形状: [B, N, N]
        # mask 形状: [B, N, 1]

        h = self.W(x)  # [B, N, out_dim]

        # 1. 计算每个节点作为“发送方”和“接收方”的原始得分
        src_score = self.a_src(h)  # [B, N, 1]
        dst_score = self.a_dst(h)  # [B, N, 1]

        # 2. 组合得分，生成注意力矩阵 [B, N, N]
        # 巧妙利用广播机制：a_i + a_j
        e = src_score + dst_score.transpose(-1, -2)
        e = self.leaky_relu(e)

        # 3. 掩码处理 (核心保命技巧)
        # 把没有边的位置强行设为负无穷
        e = e.masked_fill(adj == 0, float('-inf'))
        # 把 Padding 进来的无效节点强行设为负无穷 (列掩码)
        e = e.masked_fill(mask.transpose(-1, -2) == 0, float('-inf'))

        # 4. Softmax 归一化，得到真正的注意力权重 (Attention Weights)
        alpha = F.softmax(e, dim=-1)
        # 如果一个节点没有任何出边，Softmax 会产生 NaN，这里安全地补0
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # 5. 根据注意力权重聚合邻居特征
        out = torch.matmul(alpha, h)  # [B, N, N] @ [B, N, out_dim] -> [B, N, out_dim]

        # 6. 残差连接 (Residual Connection)，让训练更深更稳定，并过激活函数
        out = out + h
        return F.elu(out) * mask  # 再次确保无效节点全为 0


class PPOActorCritic(nn.Module):
    """
    搭载了全新 GAT (图注意力) 引擎的 PPO 双头网络
    """

    def __init__(self, node_feature_dim=6, hidden_dim=64, max_nodes=500):
        super(PPOActorCritic, self).__init__()
        self.max_nodes = max_nodes

        # 👑 [动力升级]：连续两层注意力网络
        self.gat1 = FastGATLayer(node_feature_dim, hidden_dim)
        self.gat2 = FastGATLayer(hidden_dim, hidden_dim)

        # Actor 网络保持不变
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        # Critic 网络保持不变
        self.critic = nn.Sequential(
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

        # 使用注意力网络提取特征，自带 mask 过滤
        h = self.gat1(x, adj, mask)
        h = self.gat2(h, adj, mask)

        logits = self.actor(h).squeeze(-1)

        sum_h = h.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        global_h = sum_h / valid_counts

        value = self.critic(global_h).squeeze(-1)

        if is_unbatched:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value


# ==========================================
# 本地测试模块
# ==========================================
if __name__ == "__main__":
    from dataloader import DAGDataset
    from env import DAGSchedulingEnv
    import numpy as np

    dataset = DAGDataset()
    env = DAGSchedulingEnv(dataset, max_nodes=500)

    obs, info = env.reset()

    model = PPOActorCritic(node_feature_dim=6, hidden_dim=64, max_nodes=500)
    model.eval()

    with torch.no_grad():
        obs_tensor = {
            "x": torch.tensor(obs["x"]),
            "adj": torch.tensor(obs["adj"]),
            "mask": torch.tensor(obs["mask"])
        }
        logits, value = model(obs_tensor)

    print(f"✅ GAT 引擎点火成功！")
    print(f"Actor 输出形状: {logits.shape}")
    print(f"Critic 估值 Value: {value.item():.4f}")