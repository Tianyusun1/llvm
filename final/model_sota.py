import torch
import torch.nn as nn
import torch.nn.functional as F

class FastGATLayer(nn.Module):
    """底层特征提取依然使用你优秀的 GAT 引擎"""
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

class PPOActorCritic(nn.Module):
    """
    搭载 SOTA 架构：GAT 特征引擎 + 全局感知指针网络 (Pointer Network) Actor Head
    """
    def __init__(self, node_feature_dim=6, hidden_dim=64, max_nodes=500):
        super(PPOActorCritic, self).__init__()
        self.max_nodes = max_nodes

        # 1. 骨干网络 (Backbone)
        self.gat1 = FastGATLayer(node_feature_dim, hidden_dim)
        self.gat2 = FastGATLayer(hidden_dim, hidden_dim)

        # ==================================
        # 👑 [绝杀机制]：Pointer Network Actor Head
        # ==================================
        # 生成全局 Query
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        # 生成节点 Key
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # 2. 评论家网络 (Critic) 依然负责评估图的宏观价值
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

        # 提取高维特征
        h = self.gat1(x, adj, mask)
        h = self.gat2(h, adj, mask)

        # 计算纯净的全局特征 (Global Context)
        sum_h = h.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        global_h = sum_h / valid_counts  # [B, hidden_dim]

        # ==================================
        # 👑 Actor 打分逻辑 (Attention 机制)
        # ==================================
        # 全局视角作为 Query [B, hidden_dim] -> 扩展为 [B, 1, hidden_dim]
        query = self.query_proj(global_h).unsqueeze(1)
        # 每个节点的特征作为 Key [B, N, hidden_dim]
        keys = self.key_proj(h)

        # 计算点积注意力得分 Logits: Query @ Key.T
        # 相当于用当前“全局的寄存器压力和进度”去审视“每一个节点的局部特征”
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) # 输出 [B, N]

        # Critic 估值
        value = self.critic(global_h).squeeze(-1)

        if is_unbatched:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value