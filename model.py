import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGCNLayer(nn.Module):
    """
    ONNX 友好的原生图卷积层 (完全脱离 torch_geometric)
    利用矩阵乘法 (Adjacency Matrix @ Features) 实现节点间的信息传递
    """

    def __init__(self, in_dim, out_dim):
        super(FastGCNLayer, self).__init__()
        # 处理节点自身的特征
        self.linear_self = nn.Linear(in_dim, out_dim)
        # 处理邻居传递过来的特征
        self.linear_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x 形状: [Batch, N, Features] 或 [N, Features]
        # adj 形状: [Batch, N, N] 或 [N, N]

        self_feat = self.linear_self(x)

        # 矩阵乘法：邻接矩阵 乘以 特征矩阵，瞬间完成所有邻居信息的聚合！
        # 这是 ONNX 100% 完美支持的标准算子
        neigh_aggr = torch.matmul(adj, x)
        neigh_feat = self.linear_neigh(neigh_aggr)

        # 将自身信息与邻居信息相加，并通过激活函数
        return F.elu(self_feat + neigh_feat)


class PPOActorCritic(nn.Module):
    """
    PPO 算法专用的演员-评论家 (Actor-Critic) 神经网络结构
    修复版：引入 Mask 机制，彻底清除 Padding 带来的无效节点噪音
    """

    def __init__(self, node_feature_dim=6, hidden_dim=64, max_nodes=500):
        super(PPOActorCritic, self).__init__()
        self.max_nodes = max_nodes

        # 👑 [特征提取器]：连续两层原生图卷积
        self.gcn1 = FastGCNLayer(node_feature_dim, hidden_dim)
        self.gcn2 = FastGCNLayer(hidden_dim, hidden_dim)

        # 👑 [Actor 网络 (演员)]：负责决策选哪个节点
        # 输入每个节点的高维特征，输出该节点的调度得分 (Logits)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        # 👑 [Critic 网络 (评论家)]：负责评估当前整张图的价值 (Value)
        # 辅助 PPO 算法计算优势函数 (Advantage)，稳定训练过程
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs_dict):
        """
        前向传播
        支持单步推理 (无 Batch 维度) 和 批量训练 (有 Batch 维度)
        """
        x = obs_dict["x"]          # 节点特征 [N, F] 或 [B, N, F]
        adj = obs_dict["adj"]      # 邻接矩阵 [N, N] 或 [B, N, N]
        mask = obs_dict["mask"]    # 👑 有效节点掩码 [N, 1] 或 [B, N, 1]

        # 自动处理 Batch 维度 (ONNX 导出时非常关键)
        is_unbatched = False
        if x.dim() == 2:
            is_unbatched = True
            x = x.unsqueeze(0)        # [N, F] -> [1, N, F]
            adj = adj.unsqueeze(0)    # [N, N] -> [1, N, N]
            mask = mask.unsqueeze(0)  # [N, 1] -> [1, N, 1]

        # 提取图结构特征
        # 👑 绝杀：每次经过 GCN（带 Linear 层 Bias）后，强行将无效节点特征归零！
        h = self.gcn1(x, adj) * mask
        h = self.gcn2(h, adj) * mask

        # ==================================
        # 1. 计算 Actor 输出 (动作策略)
        # ==================================
        # actor(h) 输出 [Batch, N, 1]，squeeze(-1) 变成 [Batch, N]
        logits = self.actor(h).squeeze(-1)

        # ==================================
        # 2. 计算 Critic 输出 (状态价值)
        # ==================================
        # 👑 修复 Global Pooling：只对有效节点求均值 (Masked Mean)
        # 以前直接 mean() 会把几百个 Padding 的 0 或者随机噪音一起平均进去，导致估值崩塌。
        sum_h = h.sum(dim=1)  # 对所有节点特征求和: [Batch, hidden_dim]
        valid_counts = mask.sum(dim=1).clamp(min=1.0)  # 计算有效节点数，防除0报错: [Batch, 1]
        global_h = sum_h / valid_counts  # 算出真正纯净的全局图特征: [Batch, hidden_dim]

        # critic(global_h) 输出 [Batch, 1]，squeeze(-1) 变成 [Batch]
        value = self.critic(global_h).squeeze(-1)

        # 如果输入没有 Batch 维度，输出也把 Batch 维度去掉
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

    print("加载数据集与物理环境...")
    dataset = DAGDataset()
    env = DAGSchedulingEnv(dataset, max_nodes=500)

    obs, info = env.reset()

    # 初始化我们全新的 PPO 双头网络 (确认特征维度为 6)
    model = PPOActorCritic(node_feature_dim=6, hidden_dim=64, max_nodes=500)
    model.eval()

    print("\n--- 让 PPO 网络进行一次推理 ---")

    # 注意：这里我们传入的是 obs 字典！(包含 x, adj 和 mask)
    with torch.no_grad():
        # 我们把 numpy 转成 torch.Tensor
        obs_tensor = {
            "x": torch.tensor(obs["x"]),
            "adj": torch.tensor(obs["adj"]),
            "mask": torch.tensor(obs["mask"])  # 👑 记得把 mask 也传进去
        }
        logits, value = model(obs_tensor)

    print(f"✅ PPO 双头网络带 Mask 运行成功！")
    print(f"Actor 输出形状 (所有节点的得分): {logits.shape} (应为 [{env.max_nodes}])")
    print(f"Critic 输出形状 (当前纯净图的总估值): {value.shape} (应为标量或空 [] 表示单值)")
    print(f"当前图状态的纯净网络估值 Value: {value.item():.4f}")

    # 结合掩码挑选最佳动作
    action_mask = torch.tensor(info["action_mask"], dtype=torch.bool)
    masked_logits = logits.masked_fill(~action_mask, float('-inf'))
    best_action = torch.argmax(masked_logits).item()
    print(f"\nActor 决定走的下一步是: 节点 {best_action}")