import gymnasium as gym
import numpy as np


class DAGSchedulingEnv(gym.Env):
    """
    ONNX 友好的指令调度环境
    包含物理特征模拟：真实追踪流水线 Stall 与寄存器压力 (Register Pressure)
    修复版：新增有效节点 Mask 机制，防止 Padding 噪音污染模型
    """

    def __init__(self, dataset, max_nodes=500, max_registers=32):
        super().__init__()
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.max_registers = max_registers  # 模拟 ARM64/x86 常用的可用寄存器数量

        self.action_space = gym.spaces.Discrete(self.max_nodes)

        # 观察空间现在是一个包含特征、邻接矩阵和掩码的字典
        # 6维特征：[延迟, 已调度, 就绪, 出度, 关键路径长度, 寄存器压力]
        self.observation_space = gym.spaces.Dict({
            "x": gym.spaces.Box(low=0, high=1000, shape=(self.max_nodes, 6), dtype=np.float32),
            "adj": gym.spaces.Box(low=0, high=1, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
            "mask": gym.spaces.Box(low=0, high=1, shape=(self.max_nodes, 1), dtype=np.float32) # 👑 新增掩码
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 从新的 dataset 里抽取数据 (字典格式)
        import random
        self.graph_data = random.choice(self.dataset)
        self.num_nodes = self.graph_data["num_nodes"]

        # 获取基础特征
        self.latency = self.graph_data["node_features"].numpy().flatten()
        self.adj_matrix = self.graph_data["adj_matrix"].numpy()

        # 初始化底层状态
        self.scheduled = np.zeros(self.max_nodes, dtype=bool)
        self.completion_times = np.zeros(self.max_nodes, dtype=float)
        self.in_degree = np.zeros(self.max_nodes, dtype=int)
        self.out_degree = np.zeros(self.max_nodes, dtype=float)
        self.cp_length = np.zeros(self.max_nodes, dtype=float)

        # 👑 [物理模拟]：寄存器压力追踪
        self.live_registers = set()  # 当前存活的变量（占用寄存器）
        self.current_reg_pressure = 0.0

        self.current_cycle = 0.0

        # 利用邻接矩阵计算出入度 (只计算有效节点部分)
        for i in range(self.num_nodes):
            self.out_degree[i] = np.sum(self.adj_matrix[i, :])
            self.in_degree[i] = np.sum(self.adj_matrix[:, i])

        # 拓扑排序与关键路径计算 (基于邻接矩阵)
        in_degree_temp = self.in_degree.copy()
        topo_order = []
        queue = [i for i in range(self.num_nodes) if in_degree_temp[i] == 0]
        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            children = np.where(self.adj_matrix[curr, :] == 1)[0]
            for child in children:
                in_degree_temp[child] -= 1
                if in_degree_temp[child] == 0:
                    queue.append(child)

        for node in reversed(topo_order):
            lat = self.latency[node]
            children = np.where(self.adj_matrix[node, :] == 1)[0]
            if len(children) > 0:
                self.cp_length[node] = lat + np.max(self.cp_length[children])
            else:
                self.cp_length[node] = lat

        self.ready_nodes = np.where(self.in_degree[:self.num_nodes] == 0)[0].tolist()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action >= self.num_nodes or action not in self.ready_nodes:
            return self._get_obs(), -10.0, False, False, self._get_info()

        self.ready_nodes.remove(action)
        self.scheduled[action] = True

        parents = np.where(self.adj_matrix[:, action] == 1)[0]
        max_parent_completion = np.max(self.completion_times[parents]) if len(parents) > 0 else 0.0

        start_time = max(self.current_cycle, max_parent_completion)
        stall_cycles = max(0.0, max_parent_completion - self.current_cycle)

        self.completion_times[action] = start_time + self.latency[action]
        self.current_cycle = start_time + 1.0

        children = np.where(self.adj_matrix[action, :] == 1)[0]
        for child in children:
            self.in_degree[child] -= 1
            if self.in_degree[child] == 0:
                self.ready_nodes.append(child)

        # =====================================================
        # 👑 [核心创新] 寄存器生命周期模拟 (Liveness Analysis)
        # =====================================================
        # 1. 如果当前指令有后续指令依赖它，说明它产生了一个中间值，需要占用 1 个寄存器
        if len(children) > 0:
            self.live_registers.add(action)

        # 2. 检查父节点，如果父节点的所有子节点都已经调度完，说明父节点的值没用了，释放寄存器！
        for p in parents:
            if p in self.live_registers:
                p_children = np.where(self.adj_matrix[p, :] == 1)[0]
                # 如果所有孩子都调度了
                if np.all(self.scheduled[p_children]):
                    self.live_registers.remove(p)

        self.current_reg_pressure = len(self.live_registers)

        # 判断是否结束 (有效节点是否全调度完)
        terminated = bool(np.all(self.scheduled[:self.num_nodes]))

        # 👑 [物理惩罚奖励]
        reward = -float(stall_cycles)

        # 如果寄存器压力超过硬件限制，发生 Spill，给予严重惩罚！
        if self.current_reg_pressure > self.max_registers:
            reward -= 2.0  # 发生寄存器溢出，扣分！

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        # 构造 6 维固定形状特征矩阵 [MAX_NODES, 6]
        obs_x = np.zeros((self.max_nodes, 6), dtype=np.float32)

        # 严格归一化特征
        obs_x[:, 0] = self.latency / 10.0
        obs_x[self.scheduled, 1] = 1.0
        obs_x[self.ready_nodes, 2] = 1.0
        obs_x[:, 3] = self.out_degree / 5.0
        obs_x[:, 4] = self.cp_length / 20.0

        # 第 6 维特征：将全局当前的寄存器压力广播给所有节点
        obs_x[:, 5] = self.current_reg_pressure / float(self.max_registers)

        # 👑 新增：生成只包含真实节点的 valid_mask
        valid_mask = np.zeros((self.max_nodes, 1), dtype=np.float32)
        valid_mask[:self.num_nodes, 0] = 1.0

        # 返回字典格式，包含特征矩阵、邻接矩阵和掩码
        return {
            "x": obs_x,
            "adj": self.adj_matrix,
            "mask": valid_mask  # 添加到这里供网络和损失函数使用
        }

    def _get_info(self):
        mask = np.zeros(self.max_nodes, dtype=np.int8)
        for node in self.ready_nodes:
            mask[node] = 1
        return {
            "action_mask": mask,
            "current_cycle": self.current_cycle,
            "reg_pressure": self.current_reg_pressure  # 顺便在info里暴露出压力值方便统计
        }


# ==========================================
# 本地测试模块
# ==========================================
if __name__ == "__main__":
    from dataloader import DAGDataset

    dataset = DAGDataset()
    env = DAGSchedulingEnv(dataset)
    obs, info = env.reset()

    print("\n--- 物理环境初始化成功！ ---")
    print(f"观察特征字典包含: {obs.keys()}")
    print(f"特征矩阵 x 形状: {obs['x'].shape} (6维特征！)")
    print(f"邻接矩阵 adj 形状: {obs['adj'].shape}")
    print(f"有效节点掩码 mask 形状: {obs['mask'].shape} (用来干掉Padding噪音)")

    print("\n[随便走几步，观察寄存器压力变化...]")
    for _ in range(5):
        valid_actions = np.where(info["action_mask"] == 1)[0]
        if len(valid_actions) == 0: break
        action = np.random.choice(valid_actions)
        obs, reward, done, _, info = env.step(action)
        print(f"选择指令 {action:3d} | 单步奖励: {reward:5.1f} | 此时寄存器压力: {info['reg_pressure']} 个")