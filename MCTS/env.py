import gymnasium as gym
import numpy as np

class DAGSchedulingEnv(gym.Env):
    """
    ONNX 友好的指令调度环境 (SOTA 升级版)
    👑 核心创新：引入 ILP 奖励与动态寄存器压力警戒线，打破“极度串行”的次优陷阱
    """

    def __init__(self, dataset, max_nodes=500, max_registers=32):
        super().__init__()
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.max_registers = max_registers

        self.action_space = gym.spaces.Discrete(self.max_nodes)
        self.observation_space = gym.spaces.Dict({
            "x": gym.spaces.Box(low=0, high=1000, shape=(self.max_nodes, 6), dtype=np.float32),
            "adj": gym.spaces.Box(low=0, high=1, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
            "mask": gym.spaces.Box(low=0, high=1, shape=(self.max_nodes, 1), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        import random
        self.graph_data = random.choice(self.dataset)
        self.num_nodes = self.graph_data["num_nodes"]

        self.latency = self.graph_data["node_features"].numpy().flatten()
        self.adj_matrix = self.graph_data["adj_matrix"].numpy()

        self.scheduled = np.zeros(self.max_nodes, dtype=bool)
        self.completion_times = np.zeros(self.max_nodes, dtype=float)
        self.in_degree = np.zeros(self.max_nodes, dtype=int)
        self.out_degree = np.zeros(self.max_nodes, dtype=float)
        self.cp_length = np.zeros(self.max_nodes, dtype=float)

        self.live_registers = set()
        self.current_reg_pressure = 0.0
        self.current_cycle = 0.0

        for i in range(self.num_nodes):
            self.out_degree[i] = np.sum(self.adj_matrix[i, :])
            self.in_degree[i] = np.sum(self.adj_matrix[:, i])

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

        # Liveness Analysis (更新寄存器)
        if len(children) > 0:
            self.live_registers.add(action)

        for p in parents:
            if p in self.live_registers:
                p_children = np.where(self.adj_matrix[p, :] == 1)[0]
                if np.all(self.scheduled[p_children]):
                    self.live_registers.remove(p)

        self.current_reg_pressure = len(self.live_registers)
        terminated = bool(np.all(self.scheduled[:self.num_nodes]))

        # =====================================================
        # 👑 [绝杀机制]：动态自适应奖励 (Adaptive Reward Shaping)
        # =====================================================
        reward = -float(stall_cycles)  # 基础停顿惩罚

        # 1. 鼓励并发：如果当前没有停顿，且寄存器极度充裕，给予小额 ILP 奖励
        if stall_cycles == 0 and self.current_reg_pressure <= 24:
            reward += 0.2

        # 2. 软性警告 (警戒区)：压力靠近红线时，给予轻微惩罚，引导模型收敛并行度
        if 24 < self.current_reg_pressure <= self.max_registers:
            reward -= 0.5

        # 3. 致命惩罚 (爆寄存器)：超过硬件极限，重拳出击！
        if self.current_reg_pressure > self.max_registers:
            reward -= 5.0

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        obs_x = np.zeros((self.max_nodes, 6), dtype=np.float32)
        obs_x[:, 0] = self.latency / 10.0
        obs_x[self.scheduled, 1] = 1.0
        obs_x[self.ready_nodes, 2] = 1.0
        obs_x[:, 3] = self.out_degree / 5.0
        obs_x[:, 4] = self.cp_length / 20.0
        obs_x[:, 5] = self.current_reg_pressure / float(self.max_registers)

        valid_mask = np.zeros((self.max_nodes, 1), dtype=np.float32)
        valid_mask[:self.num_nodes, 0] = 1.0

        return {"x": obs_x, "adj": self.adj_matrix, "mask": valid_mask}

    def _get_info(self):
        mask = np.zeros(self.max_nodes, dtype=np.int8)
        for node in self.ready_nodes:
            mask[node] = 1
        return {
            "action_mask": mask,
            "current_cycle": self.current_cycle,
            "reg_pressure": self.current_reg_pressure
        }