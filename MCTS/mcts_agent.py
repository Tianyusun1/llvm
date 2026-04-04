import math
import copy
import random
import numpy as np


class MCTSNode:
    def __init__(self, action=None, parent=None):
        self.action = action  # 到达此节点所采取的动作
        self.parent = parent
        self.children = {}  # action_id -> MCTSNode
        self.visits = 0  # 访问次数
        self.total_reward = 0.0  # 累计奖励
        self.untried_actions = []  # 尚未探索的合法动作


class MCTS:
    """
    轻量级蒙特卡洛树搜索 (Monte Carlo Tree Search)
    专门针对 DAGSchedulingEnv 设计
    """

    def __init__(self, num_simulations=50, exploration_weight=1.414):
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight

    def search(self, env):
        """
        输入当前真实的 env，返回当前最优的 action
        """
        root = MCTSNode()
        # 获取当前合法的动作
        valid_actions = np.where(env._get_info()["action_mask"] == 1)[0].tolist()
        root.untried_actions = valid_actions

        for _ in range(self.num_simulations):
            # 为了不破坏真实环境，每次模拟都需要深度克隆 (Deepcopy)
            sim_env = copy.deepcopy(env)
            node = root

            # 1. 选择阶段 (Selection)
            while node.untried_actions == [] and node.children != {}:
                node = self._select_child(node)
                sim_env.step(node.action)

            # 2. 扩展阶段 (Expansion)
            if node.untried_actions != []:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                _, reward, done, _, info = sim_env.step(action)
                child_node = MCTSNode(action=action, parent=node)

                # 初始化新节点的合法动作
                if not done:
                    child_node.untried_actions = np.where(info["action_mask"] == 1)[0].tolist()

                node.children[action] = child_node
                node = child_node

            # 3. 模拟阶段 (Rollout/Simulation)
            rollout_reward = 0.0
            done = False
            # 简单启发式 Rollout：随机跑到底，或者用关键路径优先跑到底
            while not done:
                info = sim_env._get_info()
                valid_actions = np.where(info["action_mask"] == 1)[0].tolist()
                if len(valid_actions) == 0:
                    break
                # 在模拟中，我们用 Heuristic (关键路径长优先) 来加速 Rollout 的质量
                action = max(valid_actions, key=lambda x: sim_env.cp_length[x])
                _, r, done, _, _ = sim_env.step(action)
                rollout_reward += r

            # 4. 回溯阶段 (Backpropagation)
            while node is not None:
                node.visits += 1
                node.total_reward += rollout_reward
                node = node.parent

        # 最终选择访问次数最多的节点作为最佳动作
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

    def _select_child(self, node):
        # UCB1 公式 (Upper Confidence Bound)
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            exploit = child.total_reward / child.visits
            explore = self.exploration_weight * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child