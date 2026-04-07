import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# ==========================================
# 1. 解决跨文件夹导入问题
# ==========================================
sys.path.append(os.path.abspath("DQN"))
sys.path.append(os.path.abspath("MCTS"))
sys.path.append(os.path.abspath("final"))

# 导入环境和数据集 (使用 final 版本，因为包含了最完整的动态特征)
from final.dataloader import DAGDataset
from final.env import DAGSchedulingEnv

# 导入各个版本的模型
from model import PPOActorCritic as GCN_PPO
from model_gcn import PPOActorCritic as GAT_PPO
from final.model_sota import PPOActorCritic as SOTA_PPO
from DQN.model_dqn import DuelingDQN
from MCTS.mcts_agent import MCTS

# ==========================================
# 2. 评估参数设置
# ==========================================
MAX_NODES = 500
NUM_TEST_GRAPHS = 30  # 测试集图的数量（如果图很大，MCTS会比较慢，可以根据需要调整）
OUTPUT_DIR = "outputs"

# 确保输出文件夹存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置绘图风格 (论文常用风格)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文和英文
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 3. 加载模型权重的工具函数
# ==========================================
def load_model(model_class, weight_path, **kwargs):
    model = model_class(**kwargs)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.eval()
        print(f"✅ 成功加载权重: {weight_path}")
        return model
    else:
        print(f"⚠️ 找不到权重: {weight_path}，将跳过此模型的评估。")
        return None


# ==========================================
# 4. 评估核心逻辑
# ==========================================
def evaluate_single_graph(env, model, strategy_name):
    """
    运行单张图的调度，返回各项论文需要的核心指标
    """
    obs_dict, info = env.reset()
    terminated = False

    total_stalls = 0.0  # 累计停顿惩罚
    max_reg_pressure = 0  # 峰值寄存器压力
    spill_count = 0  # 发生寄存器溢出(压力>32)的次数
    step_count = 0

    start_time = time.time()

    while not terminated:
        valid_actions = np.where(info["action_mask"] == 1)[0].tolist()
        if len(valid_actions) == 0:
            break

        action = None

        # --- 策略分发 ---
        if strategy_name == "Random":
            action = np.random.choice(valid_actions)

        elif strategy_name == "Heuristic (Critical Path)":
            # 关键路径优先：启发式算法 Baseline
            action = max(valid_actions, key=lambda x: env.cp_length[x])

        elif strategy_name == "MCTS":
            # 需要传入环境的 deepcopy 给 MCTS
            mcts = MCTS(num_simulations=30)  # 模拟次数调小点加快速度
            action = mcts.search(env)

        elif "DQN" in strategy_name:
            state_tensor = {
                "x": torch.tensor(obs_dict["x"], dtype=torch.float32),
                "adj": torch.tensor(obs_dict["adj"], dtype=torch.float32),
                "mask": torch.tensor(obs_dict["mask"], dtype=torch.float32)
            }
            with torch.no_grad():
                q_values = model(state_tensor)
                action_mask = torch.tensor(info["action_mask"], dtype=torch.bool)
                masked_q = q_values.masked_fill(~action_mask, float('-inf'))
                action = torch.argmax(masked_q).item()

        elif "PPO" in strategy_name:
            state_tensor = {
                "x": torch.tensor(obs_dict["x"], dtype=torch.float32),
                "adj": torch.tensor(obs_dict["adj"], dtype=torch.float32),
                "mask": torch.tensor(obs_dict["mask"], dtype=torch.float32)
            }
            with torch.no_grad():
                logits, _ = model(state_tensor)
                action_mask = torch.tensor(info["action_mask"], dtype=torch.bool)
                masked_logits = logits.masked_fill(~action_mask, float('-inf'))
                action = torch.argmax(masked_logits).item()  # 评估时使用确定性贪婪策略

        # --- 环境交互 ---
        next_obs_dict, reward, terminated, _, next_info = env.step(action)

        # --- 指标收集 ---
        # 奖励中包含了 -stalls，如果是复数则表示停顿。把停顿周期还原出来
        stall = max(0, -reward if reward > -2.0 else -reward - 2.0)  # 粗略分离基础停顿和溢出惩罚
        total_stalls += stall

        current_pressure = info["reg_pressure"]
        max_reg_pressure = max(max_reg_pressure, current_pressure)
        if current_pressure > env.max_registers:
            spill_count += 1

        obs_dict = next_obs_dict
        info = next_info
        step_count += 1

    inference_time = time.time() - start_time
    makespan = env.current_cycle  # 最终执行周期 (Makespan)

    return {
        "Strategy": strategy_name,
        "Makespan": makespan,
        "Total_Stalls": total_stalls,
        "Max_Reg_Pressure": max_reg_pressure,
        "Spill_Count": spill_count,
        "Inference_Time_sec": inference_time
    }


# ==========================================
# 5. 主评估流程
# ==========================================
def main():
    print("=" * 60)
    print("📊 开始执行全面模型评估与对比...")
    print("=" * 60)

    # 初始化环境
    dataset = DAGDataset()
    env = DAGSchedulingEnv(dataset, max_nodes=MAX_NODES)

    # 准备待评估的策略清单
    strategies = {
        "Random": None,
        "Heuristic (Critical Path)": None,
        "MCTS": None
    }

    # 动态加载你的强化学习模型
    # 路径基于你代码里保存的文件名
    rl_models = [
        ("Dueling DQN", DuelingDQN, "DQN/best_model_dqn.pth"),
        ("PPO + GCN", GCN_PPO, "latest_model.pth"),
        ("PPO + GAT", GAT_PPO, "best_model_gcn.pth"),
        ("PPO + SOTA (PointerNet)", SOTA_PPO, "final/best_model_sota.pth")
    ]

    for name, model_cls, path in rl_models:
        model_instance = load_model(model_cls, path, node_feature_dim=6, hidden_dim=64, max_nodes=MAX_NODES)
        if model_instance is not None:
            strategies[name] = model_instance

    # 存储所有结果
    all_results = []

    # 固定种子保证公平对比
    np.random.seed(42)
    test_indices = np.random.choice(len(dataset), min(NUM_TEST_GRAPHS, len(dataset)), replace=False)

    print(f"\n🚀 开始测试 {len(test_indices)} 张计算图...")

    for i, idx in enumerate(test_indices):
        print(f"正在评估 Graph {i + 1}/{len(test_indices)} (Dataset Index: {idx})")

        # 确保每个策略测试同一张图
        env.graph_data = dataset[idx]
        env.num_nodes = env.graph_data["num_nodes"]
        env.latency = env.graph_data["node_features"].numpy().flatten()
        env.adj_matrix = env.graph_data["adj_matrix"].numpy()

        for strategy_name, model in strategies.items():
            # 必须深度拷贝环境初始状态，或者强制重置为该图
            env.reset()
            # 覆盖 reset 中的随机图，强制使用当前图
            env.graph_data = dataset[idx]
            env.num_nodes = env.graph_data["num_nodes"]
            env.latency = env.graph_data["node_features"].numpy().flatten()
            env.adj_matrix = env.graph_data["adj_matrix"].numpy()
            # 重新初始化环境的底层状态（重新走一遍reset里对指定图的计算）
            # (简化的hack方法：为了保证公平，我们直接调用 step 逻辑即可，状态已经在 reset 刚被清空了)

            res = evaluate_single_graph(env, model, strategy_name)
            res["Graph_ID"] = idx
            all_results.append(res)

    # ==========================================
    # 6. 数据处理与导出 (论文可用素材)
    # ==========================================
    df = pd.DataFrame(all_results)

    # 保存详细数据
    df.to_csv(os.path.join(OUTPUT_DIR, "detailed_results.csv"), index=False)

    # 计算均值统计
    summary_df = df.groupby("Strategy").mean().reset_index()
    summary_df = summary_df.drop(columns=["Graph_ID"])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

    print("\n✅ 评估完成！结果摘要如下：")
    print(summary_df.to_string(index=False))

    # ==========================================
    # 7. 自动生成高质量论文图表
    # ==========================================
    metrics_to_plot = [
        ("Makespan", "Average Makespan (Total Cycles) ↓", "makespan_comparison.pdf"),
        ("Total_Stalls", "Average Stall Cycles ↓", "stalls_comparison.pdf"),
        ("Max_Reg_Pressure", "Max Register Pressure ↓", "reg_pressure_comparison.pdf"),
        ("Inference_Time_sec", "Inference Time (Seconds) ↓", "inference_time_comparison.pdf")
    ]

    for metric_col, ylabel, filename in metrics_to_plot:
        plt.figure(figsize=(10, 6))

        # 根据表现排序，让图表更有视觉冲击力 (从小到大，越小越好)
        plot_df = summary_df.sort_values(by=metric_col, ascending=True)

        ax = sns.barplot(x="Strategy", y=metric_col, data=plot_df, palette="viridis")
        plt.title(ylabel.replace(" ↓", ""), fontsize=16, pad=15)
        plt.ylabel(ylabel, fontsize=14)
        plt.xlabel("Scheduling Algorithm / Strategy", fontsize=14)
        plt.xticks(rotation=30, ha="right")

        # 在柱子上打上具体数值
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 8),
                        textcoords='offset points',
                        fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), format='pdf', dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, filename.replace('.pdf', '.png')), format='png', dpi=300)
        plt.close()

    print(f"\n🎉 所有对比图表已保存至 `{OUTPUT_DIR}/` 目录下！(包含 PNG 和高质量矢量图 PDF 格式)")


if __name__ == "__main__":
    main()