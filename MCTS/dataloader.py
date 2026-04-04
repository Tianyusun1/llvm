import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class DAGDataset(Dataset):
    """
    ONNX 友好的 DAG 数据集加载器
    弃用 PyG，改用固定维度的 邻接矩阵(Adj Matrix) 和 特征矩阵(Node Features)
    """
    def __init__(self, data_dir="D:\\DAG_dataset", max_nodes=500):
        super().__init__()
        self.data_dir = data_dir
        self.max_nodes = max_nodes
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        self.data_list = self._process_all_files()

    def _process_all_files(self):
        print(f"开始加载数据集，共找到 {len(self.file_list)} 个 JSON 文件...")
        data_list = []

        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    raw_data = json.load(f)
                except json.JSONDecodeError:
                    continue

            nodes = raw_data.get("Nodes", [])
            edges = raw_data.get("Edges", [])

            if not nodes or len(nodes) > self.max_nodes:
                continue  # 过滤掉空图和超过最大节点数的超大图

            num_nodes = len(nodes)
            id_map = {node["ID"]: idx for idx, node in enumerate(nodes)}

            # 1. 节点特征矩阵 (预分配最大长度 MAX_NODES，全 0 填充)
            # 我们先提取基础的 Latency，其他特征由 env 动态生成
            node_features = np.zeros((self.max_nodes, 1), dtype=np.float32)
            for idx, node in enumerate(nodes):
                node_features[idx, 0] = float(node["Latency"])

            # 2. 邻接矩阵 (Adjacency Matrix)
            # 形状: [MAX_NODES, MAX_NODES]，有边则为 1，无边为 0
            adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
            for edge in edges:
                if edge["From"] in id_map and edge["To"] in id_map:
                    src_idx = id_map[edge["From"]]
                    dst_idx = id_map[edge["To"]]
                    adj_matrix[src_idx, dst_idx] = 1.0

            # 3. 将有效节点数记录下来，方便后续 Mask 掩码处理
            data = {
                "filename": file_name,
                "num_nodes": num_nodes,
                "node_features": torch.tensor(node_features),
                "adj_matrix": torch.tensor(adj_matrix)
            }
            data_list.append(data)

        print(f"✅ 数据加载完毕！成功转换为 {len(data_list)} 个固定维度张量数据 (ONNX-Ready)。")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# ==========================================
# 本地测试模块
# ==========================================
if __name__ == "__main__":
    dataset = DAGDataset()
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n--- 抽取第 1 张图看看它长什么样 ---")
        print(f"图文件名: {sample['filename']}")
        print(f"实际节点数: {sample['num_nodes']}")
        print(f"节点特征矩阵形状 (Pad 到 500): {sample['node_features'].shape}")
        print(f"邻接矩阵形状 (Pad 到 500): {sample['adj_matrix'].shape}")