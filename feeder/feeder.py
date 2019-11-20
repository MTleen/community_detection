import numpy as np
import random
import torch
import torch.utils.data as data


class Feeder(data.Dataset):
    """
    构建某个节点的子图
    当前子图的深度暂定为 2
    """

    def __init__(self, feat_path, knn_graph_path, label_path, seed=1,
                 k_at_hop=[200, 5], active_connection=5, train=True):
        np.random.seed(seed)
        random.seed(seed)
        # 特征向量路径
        self.features = np.load(feat_path)
        # knn 路径， 直接取一阶邻居
        self.knn_graph = np.load(knn_graph_path)
        # 判断 1-hop 的数量是否大于网络所有节点数
        num_1_hop = k_at_hop[0]
        if self.knn_graph.shape[1] <= num_1_hop:
            k_at_hop[0] = self.knn_graph.shape[1] - 1
        self.knn_graph = self.knn_graph[:, :k_at_hop[0] + 1]
        # 标签路径
        self.labels = np.load(label_path)
        self.num_samples = len(self.features)
        # 邻接子图的深度
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.train = train
        assert np.mean(k_at_hop) >= active_connection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        返回节点特征矩阵、邻接矩阵A、中心节点index和其一阶邻居节点
        hops[0]：1阶邻居节点的个数
        hops[1]: 2阶邻居节点的个数
        """
        hops = list()
        center_node = index
        # hops 中添加中心节点的 knns
        hops.append(set(self.knn_graph[center_node][1:]))

        # hops 中添加高阶邻居的 knns
        for d in range(1, self.depth):
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(self.knn_graph[h][1:self.k_at_hop[d] + 1]))

        # 中心节点所有 knns 中去除重复节点
        hops_set = set([h for hop in hops for h in hop])
        # hops_set 最后一个元素为中心节点 index
        hops_set.update([center_node, ])
        unique_nodes_list = list(hops_set)
        unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)}

        center_idx = torch.Tensor([unique_nodes_map[center_node], ]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(self.features[center_node]).type(torch.float)
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        # 邻接节点和中心节点相对位置关系编码
        feat = feat - center_feat

        # max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        max_num_nodes = 1
        for num_hop in self.k_at_hop:
            max_num_nodes *= num_hop
        max_num_nodes += 1
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)

        # 在邻接子图中添加边
        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection + 1]
            for n in neighbors:
                if n in unique_nodes_list:
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1

        D = A.sum(1, keepdim=True)
        A = A.div(D)
        # 使每一个邻接子图的邻接矩阵维度相同
        A_ = torch.zeros(max_num_nodes, max_num_nodes)
        A_[:num_nodes, :num_nodes] = A

        labels = self.labels[np.asarray(unique_nodes_list)]
        labels = torch.from_numpy(labels).type(torch.long)
        # edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())
        one_hop_labels = labels[one_hop_idcs]
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).long()

        # training
        if self.train:
            return (feat, A_, center_idx, one_hop_idcs), edge_labels

        # Testing
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
            [unique_nodes_list, torch.zeros(max_num_nodes - num_nodes)], dim=0)
        return (feat, A_, center_idx, one_hop_idcs, unique_nodes_list), edge_labels
