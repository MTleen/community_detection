import numpy as np
import random
import sys

class RandomWalk:
    def __init__(self, graph, is_directed):
        self.graph = graph
        self.is_directed = is_directed

    # 计算转移概率
    def preprocess_transition_probs(self):
        graph = self.graph
        is_directed = self.is_directed
        # 标准化转移概率
        alias_node = {}
        for node in graph.nodes():
            unnormalized_probs = [graph[node][nbr]['weight'] for nbr in sorted(graph.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_node[node] = alias_setup(normalized_probs)
        self.alias_node = alias_node
    
    def bias_random_walk(self, walk_length, start_node):
        graph = self.graph
        alias_node = self.alias_node
        walk = [start_node]
        while len(walk) < walk_length:
            # 定位当前节点
            cur_node = walk[-1]
            # 获取当前节点所有邻居节点
            cur_nbrs = sorted(graph.neighbors(cur_node))
            if len(cur_nbrs) > 0:
                walk.append(cur_nbrs[alias_draw(alias_node[cur_node][0], alias_node[cur_node][1])])
            else:
                break

    def simulate_walks(self, num_walks, walk_length):
        graph = self.graph
        walks = []
        nodes = list(graph.nodes())
        print('walk iteration...')
        for walk_iter in range(num_walks):
            print('walks per node: %d//%d' % (walk_iter + 1, num_walks))
            random.shuffle(nodes)
            for index, node in enumerate(nodes):
                sys.stdout.write('\r%d//%d' % (index + 1, len(nodes)))
                sys.stdout.flush()
                walks.append(self.bias_random_walk(walk_length=walk_length, start_node=node))

        return walks


def alias_setup(probs):
    # 从离散分布非均衡采样
    length = len(probs)
    q = np.zeros(length)
    J = np.zeros(length, dtype=np.int)

    smaller = []
    larger = []

    for index, prob in enumerate(probs):
        q[index] = length * prob
        if q[index] < 1.0:
            smaller.append(index)
        else: 
            larger.append(index)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    length = len(J)
    rand_index = int(np.floor(np.random.rand() * length))
    if np.random.rand() < q[rand_index]:
        return rand_index
    else: 
        return J[rand_index]

if __name__ == '__main__':
    a, b = alias_setup([0.2, 0.3, 0.25, 0.25])
    print(a, b)
