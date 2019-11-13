import numpy as np
# import networkx as nx
import random
import sys


class Walk:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def bias_random_walk(self, walk_length, start_node):
        '''
		Simulate a random walk starting from start node.
		'''
        G = self.G
        alias_nodes = self.alias_nodes
        # alias_edges = self.alias_edges
        # 随机游走序列
        walk = [start_node]

        while len(walk) < walk_length:
            # 更改游走起始点
            cur = walk[-1]
            # 当前节点邻居
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                # if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
            # else:
            # 	prev = walk[-2]
            # 	next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
            # 		alias_edges[(prev, cur)][1])]
            # 	walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
		Repeatedly simulate random walks from each node.
		'''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print('\nWalks per node:{}/{}'.format(walk_iter + 1, num_walks))
            random.shuffle(nodes)
            # for node in nodes:
            for j, node in enumerate(nodes):
                sys.stdout.write('\r{}/{}'.format(j + 1, len(nodes)))
                sys.stdout.flush()
                walks.append(self.bias_random_walk(walk_length=walk_length, start_node=node))

        return walks

    # def get_alias_edge(self, src, dst):
    # 	'''
    # 	Get the alias edge setup lists for a given edge.
    # 	'''
    # 	G = self.G
    # 	p = self.p
    # 	q = self.q

    # 	unnormalized_probs = []
    # 	for dst_nbr in sorted(G.neighbors(dst)):
    # 		if dst_nbr == src:
    # 			unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
    # 		elif G.has_edge(dst_nbr, src):
    # 			unnormalized_probs.append(G[dst][dst_nbr]['weight'])
    # 		else:
    # 			unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
    # 	norm_const = sum(unnormalized_probs)
    # 	normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

    # 	return alias_setup(normalized_probs)

    # 计算转移概率
    def preprocess_transition_probs(self):
        '''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
        G = self.G
        is_directed = self.is_directed
        # 标准化转移概率
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        # print(alias_nodes)

        # alias_edges = {}
        # triads = {}

        # if is_directed:
        # 	for edge in G.edges():
        # 		alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        # else:
        # 	for edge in G.edges():
        # 		alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        # 		alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        # print(alias_edges)

        self.alias_nodes = alias_nodes
        # self.alias_edges = alias_edges

        return

    def truncated_random_walk(self):
        pass


def alias_setup(probs):
    """
	Compute utility lists for non-uniform sampling from discrete distributions.
	非均衡采样
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	@ probs: 当前节点到各邻居的转移概率
	"""
    # K: 某节点邻居节点个数
    length = len(probs)
    # q:
    q = np.zeros(length)
    # J:
    J = np.zeros(length, dtype=np.int)

    smaller = []
    larger = []

    # kk: index
    for index, prob in enumerate(probs):
        q[index] = length * prob
        # 判断节点到index=kk邻居的转移概率是否大于平均值
        if q[index] < 1.0:
            smaller.append(index)
        else:
            larger.append(index)

    while len(smaller) > 0 and len(larger) > 0:
        # small, large: 邻居节点 index
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        # K*prob_large + k*prob_small - 1.0
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
	Draw sample from a non-uniform discrete distribution using alias sampling.
	"""
    length = len(J)

    rand_index = int(np.floor(np.random.rand() * length))
    if np.random.rand() < q[rand_index]:
        return rand_index
    else:
        return J[rand_index]


if __name__ == '__main__':
    a, b = alias_setup([0.2, 0.3, 0.25, 0.25])
    print(alias_draw(a, b))


