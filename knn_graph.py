from sklearn import neighbors
import numpy as np
import os


def cal_knn(embedding, k):
    node_num = embedding.shape[0]
    if k >= node_num:
        k = node_num - 1
    knn_graph = np.zeros([node_num, k + 1], dtype=np.int)
    for i, node in enumerate(embedding):
        dis = np.sum((embedding - node) ** 2, axis=1)
        dis_sorted = np.argsort(dis)
        knn_graph[i] = dis_sorted[:k + 1]
    return knn_graph


def main():
    n_neighbors = 200
    embedding = np.load('./embedding/football_embedding.npy')
    knn_graph = cal_knn(embedding, n_neighbors)
    print(knn_graph)
    np.save('./embedding/football_knn_graph.npy', knn_graph)


if __name__ == '__main__':
    main()
    # working_dir = os.path.dirname(os.path.abspath(__file__))
    # print(os.listdir(working_dir))