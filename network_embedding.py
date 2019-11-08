# -*- coding: utf-8 -*-
"""
@author: HuShengxiang
@file_name: network_embedding
@time: 2019/11/5 14:41
"""
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import networkx as nx
import scipy.io as sio
from random_walk import RandomWalk
from embedding_model import EmbeddingModel
import os, sys
import argparse

def build_encode_arr(corpus, vocab_2_int):
    encoded_input = np.zeros([len(corpus), len(corpus[0])], dtype=np.int32)
    for i, seq in enumerate(corpus):
        tmp = list(map(lambda x: vocab_2_int[x], seq))
        encoded_input[i] = np.array(tmp, dtype=np.int32)
    return encoded_input

def get_batch(arr, n_seqs, n_steps):
    # 计算完整 batch 的数量
    n = int(arr.shape[0] / n_seqs) * n_seqs
    nn = int(arr.shape[1] / n_steps) * n_steps
    arr = arr[:n, :nn]
    for n in range(0, arr.shape[0], n_seqs):
        for nn in range(0, arr.shape[1], n_steps):
            x = arr[n: n + n_seqs, nn: nn + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

def cal_loss(logits, targets, class_num):
    y_one_hot = keras.utils.to_categorical(targets, class_num)
    y = tf.reshape(y_one_hot, logits.shape)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    # return keras.losses.
    return tf.reduce_mean(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_format', default='gml', help='原始输入数据格式')
    parser.add_argument('--input_file_name', default='football/football.gml', help='原始输入数据路径')
    parser.add_argument('--output', default='./results/network_embedding.csv', help='输出路径')
    parser.add_argument('--walk_nums', default=100, type=int, help='以不同节点为起始点的随机游走次数')
    parser.add_argument('--walk_length', default=100, type=int, help='随机游走路径长度')
    parser.add_argument('--batch_size', default=128, type=int, help='每个 batch 大小')
    parser.add_argument('--epochs', default=5, type=int, help='模型训练迭代次数')
    args = parser.parse_args()

    data_2_graph = {
        'gml': lambda input_data: nx.read_gml(input_data)
    }

    # 从原始数据构造图
    data_dir = './data'
    if args.input_format in data_2_graph.keys():
        print('从文件 %s 构造图' % os.path.join(data_dir, args.input_file_name))
        graph = data_2_graph[args.input_format](os.path.join(data_dir, args.input_file_name))
        print(graph.nodes())
    else:
        raise Exception('未知原始数据格式')

    # 随机游走
    print('start walking...\n****************')
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] *= min(graph.degree(edge[0]), graph.degree(edge[1])) / max(
            graph.degree(edge[0]), graph.degree(edge[1]))
    walk = RandomWalk(graph, False)
    walk.preprocess_transition_probs()

    # 随机游走生成语料
    corpus = walk.simulate_walks(args.walk_nums, args.walk_length)
    vocab = list(graph.nodes())
    vocab_2_int = {node: index for index, node in enumerate(vocab)}
    int_2_vocab = dict(enumerate(vocab))
    encoded_input_arr = build_encode_arr(corpus, vocab_2_int)

    # 训练模型
    optimizer = keras.optimizers.RMSprop()
    # model = EmbeddingModel(name='embedding model', inputs_dim=args.walk_length, lstm_cell_num=, lstm_layer_num=, keep_prob=, output_dim=)
    for epoch in range(args.epochs):
        print('start of epoch %d' % epoch)
        for x, y in get_batch(encoded_input_arr, args.batch_size, args.walk_length):
            pass

