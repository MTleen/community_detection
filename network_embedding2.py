import tensorflow as tf
import numpy as np
import random
import networkx as nx
import scipy.io as sio
import os, sys
import argparse
from expriments.community_detection import random_walk as walk


def get_batch(arr, n_seqs, n_steps):
    #	arr = arr.reshape([-1])
    #	batches = n_steps * n_seqs
    #	n_batches = int(len(arr) / batches)

    #	arr = arr[:n_batches*batches]
    #	arr = arr.reshape([n_seqs,-1])
    n = int(arr.shape[0] / n_seqs) * n_seqs
    nn = int(arr.shape[1] / n_steps) * n_steps
    arr = arr[:n, :nn]

    for n in range(0, arr.shape[0], n_seqs):
        for nn in range(0, arr.shape[1], n_steps):
            #		x = arr[:,n:n+n_steps]
            x = arr[n:n + n_seqs, nn:nn + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


def build_encode_arr(corpus, vocab_to_int):
    encode_input = np.zeros([len(corpus), len(corpus[0])], dtype=np.int32)
    for i, path in enumerate(corpus):
        tmp = list(map(lambda x: vocab_to_int[x], path))
        encode_input[i] = np.array(tmp, dtype=np.int32)
    # encode_output = np.transpose(encode_input)
    # np.random.shuffle(encode_output)

    # return encode_input,np.transpose(encode_output)
    return encode_input


def input_layer(n_steps, n_seqs):
    input = tf.placeholder(dtype=tf.int32, shape=(n_seqs, n_steps), name='input')
    targets = tf.placeholder(dtype=tf.int32, shape=(n_seqs, n_steps), name='targets')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    return input, targets, keep_prob


def basic_cell(lstm_cell_num, keep_prob, mode, n_seqs, n_steps, d):
    if mode == 'lstm':
        lstm = tf.contrib.rnn.LSTMCell(lstm_cell_num, initializer=tf.orthogonal_initializer, forget_bias=0.0)
    elif mode == 'rnn':
        lstm = tf.contrib.rnn.BasicRNNCell(lstm_cell_num)
    else:
        raise Exception('Unkown mode:{},only support rnn and lstm mode'.format(mode))

    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)


def hidden_layer(lstm_cell_num, lstm_layer_num, n_seqs, keep_prob, mode, n_steps, d):
    multi_lstm = tf.contrib.rnn.MultiRNNCell(
        [basic_cell(lstm_cell_num, keep_prob, mode, n_seqs, n_steps, d) for _ in range(lstm_layer_num)])
    initial_state = multi_lstm.zero_state(n_seqs, tf.float32)

    return multi_lstm, initial_state


def output_layer(lstm_output, in_size, out_size):
    out = tf.concat(lstm_output, 1)
    x = tf.reshape(out, [-1, in_size])

    with tf.variable_scope('softmax'):
        W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, W) + b
    prob_distrib = tf.nn.softmax(logits, name='predictions')

    return logits, prob_distrib


def cal_loss(logits, targets, class_num, t, embedding, alpha):
    y_one_hot = tf.one_hot(targets, class_num)
    y = tf.reshape(y_one_hot, logits.get_shape())
    # y = tf.reshape(targets,[logits.get_shape()[0],-1])

    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits/t,labels=y) + alpha*tf.nn.l2_loss(embedding)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)

    return tf.reduce_mean(loss)


def optimizer(loss, learning_rate, grad_clip):
    with tf.name_scope('gradient'):
        tvars = tf.trainable_variables()  # return the trainable variables,not all variables
        unclip_grad = tf.gradients(loss, tvars)
        grad, _ = tf.clip_by_global_norm(unclip_grad, grad_clip)
    # tf.summary.histogram('unclip_grad',unclip_grad)
    # tf.summary.histogram('grad',grad)

    train_op = tf.train.AdamOptimizer(learning_rate)

    return train_op.apply_gradients(zip(grad, tvars))


class LSTM:
    def __init__(self, class_num, n_steps, n_seqs, lstm_cell_num, lstm_layer_num, learning_rate, grad_clip, mode, d, t,
                 alpha):
        tf.reset_default_graph()
        # input layer
        self.input, self.targets, self.keep_prob = input_layer(n_steps, n_seqs)

        # lstm layer
        lstm_cell, self.initial_state = hidden_layer(lstm_cell_num, lstm_layer_num, n_seqs, self.keep_prob, mode,
                                                     n_steps, d)

        with tf.variable_scope('embedding_layer'):
            embedding = tf.get_variable(name='embedding', shape=[class_num, d],
                                        initializer=tf.random_uniform_initializer())

        x_input = tf.nn.embedding_lookup(embedding, self.input)
        if mode == 'clstm':
            x_input = tf.expand_dims(x_input, -1)
        # x_input_ = tf.unstack(x_input[...,None],128,2)

        self.embedding = embedding

        output, state = tf.nn.dynamic_rnn(lstm_cell, x_input, initial_state=self.initial_state)
        # output,state = tf.nn.dynamic_rnn(lstm_cell,x_input_,initial_state=self.initial_state)
        self.out = output
        self.final_state = state

        self.logits, self.pred = output_layer(output, lstm_cell_num, class_num)

        with tf.name_scope('loss'):
            self.loss = cal_loss(self.logits, self.targets, class_num, t, x_input, alpha)

        self.optimizer = optimizer(self.loss, learning_rate, grad_clip)


class Dis:
    def __init__(self, lr, class_num, d, beta):
        self.adj = tf.placeholder(dtype=tf.float32, name='adj')
        self.index = tf.placeholder(dtype=tf.int32, name='index')

        with tf.name_scope('rep'):
            with tf.variable_scope('embedding_layer', reuse=True):
                embedding = tf.get_variable(name='embedding', shape=[class_num, d],
                                            initializer=tf.random_uniform_initializer)
        # tf.summary.histogram('rep',embedding)

        D = tf.diag(tf.reduce_sum(self.adj, 1))
        L = D - self.adj

        batch_emb = tf.nn.embedding_lookup(embedding, self.index)

        with tf.name_scope('lap_loss'):
            self.lap_loss = 2 * tf.trace(
                tf.matmul(tf.matmul(tf.transpose(batch_emb), L), batch_emb)) + beta * tf.nn.l2_loss(batch_emb)
        tvars = tf.trainable_variables('embedding_layer')
        grad = tf.gradients(self.lap_loss, tvars)

        self.lap_optimizer = tf.train.RMSPropOptimizer(lr).apply_gradients(zip(grad, tvars))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', default='gml', help='the file format', type=str)
    # 输入目录
    parser.add_argument('--dataset', default='football', help='数据集名称', type=str)
    parser.add_argument('--input', default='football.gml', help='the file name', type=str)
    parser.add_argument('--output', default='./embedding/football_embedding.npy', help='saved as npy', type=str)
    # 嵌入模型，rnn 或 lstm
    parser.add_argument('--mode', default='lstm', help='which mode will used', type=str)
    # 每个节点游走次数
    parser.add_argument('--node_num', default=100, help='nums of per node', type=int)
    # 随机游走长度
    parser.add_argument('--path_length', default=100, help='length of path', type=int)
    parser.add_argument('--timesteps', default=100, help='length of one sequence', type=int)
    parser.add_argument('--sequences', default=100, help='how many sequences in one batch', type=int)
    parser.add_argument('--hidden_size', default=512, help='hidden size', type=int)
    parser.add_argument('--batches', default=128, help='batches size', type=int)
    parser.add_argument('--layer', default=1, help='how many layers', type=int)
    parser.add_argument('--representation_size', default=128, help='representation size', type=int)
    parser.add_argument('--t', default=1.0, help='smooth softmax', type=float)
    parser.add_argument('--alpha', default=1.0, help='LSTM L2 reg ', type=float)
    parser.add_argument('--beta', default=1.0, help='Lap L2 reg', type=float)
    parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
    parser.add_argument('--lap_lr', default=0.001, help='learning rate', type=float)
    parser.add_argument('--keep_prob', default=0.5, help='keep prob', type=float)
    parser.add_argument('--grad_clip', default=5, help='gradient clipping', type=int)
    parser.add_argument('--gen_epoches', default=5, help='iter nums', type=int)
    parser.add_argument('--dis_epoches', default=5, help='iter nums', type=int)
    parser.add_argument('--epoches', default=10, help='iter nums', type=int)
    parser.add_argument('--k_neighbors', default=200, help='节点 knn 个数', type=int)
    args = parser.parse_args()

    # if args.format == 'mat':
    #     mat = sio.loadmat(args.input)['network']
    #     G = nx.from_scipy_sparse_matrix(mat)
    # elif args.format == 'adjlist':
    #     G = nx.read_adjlist(args.input)
    # elif args.format == 'edgelist':
    #     G = nx.read_edgelist(args.input)
    # else:
    #     raise Exception("Unkown file format:{}.Valid format is 'mat','adjlist','edgelist'".format(args.format))
    # mat = sio.loadmat('./data/3-NG.mat')['network']
    # G = nx.from_scipy_sparse_matrix(mat)
    # mat = nx.to_scipy_sparse_matrix(G)

    data_2_graph = {
        'gml': lambda input_data: nx.read_gml(input_data)
    }

    # 从原始数据构造图
    data_dir = './data'
    if args.format in data_2_graph.keys():
        print('从文件 %s 构造图' % os.path.join(data_dir, args.input))
        G = data_2_graph[args.format](os.path.join(data_dir, args.input))
        # 生成邻接矩阵
        nodes = list(G.nodes())
        node_values = G._node
        adj = np.zeros([len(nodes), len(nodes)])
        edges = G.edges()
        for edge in edges:
            x = nodes.index(edge[0])
            y = nodes.index(edge[1])
            adj[(x, y), (y, x)] = 1
    else:
        raise Exception('未知原始数据格式')
    print('Start Walking...\n **********')
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = min(G.degree(edge[0]), G.degree(edge[1])) / max(G.degree(edge[0]),
                                                                                         G.degree(edge[1]))
    G_ = walk.Walk(G, False, 0.25, 0.25)
    G_.preprocess_transition_probs()
    corpus = G_.simulate_walks(args.node_num, args.path_length)
    vocab = list(G.nodes())
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))

    encode_arr_input = build_encode_arr(corpus, vocab_to_int)
    tf.set_random_seed(1)
    lstm = LSTM(len(vocab), args.timesteps, args.sequences, args.hidden_size, args.layer, args.lr, args.grad_clip,
                args.mode, args.representation_size, args.t, args.alpha)
    dis = Dis(args.lap_lr, len(vocab), args.representation_size, args.beta)
    # saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print('Launch LSTM...')

    for i in range(args.epoches):
        print('*' * 30, '当前迭代：', i, '*' * 30)
        for j in range(args.gen_epoches):
            new_state = sess.run(lstm.initial_state, {lstm.keep_prob: args.keep_prob})
            for x, y in get_batch(encode_arr_input, args.sequences, args.timesteps):
                feed_dict = {lstm.input: x, lstm.targets: y, lstm.keep_prob: args.keep_prob,
                             lstm.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([lstm.loss, lstm.final_state, lstm.optimizer], feed_dict=feed_dict)
                print('batch_loss: ', batch_loss)

        for k in range(args.dis_epoches):
            # adj = mat.toarray()
            for index in range(0, adj.shape[0], args.batches):
                batch_adj = adj[index:index + args.batches, index:index + args.batches]
                feed_dict = {dis.adj: batch_adj, dis.index: np.arange(adj.shape[0])[index:index + args.batches]}
                lap_loss, _ = sess.run([dis.lap_loss, dis.lap_optimizer], feed_dict=feed_dict)
                print('lap_loss:', lap_loss)

    network_embedding = sess.run(lstm.embedding)
    # 提取节点标签
    labels = []
    for i, node in enumerate(nodes):
        value = node_values[node]['value']
        labels.append(value)
    # 节点 knn
    knn_graph = cal_knn(network_embedding, args.k_neighbors)
    print(knn_graph)
    np.save(os.path.join('./embedding', args.dataset + '_labels.npy'), labels)
    np.save(args.output, network_embedding)
    np.save(os.path.join('./embedding', args.dataset + '_knn_graph.npy'), knn_graph)


