# -*- coding: utf-8 -*-
"""
@author: HuShengxiang
@file_name: embedding_model
@time: 2019/11/5 15:42
"""
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

class BasicLSTMCell(layers.Layer):
    def __init__(self, lstm_cell_num, keep_prob, initializer=keras.initializers.orthogonal):
        super(BasicLSTMCell, self).__init__()
        self.lstm_cell = layers.LSTMCell(lstm_cell_num, kernel_initializer=initializer, forget_bias=0.0)
        self.lstm_wrapper = layers.DropoutWrapper(output_keep_prob=keep_prob)

    def call(self, inputs, **kwargs):
        return self.lstm_wrapper(self.lstm_cell(inputs))

class EmbeddingModel(keras.Model):
    def __init__(self, name, inputs_dim, lstm_cell_num, keep_prob, lstm_layer_num, output_dim):
        super(EmbeddingModel, self).__init__(name=name)
        self.inputs = keras.Input(inputs_dim)
        self.stacked_lstm = layers.StackedRNNCells(
            [BasicLSTMCell(lstm_cell_num, keep_prob) for _ in range(lstm_layer_num)]
        )
        self.logits = layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        h1 = self.inputs(inputs)
        h2 = self.stacked_lstm(h1)
        logits = self.logits(h2)
        prob_distrib = tf.nn.softmax(logits, name='predictions')
        return logits, prob_distrib