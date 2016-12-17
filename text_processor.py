import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import time
import math
import cv2
import codecs, json
from sklearn.metrics import average_precision_score

class Deeper_LSTM:
    def __init__(self, params):
        self.rnn_size = params['rnn_size'] # RNN Cell Dimension
        self.rnn_layer = params['rnn_layer'] # Number of RNN Cells
        self.init_bound = params['init_bound'] # Random Initialization Bound Values
        self.max_que_length = params['max_que_length'] # T: RNN Timesteps = Question Sentence Length
        self.que_vocab_size = params['que_vocab_size']
        self.vocab_embed_size = params['vocab_embed_size']
        self.dropout_rate = params['dropout_rate']
        self.batch_size = params['batch_size']

    def build_base_cell(self):
        self.que_embed_W = tf.Variable(tf.random_uniform([self.que_vocab_size, self.vocab_embed_size], \
                                                            -self.init_bound, self.init_bound), \
                                                            name='que_embed_W')
        self.base_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        self.drop_cell = tf.nn.rnn_cell.DropoutWrapper(self.base_cell, output_keep_prob=self.dropout_rate)
        self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.drop_cell] * self.rnn_layer, state_is_tuple=True)

    def train_base_cell(self):
        sentence_batch = tf.placeholder('int32', [None, self.max_que_length])
        state = self.stacked_cell.zero_state(tf.shape(sentence_batch)[0], tf.float32)
        loss = 0.0
        with tf.variable_scope("RNN"):
            for time_step in range(self.max_que_length): # Max Question Length is the Number of Steps
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                linear_embedding = tf.nn.embedding_lookup(self.que_embed_W, sentence_batch[:, time_step - 1])
                drop_embedding = tf.nn.dropout(linear_embedding, 1 - self.dropout_rate)
                que_embedding = tf.tanh(drop_embedding)
        output, state = self.stacked_cell(que_embedding, state)
        return state, sentence_batch

    def build_hierarchy_cell(self):
        self.uni_filter = tf.Variable(tf.random_uniform([1, self.vocab_embed_size, self.rnn_size], \
                                                            -self.init_bound, self.init_bound), \
                                                            name='uni_filter')
        self.bi_filter = tf.Variable(tf.random_uniform([2, self.vocab_embed_size, self.rnn_size], \
                                                            -self.init_bound, self.init_bound), \
                                                            name='bi_filter')
        self.tri_filter = tf.Variable(tf.random_uniform([3, self.vocab_embed_size, self.rnn_size], \
                                                            -self.init_bound, self.init_bound), \
                                                            name='tri_filter')
        self.que_embed_W = tf.Variable(tf.random_uniform([self.que_vocab_size, self.vocab_embed_size], \
                                                            -self.init_bound, self.init_bound), \
                                                            name='que_embed_W')
        self.base_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        self.drop_cell = tf.nn.rnn_cell.DropoutWrapper(self.base_cell, output_keep_prob=self.dropout_rate)
        self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.drop_cell] * self.rnn_layer, state_is_tuple=True)

    def train_hierarchy_cell(self):
        sentence_batch = tf.placeholder('int32', [None, self.max_que_length])
        state = self.stacked_cell.zero_state(tf.shape(sentence_batch)[0], tf.float32)
        loss = 0.0
        with tf.variable_scope("RNN"):
            for time_step in range(self.max_que_length): # Max Question Length is the Number of Steps
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                word_embedding = tf.nn.embedding_lookup(self.que_embed_W, sentence_batch[:, time_step - 1])
                word_embedding = tf.expand_dims(word_embedding, 0)
                uni_embedding = tf.nn.conv1d(word_embedding, self.uni_filter, stride=1, padding='SAME')
                bi_embedding = tf.nn.conv1d(word_embedding, self.bi_filter, stride=1, padding='SAME')
                tri_embedding = tf.nn.conv1d(word_embedding, self.tri_filter, stride=1, padding='SAME')
                uni_embedding = tf.reshape(uni_embedding, [tf.shape(uni_embedding)[1], 1, self.rnn_size])
                bi_embedding = tf.reshape(bi_embedding, [tf.shape(bi_embedding)[1], 1, self.rnn_size])
                tri_embedding = tf.reshape(tri_embedding, [tf.shape(tri_embedding)[1], 1, self.rnn_size])
                que_embedding = np.stack([uni_embedding, np.stack([bi_embedding, tri_embedding], axis=1)], axis=1)
                print tf.Tensor.get_shape(uni_embedding)
                print tf.Tensor.get_shape(bi_embedding)
                print tf.Tensor.get_shape(tri_embedding)
                # print tf.Tensor.get_shape(que_embedding)
        output, state = self.stacked_cell(uni_embedding, state)
        return state, sentence_batch