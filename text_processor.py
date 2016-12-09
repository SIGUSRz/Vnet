import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from sklearn.metrics import average_precision_score

class Deeper_LSTM:
    def __init__(self, params):
        self.rnn_size = params['rnn_size'] # Number of RNN Cell Units
        self.rnn_layer = params['rnn_layer']
        self.init_bound = params['init_bound'] # Random Initialization Bound Values
        self.max_que_length = params['max_que_length']
        self.que_vocab_size = params['que_vocab_size']
        self.que_embed_size = params['que_embed_size']
        self.dropout_rate = params['dropout_rate']
        self.batch_size = params['batch_size']

        self.que_embed_W = tf.Variable(tf.random_uniform([self.que_vocab_size, self.que_embed_size], \
                                                            -self.init_bound, self.init_bound), 
                                                            name='que_embed_W')
        self.base_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        self.drop_cell = tf.nn.rnn_cell.DropoutWrapper(self.base_cell, output_keep_prob=self.dropout_rate)
        self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.drop_cell] * self.rnn_layer, state_is_tuple=True)
        # self._initial_state = self.stacked_cell.zero_state(self.batch_size, tf.float32)

    def train(self):
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