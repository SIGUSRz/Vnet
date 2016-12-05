import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from tensorflow.models.rnn import rnn_cell
from sklearn.metrics import average_precision_score

class Basic_LSTM:
    def __init__(self, params):
        self.rnn_size = params['rnn_size']
        self.num_rnn_layer = params['num_rnn_layer']
        self.max_que_length = params['max_que_length']
        self.que_vocab_size = params['que_vocab_size']
        self.que_embed_size = params['que_embed_size']

    def build(self):
        sentence = tf.placeholder('int32', [None, self.seq_length], name='sentence')

        word_embedding = []
        for i in range()