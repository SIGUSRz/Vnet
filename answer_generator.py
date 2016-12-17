import tensorflow as tf

import image_processor
import text_processor

class Answer_Generator():
    def __init__(self, params):
        self.params = params
        self.batch_size = params['batch_size']
        self.img_dim = params['img_dim'] # N: Image Feature Matrix Dimension
        self.hidden_dim = params['hidden_dim']
        self.rnn_size = params['rnn_size'] # d: RNN Cell Dimension
        self.rnn_layer = params['rnn_layer']
        self.init_bound = params['init_bound']
        self.num_output = params['num_output']
        self.dropout_rate = params['dropout_rate']
        self.ans_vocab_size = params['ans_vocab_size']
        self.max_que_length = params['max_que_length'] # T: RNN Timesteps = Question Sentence Length
        self.img_processor = image_processor.Vgg16()
        self.que_processor = text_processor.Deeper_LSTM({
            'rnn_size': self.rnn_size,
            'rnn_layer': self.rnn_layer,
            'init_bound': self.init_bound,
            'que_vocab_size': params['que_vocab_size'],
            'vocab_embed_size': params['vocab_embed_size'],
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'max_que_length': self.max_que_length
        })

    def build_base_model(self):
        self.que_processor.build_base_cell()
        self.que_W = tf.Variable(tf.random_uniform([2 * self.rnn_layer * self.rnn_size , self.hidden_dim], \
                                                        -self.init_bound, self.init_bound),
                                                        name='que_W')
        self.que_b = tf.Variable(tf.random_uniform([self.hidden_dim], -self.init_bound, self.init_bound), name='que_b')
        self.img_W = tf.Variable(tf.random_uniform([self.img_dim, self.hidden_dim], \
                                                        -self.init_bound, self.init_bound), 
                                                        name='img_W')
        self.img_b = tf.Variable(tf.random_uniform([self.hidden_dim], -self.init_bound, self.init_bound), name='img_b')
        self.score_W = tf.Variable(tf.random_uniform([self.hidden_dim, self.num_output], \
                                                        -self.init_bound, self.init_bound),
                                                        name='score_W')
        self.score_b = tf.Variable(tf.random_uniform([self.num_output], -self.init_bound, self.init_bound, name='score_b'))

    def train_base_model(self):
        img_state = tf.placeholder('float32', [None, self.img_dim], name='img_state')
        label_batch = tf.placeholder('float32', [None, self.ans_vocab_size], name='label_batch')
        real_size = tf.shape(img_state)[0]

        que_state, sentence_batch = self.que_processor.train_base_cell()
        drop_que_state = tf.nn.dropout(que_state, 1 - self.dropout_rate)
        drop_que_state = tf.reshape(drop_que_state, [real_size, 2 * self.rnn_layer * self.rnn_size])
        linear_que_state = tf.nn.xw_plus_b(drop_que_state, self.que_W, self.que_b)
        que_feature = tf.tanh(linear_que_state)

        drop_img_state = tf.nn.dropout(img_state, 1 - self.dropout_rate)
        linear_img_state = tf.nn.xw_plus_b(drop_img_state, self.img_W, self.img_b)
        img_feature = tf.tanh(linear_img_state)

        score = tf.mul(que_feature, img_feature)
        drop_score = tf.nn.dropout(score, 1 - self.dropout_rate)
        logits = tf.nn.xw_plus_b(drop_score, self.score_W, self.score_b)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, label_batch, name='entropy')
        ans_probability = tf.nn.softmax(logits, name='answer_prob')

        predict = tf.argmax(ans_probability, 1)
        correct_predict = tf.equal(tf.argmax(ans_probability, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        loss = tf.reduce_sum(cross_entropy, name='loss')
        return loss, accuracy, predict, img_state, sentence_batch, label_batch

    def build_coattention_model(self):
        self.que_processor.build_hierarchy_cell()
        self.att_hidden_dim = self.params['att_hidden_dim']
        self.hidden_W = tf.Variable(tf.random_uniform([self.rnn_size, self.rnn_size], \
                                                        -self.init_bound, self.init_bound),
                                                        name='hidden_W')
        self.que_W = tf.Variable(tf.random_uniform([self.att_hidden_dim , self.rnn_size], \
                                                        -self.init_bound, self.init_bound),
                                                        name='que_W')
        self.que_b = tf.Variable(tf.random_uniform([self.img_dim, self.att_hidden_dim], \
                                                        -self.init_bound, self.init_bound), 
                                                        name='que_b')
        self.img_W = tf.Variable(tf.random_uniform([self.att_hidden_dim, self.rnn_size], \
                                                        -self.init_bound, self.init_bound),
                                                        name='img_W')
        self.img_b = tf.Variable(tf.random_uniform([self.max_que_length, self.att_hidden_dim], \
                                                        -self.init_bound, self.init_bound),
                                                        name='img_b')
        self.img_att_W = tf.Variable(tf.random_uniform([self.att_hidden_dim, 1], \
                                                        -self.init_bound, self.init_bound),
                                                        name='img_att_W')
        self.img_att_b = tf.Variable(tf.random_uniform([self.img_dim, 1], \
                                                        -self.init_bound, self.init_bound),
                                                        name='img_att_b')
        self.que_att_W = tf.Variable(tf.random_uniform([self.att_hidden_dim, 1], \
                                                        -self.init_bound, self.init_bound),
                                                        name='que_att_W')
        self.que_att_b = tf.Variable(tf.random_uniform([self.max_que_length, 1], \
                                                        -self.init_bound, self.init_bound),
                                                        name='que_att_b')
        self.drop_W = tf.Variable(tf.random_uniform([self.rnn_size, self.hidden_dim], \
                                                        -self.init_bound, self.init_bound),
                                                        name='drop_W')
        self.drop_b = tf.Variable(tf.random_uniform([self.hidden_dim], \
                                                        -self.init_bound, self.init_bound),
                                                        name='drop_b')
        self.score_W = tf.Variable(tf.random_uniform([self.hidden_dim, self.num_output], \
                                                        -self.init_bound, self.init_bound),
                                                        name='score_W')
        self.score_b = tf.Variable(tf.random_uniform([self.num_output], \
                                                        -self.init_bound, self.init_bound),
                                                        name='score_b')
    def train_coattention_model(self):
        img_state = tf.placeholder('float32', [None, self.img_dim[0], self.img_dim[1], self.img_dim[2]], name='img_state')
        img_state = tf.contrib.layers.batch_norm(img_state)
        label_batch = tf.placeholder('float32', [None, self.ans_vocab_size], name='label_batch')
        real_size = tf.shape(img_state)[0]

        que_state, sentence_batch = self.que_processor.train_hierarchy_cell()
        with tf.variable_scope('attention'):
            img_attention, que_attention, new_img_state, new_que_state, hidden_state = self.add_attention(img_state, que_state)
        score = tf.tanh(tf.matmul(img_attention, self.score_W) + tf.matmul(que_attention, self.score_W) + self.score_b)
        score = tf.nn.dropout(score, 1 - self.dropout_rate)
        logits = tf.relu(tf.nn.xw_plus_b(score, self.score_W, self.score_b))
        # logits = tf.contrib.layers.batch_norm(logtis)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, label_batch, name='entropy')
        ans_probability = tf.nn.softmax(logits, name='answer_prob')

        predict = tf.argmax(ans_probability, 1)
        correct_predict = tf.equal(tf.argmax(ans_probability, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        loss = tf.reduce_sum(cross_entropy, name='loss')
        return loss, accuracy, predict, img_state, sentence_batch, label_batch

    def add_attention(self, img_state, que_state, hidden_state=None):
        update = None
        if hidden_state == None:
            update = True
            hidden_state = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl', img_state, self.hidden_W), que_state)

        img_hidden_state = tf.nn.tanh(tf.einsum('aij,kj->ik', img_state, self.img_W) + \
                                                        tf.einsum('ij,ki->jk', tf.einsum('aij,ki->kj', que_state, \
                                                        hidden_state)) + self.que_b)
        que_hidden_state = tf.nn.tanh(tf.einsum('aij,kj->ik', que_state, self.que_W) + \
                                                        tf.einsum('ij,ik->jk', tf.einsum('aij,ki->kj', img_state, \
                                                        hidden_state)) + self.img_b)
        img_attention_W = tf.nn.softmax(tf.matmul(img_hidden_state, self.img_att_W) + self.img_att_b)
        que_attention_W = tf.nn.softmax(tf.matmul(que_hidden_state, self.que_att_W) + self.que_att_b)

        img_attention = tf.einsum('ijk,j->ik', img_state, tf.squeeze(img_attention_W))
        que_attention = tf.einsum('ijk,j->ik', que_state, tf.squeeze(que_attention_W))

        new_img_state = tf.mul(img_state, img_attention_W)
        new_que_state = tf.mul(que_state, que_attention_W)
        if update:
            hidden_state = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl', img_state, self.hidden_W), que_state)
        return img_attention, que_attention, new_img_state, new_que_state, hidden_state