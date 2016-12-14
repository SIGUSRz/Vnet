import tensorflow as tf

import image_processor
import text_processor

class Answer_Generator():
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.img_dim = params['img_dim']
        self.hidden_dim = params['hidden_dim']
        self.rnn_size = params['rnn_size']
        self.rnn_layer = params['rnn_layer']
        self.init_bound = params['init_bound']
        self.num_output = params['num_output']
        self.dropout_rate = params['dropout_rate']
        self.ans_vocab_size = params['ans_vocab_size']
        self.max_que_length = params['max_que_length']
        self.img_processor = image_processor.Vgg16()
        self.que_processor = text_processor.Deeper_LSTM({
            'rnn_size': self.rnn_size,
            'rnn_layer': self.rnn_layer,
            'init_bound': self.init_bound,
            'que_vocab_size': params['que_vocab_size'],
            'que_embed_size': params['que_embed_size'],
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'max_que_length': self.max_que_length
        })


        self.que_W = tf.Variable(tf.random_uniform([2 * self.rnn_layer * self.rnn_size , self.hidden_dim], \
                                                        -self.init_bound, self.init_bound), 
                                                        name='text_W')
        self.que_b = tf.Variable(tf.random_uniform([self.hidden_dim], -self.init_bound, self.init_bound), name='text_b')
        self.img_W = tf.Variable(tf.random_uniform([self.img_dim, self.hidden_dim], \
                                                        -self.init_bound, self.init_bound), 
                                                        name='img_W')
        self.img_b = tf.Variable(tf.random_uniform([self.hidden_dim], -self.init_bound, self.init_bound), name='img_b')
        self.score_W = tf.Variable(tf.random_uniform([self.hidden_dim, self.num_output], \
                                                        -self.init_bound, self.init_bound),
                                                        name='score_W')
        self.score_b = tf.Variable(tf.random_uniform([self.num_output], -self.init_bound, self.init_bound, name='score_b'))

    def train_model(self):
        img_state = tf.placeholder('float32', [None, self.img_dim], name='img_state')
        label_batch = tf.placeholder('float32', [None, self.ans_vocab_size], name='label_batch')
        real_size = tf.shape(img_state)[0]

        que_state, sentence_batch = self.que_processor.train()
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