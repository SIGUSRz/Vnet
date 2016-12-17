import tensorflow as tf
import numpy as np
import argparse
import os
import h5py

import text_processor
import image_processor
import answer_generator
import data_loader

from gpu import define_gpu
define_gpu(1)

def build_batch(batch_num, batch_size, img_feature, img_id_map, qa_data, vocab_data, split):
    qa = qa_data[split]
    batch_start = (batch_num * batch_size) % len(qa)
    batch_end = min(len(qa), batch_start + batch_size)
    size = batch_end - batch_start
    sentence = np.ndarray((size, vocab_data['max_que_length']), dtype='int32')
    answer = np.zeros((size, len(vocab_data['ans_vocab'])))
    img = np.ndarray((size, 7, 7, 512))

    counter = 0
    for i in range(batch_start, batch_end):
        sentence[counter, :] = qa[i]['question'][:]
        answer[counter, qa[i]['answer']] = 1.0
        img_index = img_id_map[qa[i]['image_id']]
        img[counter, :] = img_feature[img_index][:]
        counter += 1
    return sentence, answer, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data Directory')
    parser.add_argument('--log_dir', type=str, default='log', help='Checkpoint File Directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')

    parser.add_argument('--batch_size', type=int, default=64, help='Image Training Batch Size')
    parser.add_argument('--num_output', type=int, default=1000, help='Number of Output')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout Rate')
    parser.add_argument('--init_bound', type=float, default=1.0, help='Parameter Initialization Distribution Bound')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning Rate Decay Factor')
    parser.add_argument('--num_epoch', type=int, default=30, help='Number of Training Epochs')

    parser.add_argument('--hidden_dim', type=int, default=1024, help='RNN Hidden State Dimension')
    parser.add_argument('--rnn_size', type=int, default=512, help='RNN Cell Size. Question Feature Embedding Dimension. \
                                                                Image Feature Channel Number')
    parser.add_argument('--rnn_layer', type=int, default=2, help='Number of RNN Layers. 2 For Bidirection RNN Cell')
    parser.add_argument('--vocab_embed_size', type=int, default=200, help='Vocabulary Embedding Dimension. \
                                                                Used When Embedding Vocabulary In Sentence.')

    parser.add_argument('--use_attention', type=bool, default=True, help='Layer to Extract in Image CNN Model')
    parser.add_argument('--att_hidden_dim', type=int, default=16, help='Hidden Dimension of Attention Hidden State')
    parser.add_argument('--att_round', type=int, default=0, help='Round to Apply Attention Mechanism')
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.makedirs(os.path.join(args.log_dir, 'model'))
        os.makedirs(os.path.join(args.log_dir, 'summary'))

    #Reading Question Answer Data
    print 'Reading Question Answer Data'
    qa_data, vocab_data = data_loader.load_qa_data(args.data_dir, args.top_num)
    train_img_feature, train_img_id_list = None, None
    dev_img_feature, dev_img_id_list = None, None
    if args.use_attention:
        print 'Loading Image Feature Data of VGG-16 Model Layer pool5'
        file_code = 0
        train_img_feature, train_img_id_list = image_processor.VGG_16_extract_pool5('train', args, file_code)

    print 'Building Image ID Map and Answer Map'
    train_img_id_map = {}
    for i in xrange(len(train_img_id_list)):
        train_img_id_map[train_img_id_list[i]] = i
    ans_map = {vocab_data['ans_vocab'][ans] : ans for ans in vocab_data['ans_vocab']}

    print 'Building Answer Generator Model'
    generator = answer_generator.Answer_Generator({
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'img_dim': train_img_feature.shape[1:],
        'rnn_size': args.rnn_size,
        'rnn_layer': args.rnn_layer,
        'vocab_embed_size': args.vocab_embed_size,
        'que_vocab_size': len(vocab_data['que_vocab']),
        'ans_vocab_size': len(vocab_data['ans_vocab']),
        'max_que_length': vocab_data['max_que_length'],
        'att_hidden_dim': args.att_hidden_dim,
        'att_round': args.att_round,
        'num_output': args.num_output,
        'dropout_rate': args.dropout_rate,
        'data_dir': args.data_dir,
        'top_num': args.top_num,
        'init_bound': args.init_bound
        })
    generator.build_coattention_model()
    loss, feed_img, feed_que, feed_label = generator.train_coattention_model()
    sess = tf.Session()

    tf.initialize_all_variables().run(session=sess)
    train_batch_num = 0
    que_batch, ans_batch, img_batch = build_batch(train_batch_num, args.batch_size, \
                                        train_img_feature, train_img_id_map, qa_data, vocab_data, 'train')
    loss_value = sess.run([loss],
                                            feed_dict={
                                                feed_img: img_batch,
                                                feed_que: que_batch,
                                                feed_label: ans_batch
                                            })
if __name__ == '__main__':
    main()