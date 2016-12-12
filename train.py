import tensorflow as tf
import numpy as np
import data_loader
import argparse
import os

import image_processor
import answer_generator
from gpu import define_gpu
define_gpu(2)
#testing again...

def build_batch(batch_num, batch_size, img_feature, img_id_map, qa_data, vocab_data, split):
    qa = qa_data[split]
    batch_start = (batch_num * batch_size) % len(qa)
    batch_end = min(len(qa), batch_start + batch_size)
    size = batch_end - batch_start
    sentence = np.ndarray((size, vocab_data['max_que_length']), dtype='int32')
    answer = np.zeros((size, len(vocab_data['ans_vocab'])))
    img = np.ndarray((size, 4096))

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
    parser.add_argument('--data_dir', type=str, default='/home/zs/NLP/Vnet/data', help='Data Directory')
    parser.add_argument('--log_dir', type=str, default='log', help='Checkpoint File Directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')

    parser.add_argument('--batch_size', type=int, default=64, help='Image Training Batch Size')
    parser.add_argument('--num_output', type=int, default=1000, help='Number of Output')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout Rate')
    parser.add_argument('--init_bound', type=float, default=0.5, help='Parameter Initialization Distribution Bound')

    parser.add_argument('--hidden_dim', type=int, default=1024, help='RNN Hidden State Dimension')
    parser.add_argument('--rnn_size', type=int, default=512, help='Size of RNN Cell')
    parser.add_argument('--rnn_layer', type=int, default=2, help='Number of RNN Layers')
    parser.add_argument('--que_embed_size', type=int, default=200, help='Question Embedding Dimension')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning Rate Decay Factor')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of Training Epochs')
    parser.add_argument('--grad_norm', type=int, default=5, help='Maximum Norm of the Gradient')
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.makedirs(os.path.join(args.log_dir, 'model'))
        os.makedirs(os.path.join(args.log_dir, 'summary'))

    print 'Reading Question Answer Data'
    qa_data, vocab_data = data_loader.load_qa_data(args.data_dir, args.top_num)
    train_img_feature, train_img_id_list = image_processor.VGG_16_extract('train', args)

    print 'Building Image ID Map and Answer Map'
    img_id_map = {}
    for i in xrange(len(train_img_id_list)):
        img_id_map[train_img_id_list[i]] = i
    ans_map = {vocab_data['ans_vocab'][ans] : ans for ans in vocab_data['ans_vocab']}

    print 'Building Answer Generator Model'
    generator = answer_generator.Answer_Generator({
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'img_dim': train_img_feature.shape[1],
        'rnn_size': args.rnn_size,
        'rnn_layer': args.rnn_layer,
        'que_embed_size': args.que_embed_size,
        'que_vocab_size': len(vocab_data['que_vocab']),
        'ans_vocab_size': len(vocab_data['ans_vocab']),
        'max_que_length': vocab_data['max_que_length'],
        'num_output': args.num_output,
        'dropout_rate': args.dropout_rate,
        'data_dir': args.data_dir,
        'top_num': args.top_num,
        'init_bound': args.init_bound
        })

    lr = args.learning_rate
    loss, accuracy, predict, feed_img, feed_que, feed_label = generator.train_model()
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()

    tf.initialize_all_variables().run(session=sess)

    train_summary_writer = tf.train.SummaryWriter(os.path.join(args.log_dir, "summaries", "train"), sess.graph)
    dev_summary_writer = tf.train.SummaryWriter(os.path.join(args.log_dir, "summaries", "dev"), sess.graph)

    print 'Training Start...'
    saver = tf.train.Saver()
    for epoch in range(args.num_epoch):
        print 'Epoch %d #############' % epoch
        train_batch_num = 0
        dev_batch_num = 0
        dev_acc_list = []
        while train_batch_num * args.batch_size < len(qa_data['train']):
            que_batch, ans_batch, img_batch = build_batch(train_batch_num, args.batch_size, \
                                                train_img_feature, img_id_map, qa_data, vocab_data, 'train')
            _, loss_value, acc, pred = sess.run([train_op, loss, accuracy, predict],
                                                    feed_dict={
                                                        feed_img: img_batch,
                                                        feed_que: que_batch,
                                                        feed_label: ans_batch
                                                    })
            train_batch_num += 1
            if train_batch_num % 500 == 0:
                print "Batch: ", train_batch_num, " Loss: ", loss_value, " Learning Rate: ", lr
                train_loss_summary = tf.Summary()
                cost = train_loss_summary.value.add()
                cost.tag = "train_loss"
                cost.simple_value = float(loss_value)
                train_summary_writer.add_summary(train_loss_summary)
        while dev_batch_num * args.batch_size < len(qa_data['dev']):
            que_batch, ans_batch, img_batch = build_batch(dev_batch_num, args.batch_size, \
                                                train_img_feature, img_id_map, qa_data, vocab_data, 'train')
            loss_value, acc, pred = sess.run([loss, accuracy, predict],
                                                    feed_dict={
                                                        feed_img: img_batch,
                                                        feed_que: que_batch,
                                                        feed_label: ans_batch
                                                    })
            dev_batch_num += 1
            dev_acc_list.append(float(acc))
            dev_loss_summary = tf.Summary()
            cost = dev_loss_summary.value.add()
            cost.tag = "dev_loss"
            cost.simple_value = float(loss_value)
            dev_summary_writer.add_summary(dev_loss_summary)
        print 'Epoch: ', epoch, ' Accuracy: ', max(dev_acc_list)
        dev_acc_summary = tf.Summary()
        dev_acc = dev_acc_summary.value.add()
        dev_acc.tag = "dev_accuracy"
        dev_acc.simple_value = float(acc)
        dev_summary_writer.add_summary(dev_acc_summary)
        saving = saver.save(sess, os.path.join(args.log_dir, 'model%d.ckpt' % i))
        lr = lr * args.lr_decay

if __name__ == '__main__':
    main()