import tensorflow as tf
import numpy as np
import data_loader
import argparse
import os

import image_processor
import answer_generator

from gpu import define_gpu
define_gpu(1)

def build_fc7_batch(batch_head, batch_size, img_feature, img_id_map, qa_data, vocab_data, split):
    qa = qa_data[split]
    sentence = np.ndarray((batch_size, vocab_data['max_que_length']), dtype='int32')
    answer = np.zeros((batch_size, len(vocab_data['ans_vocab'])))
    img = np.ndarray((batch_size, 4096), dtype='float32')

    counter = 0
    while batch_head < len(qa) and counter < batch_size:
        if qa[batch_head]['image_id'] in img_id_map:
            sentence[counter, :] = qa[batch_head]['question'][:]
            answer[counter, qa[batch_head]['answer']] = 1.0
            img_index = img_id_map[qa[batch_head]['image_id']]
            img[counter, :] = img_feature[img_index][:]
            counter += 1
            batch_head += 1
        else:
            batch_head += 1
    if counter < batch_size:
        sentence = sentence[:counter, :]
        answer = answer[:counter, :]
        img = img[:counter, :]
    return sentence, answer, img, batch_head

def build_pool5_batch(batch_head, batch_size, img_feature, img_id_map, qa_data, vocab_data, split):
    qa = qa_data[split]
    sentence = np.ndarray((batch_size, vocab_data['max_que_length']), dtype='int32')
    answer = np.zeros((batch_size, len(vocab_data['ans_vocab'])))
    img = np.ndarray((batch_size, 7, 7, 512), dtype='float32')

    counter = 0
    while batch_head < len(qa) and counter < batch_size:
        if qa[batch_head]['image_id'] in img_id_map:
            sentence[counter, :] = qa[batch_head]['question'][:]
            answer[counter, qa[batch_head]['answer']] = 1.0
            img_index = img_id_map[qa[batch_head]['image_id']]
            img[counter, :, :, :] = img_feature[img_index][:]
            counter += 1
            batch_head += 1
        else:
            batch_head += 1
    if counter < batch_size:
        sentence = sentence[:counter, :]
        answer = answer[:counter, :]
        img = img[:counter, :, :, :]
    img = np.reshape(img, (img.shape[0], -1, img.shape[-1]))
    return sentence, answer, img, batch_head

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data Directory')
    parser.add_argument('--log_dir', type=str, default='log', help='Checkpoint File Directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')

    parser.add_argument('--batch_size', type=int, default=16, help='Image Training Batch Size')
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
    parser.add_argument('--att_hidden_dim', type=int, default=512, help='Hidden Dimension of Attention Hidden State')
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
        dev_img_feature, dev_img_id_list = image_processor.VGG_16_extract_pool5('val', args, file_code)
    else:
        print 'Loading Image Feature Data of VGG-16 Model Layer fc7'
        train_img_feature, train_img_id_list = image_processor.VGG_16_extract_fc7('train', args)
        dev_img_feature, dev_img_id_list = image_processor.VGG_16_extract_fc7('val', args)

    print 'Building Image ID Map and Answer Map'
    train_img_id_map = {}
    for i in xrange(len(train_img_id_list)):
        train_img_id_map[train_img_id_list[i]] = i
    dev_img_id_map = {}
    for i in xrange(len(dev_img_id_list)):
        dev_img_id_map[dev_img_id_list[i]] = i
    ans_map = {vocab_data['ans_vocab'][ans] : ans for ans in vocab_data['ans_vocab']}

    print 'Dataset Stats ##############'
    print 'Loaded Train Question Annotation: %d' % (len(qa_data['train']))
    print 'Loaded Dev Question Annotation: %d' % (len(qa_data['val']))
    print 'Loaded Train Image Feature: %d' % (train_img_feature.shape[0])
    print 'Loaded Dev Image Feature: %d' % (dev_img_feature.shape[0])
    print '############################'


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

    lr = args.learning_rate
    if args.use_attention:
        generator.build_coattention_model()
        loss, accuracy, predict, feed_img, feed_que, feed_label = generator.train_coattention_model()
    else:
        generator.build_base_model()
        loss, accuracy, predict, feed_img, feed_que, feed_label = generator.train_base_model()
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()

    tf.initialize_all_variables().run(session=sess)

    train_summary_writer = tf.train.SummaryWriter(os.path.join(args.log_dir, "summaries", "train"), sess.graph)
    dev_summary_writer = tf.train.SummaryWriter(os.path.join(args.log_dir, "summaries", "val"), sess.graph)

    print 'Training Start...'
    saver = tf.train.Saver()
    frozen_acc_flag = 0
    last_acc = 0
    for epoch in range(args.num_epoch):
        print 'Epoch %d #############' % (epoch + 1)
        train_batch_head = 0
        dev_batch_head = 0
        train_batch_num = 0
        dev_batch_num = 0
        dev_loss_list = []
        dev_acc_list = []
        # Training Using Training Data
        while train_batch_head < len(train_img_id_list):
            que_batch, ans_batch, img_batch = None, None, None
            if args.use_attention:
                que_batch, ans_batch, img_batch, train_batch_head = build_pool5_batch(train_batch_head, args.batch_size, \
                                                    train_img_feature, train_img_id_map, qa_data, vocab_data, 'train')
            else:
                que_batch, ans_batch, img_batch, train_batch_head = build_fc7_batch(train_batch_head, args.batch_size, \
                                                    train_img_feature, train_img_id_map, qa_data, vocab_data, 'train')
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
                cost.tag = "train_loss%d" % train_batch_num
                cost.simple_value = float(loss_value)
                train_summary_writer.add_summary(train_loss_summary, epoch + 1)
        # Validating Using Dev Data
        while dev_batch_head < len(dev_img_id_list):
            if args.use_attention:
                que_batch, ans_batch, img_batch, dev_batch_head = build_pool5_batch(dev_batch_head, args.batch_size, \
                                                dev_img_feature, dev_img_id_map, qa_data, vocab_data, 'val')
            else:
                que_batch, ans_batch, img_batch, dev_batch_head = build_fc7_batch(dev_batch_head, args.batch_size, \
                                                dev_img_feature, dev_img_id_map, qa_data, vocab_data, 'val')
            loss_value, acc, pred = sess.run([loss, accuracy, predict],
                                                    feed_dict={
                                                        feed_img: img_batch,
                                                        feed_que: que_batch,
                                                        feed_label: ans_batch
                                                    })
            dev_batch_num += 1
            dev_acc_list.append(float(acc))
            dev_loss_list.append(float(loss_value))

        epoch_loss = min(dev_loss_list)
        epoch_acc = max(dev_acc_list)
        # Record Epoch Training Loss Value
        dev_loss_summary = tf.Summary()
        cost = dev_loss_summary.value.add()
        cost.tag = "dev_loss"
        cost.simple_value = epoch_loss
        dev_summary_writer.add_summary(dev_loss_summary, epoch + 1)
        # Record Epoch Training Accuracy Value
        print 'Epoch: ', epoch + 1, ' Accuracy: ', epoch_acc
        dev_acc_summary = tf.Summary()
        dev_acc = dev_acc_summary.value.add()
        dev_acc.tag = "dev_accuracy"
        dev_acc.simple_value = epoch_acc
        dev_summary_writer.add_summary(dev_acc_summary, epoch + 1)
        # Saving Log
        saving = saver.save(sess, os.path.join(args.log_dir, 'model%d.ckpt' % i))
        lr = lr * args.lr_decay
        # Early Stopping
        if epoch_acc <= last_acc:
            frozen_acc_flag += 1
        else:
            frozen_acc_flag = 0
        last_acc = epoch_acc
        if frozen_acc_flag == 5:
            break
if __name__ == '__main__':
    main()