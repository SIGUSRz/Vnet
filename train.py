import tensorflow as tf
import numpy as np
import data_loader
import argparse

import image_processor
import answer_generator

def right_align(seq, length):
    mask = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        mask[i][N - length[i]:N - 1] = seq[i][0:length[i] - 1]
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data Directory')
    parser.add_argument('--log_dir', type=str, default='log', help='Checkpoint File Directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')

    parser.add_argument('--batch_size', type=int, default=64, help='Image Training Batch Size')
    parser.add_argument('--num_output', type=int, default=1000, help='Number of Output')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout Rate')
    parser.add_argument('--init_bound', type=float, default=0.8, help='Parameter Initialization Distribution Bound')

    parser.add_argument('--hidden_dim', type=int, default=1024, help='RNN Hidden State Dimension')
    parser.add_argument('--rnn_size', type=int, default=512, help='Size of RNN Cell')
    parser.add_argument('--rnn_layer', type=int, default=2, help='Number of RNN Layers')
    parser.add_argument('--que_embed_size', type=int, default=200, help='Question Embedding Dimension')

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning Rate Decay Factor')
    parser.add_argument('--num_iteration', type=int, default=15000, help='Number of Training Iterations')
    parser.add_argument('--num_epoch', type=int, default=300, help='Number of Training Epochs')
    parser.add_argument('--grad_norm', type=int, default=5, 'Maximum Norm of the Gradient')
    args = parser.parse_args()


    print 'Reading Question Answer Data'
    qa_data, vocab_data = data_loader.load_qa_data(args.data_dir, args.top_num)
    train_img_feature, train_img_id_list = image_processor.VGG_16_extract('train', args)

    print 'Building Answer Generator Model'
    generator = answer_generator.Answer_Generator({
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'img_dim': train_img_feature.shape[1],
        'rnn_size': args.rnn_size,
        'rnn_layer': args.rnn_layer,
        'que_embed_size': args.que_embed_size,
        'que_vocab_size': len(vocab_data['que_vocab']),
        'max_que_length': vocab_data['max_que_length'],
        'num_output': args.num_output,
        'dropout_rate': args.dropout_rate,
        'data_dir': args.data_dir,
        'top_num': args.top_num,
        'init_bound': args.init_bound
        })

    loss, feed_img, feed_que, feed_label = generator.train_model()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=10)

    lr = tf.Variable(args.learning_rate, trainable=False)
    tvar = tf.trainable_variables()
    grads, _ = tf.clip_by_norm(tf.gradients(loss, tvar), args.grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)

    train_op = optimizer.apply_gradients(zip(grads, tvar), global_step=tf.contrib.framework.get_or_create_global_step())

    tf.initialize_all_variables().run()

    print 'Training Start...'
    train_data = qa_data['train']
    num_train = len(train_data)
    for itr in range(args.num_iteration):
        tStart = time.time()
        index = np.random.random_integers(0, num_train - 1, args.batch_size)

        que_batch = train_data[index, :]['question']
        ans_batch = train_data[index, :]['answer']
        id_list = train_img_id_list[index]
        img_batch = train_img_feature[id_list, :]

        _, loss = sess.run(
            [train_op, loss],
            feed_dict={
                feed_img: img_batch,
                feed_que: que_batch,
                feed_label: ans_batch
            }
        )

        new_lr = lr * args.lr_decay
        lr.assign(new_lr).eval()

        tStop = time.time()
        if np.mod(itr, 100) == 0:
            print 'Iteration: %d; Loss: %.5f; Learning Rate: %.5f' % (itr, loss, lr.eval())
            print 'Time Cost: %.3f s' % round(tStop - tStart, 2)
        if np.mod(itr, 15000) == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), global_step=itr)

        saver.save(sess, os.path.join(args.log_dir, 'model'), global_step=args.num_epoch)



if __name__ == '__main__':
    main()