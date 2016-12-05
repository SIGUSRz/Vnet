import tensorflow as tf
import numpy as np
import time
import os
import argparse

import data_loader
import img_model
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')
    parser.add_argument('--batch_size', type=int, default=1, help='Image Training Batch Size')
    args = parser.parse_args()

    # train_img_feature, train_img_id_list = img_processor.VGG_16_extract('train', args)
    # print "Train Image Features: ", train_img_feature.shape
    # print "Train Image Id List: ", train_img_id_list.shape

    num_img = 2
    with tf.Session() as sess:
        img_batch = tf.placeholder("float", [None, 224, 224, 3])
        fc7_feature = np.ndarray((num_img, 4096))
        idx = 0
        vgg = img_model.Vgg16()
        vgg.build(img_batch)

        while idx < num_img:
            start = time.clock()
            batch = np.ndarray((args.batch_size, 224, 224, 3))

            img_counter = 0
            for i in range(args.batch_size):
                if idx >= num_img:
                    print idx, num_img
                    break
                img_file_path = os.path.join(args.data_dir, 'test%d.png' % (idx + 1))
                batch[i, :, :, :] = utils.load_image(img_file_path)
                idx += 1
                img_counter += 1
            feed_dict = {img_batch : batch[0:img_counter, :, :, :]}

            # with tf.name_scope("img_model"):
            #     vgg.build(img_batch)
            # Feed in Images and Run through the Model
            fc7_feature_batch = sess.run(vgg.fc7, feed_dict=feed_dict)
            fc7_feature[(idx - img_counter):idx, :] = fc7_feature_batch[0:img_counter, :]
            end = time.clock()
            print 'Time Spent: ', end - start
            print 'Image Processed To Index: ', idx
        print fc7_feature

if __name__ == '__main__':
    main()