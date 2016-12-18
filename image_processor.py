import tensorflow as tf
import numpy as np
import scipy
import os
import utils
import pickle
import h5py
import time

import data_loader
import utils

def VGG_16_extract_pool5(split, args, file_code):
    # If the feature is already recorded
    if os.path.exists(os.path.join(args.data_dir, split, '%s_vgg16_pool5_%d.h5' % (split, file_code))):
        print 'Image Feature Data Already Exists. Start Loading Feature Data from Layer pool5'
        return data_loader.load_VGG_feature_pool5(args.data_dir, split, file_code)
    print 'Image Feature Extraction Starts...'
    qa_data, vocab_data = data_loader.load_qa_data(args.data_dir, args.top_num)
    if split == 'train':
        qa_data = qa_data['train']
    else:
        qa_data = qa_data['val']

    img_id_dict = {}
    for qa in qa_data:
        img_id_dict[qa['image_id']] = 1

    img_id_list = [img_id for img_id in img_id_dict]
    num_img = len(img_id_list)
    print 'Total Image ID: ', num_img
    file_size = num_img // 2
    file_code = 0

    with tf.Session() as sess:
        print 'Processing Image Dataset of ', split
        img_batch = tf.placeholder("float", [None, 224, 224, 3])
        feature = np.ndarray((file_size, 7, 7, 512))
        idx = 0
        vgg = Vgg16()
        vgg.build(img_batch)
        filled_size = 0
        id_list = []

        while idx < num_img:
            start = time.clock()
            batch = np.ndarray((args.batch_size, 224, 224, 3))

            img_counter = 0
            for i in range(args.batch_size):
                if idx >= num_img or filled_size >= file_size:
                    break
                img_file_path = os.path.join(args.data_dir,'%s' % split,
                                                'COCO_%s2014_%.12d.jpg' % (split, img_id_list[idx]))
                if os.path.exists(img_file_path):
                    batch[i, :, :, :] = utils.load_image(img_file_path)
                    id_list.append(img_id_list[idx])
                    img_counter += 1
                    filled_size += 1
                idx += 1
            
            feed_dict = {img_batch : batch[0:img_counter, :, :, :]}
            # Feed in Images and Run through the Model
            feature_batch = sess.run(vgg.pool5, feed_dict=feed_dict)
            feature[(filled_size - img_counter):filled_size, :, :, :] = feature_batch[0:img_counter, :, : ,:]
            end = time.clock()
            print 'Time Spent: ', end - start
            print 'Image Processed: ', idx
            if filled_size >= file_size or idx >= num_img:
                print 'Saving VGG-16 Layer Features'
                hf5_feat = h5py.File(os.path.join(args.data_dir, split, '%s_vgg16_pool5_%d.h5' % \
                                                        (split, file_code)), 'w')
                hf5_feat.create_dataset('pool5_feature', data=feature)
                hf5_feat.close()

                print 'Saving Image ID List'
                hf5_img_id = h5py.File(os.path.join(args.data_dir, split, '%s_img_id_pool5_%d.h5' % \
                                                        (split, file_code)), 'w')
                hf5_img_id.create_dataset('img_id', data=id_list)
                hf5_img_id.close()

                print 'Finishing Saving Data Bucket %d' % file_code
                file_code += 1
                filled_size = 0
                id_list = []
                feature = np.ndarray((file_size, 7, 7, 512))
        print 'Image Information Encoding Done'   
        return data_loader.load_VGG_feature_pool5(args.data_dir, split, 0)

def VGG_16_extract_fc7(split, args):
    # If the feature is already recorded
    if os.path.exists(os.path.join(args.data_dir, split, split + '_vgg16_fc7.h5')):
        print 'Image Feature Data Already Exists. Start Loading Feature Data from Layer fc7'
        return data_loader.load_VGG_feature_fc7(args.data_dir, split)
    print 'Image Feature Extraction Starts...'
    qa_data, vocab_data = data_loader.load_qa_data(args.data_dir, args.top_num)
    if split == 'train':
        qa_data = qa_data['train']
    else:
        qa_data = qa_data['val']

    img_id_dict = {}
    for qa in qa_data:
        img_id_dict[qa['image_id']] = 1

    img_id_list = [img_id for img_id in img_id_dict]
    num_img = len(img_id_list)
    print 'Total Image ID: ', num_img

    with tf.Session() as sess:
        print 'Processing Image Dataset of ', split
        img_batch = tf.placeholder("float", [None, 224, 224, 3])
        feature = np.ndarray((num_img, 4096))
        idx = 0
        vgg = Vgg16()
        vgg.build(img_batch)

        while idx < num_img:
            start = time.clock()
            batch = np.ndarray((args.batch_size, 224, 224, 3))

            img_counter = 0
            for i in range(args.batch_size):
                if idx >= num_img:
                    break
                img_file_path = os.path.join(args.data_dir,'%s' % split,
                                                'COCO_%s2014_%.12d.jpg' % (split, img_id_list[idx]))
                if os.path.exists(img_file_path):
                    batch[i, :, :, :] = utils.load_image(img_file_path)
                    img_counter += 1
                idx += 1

            feed_dict = {img_batch : batch[0:img_counter, :, :, :]}
            # Feed in Images and Run through the Model
            feature_batch = sess.run(vgg.fc7, feed_dict=feed_dict)
            feature[(idx - img_counter):idx, :] = feature_batch[0:img_counter, :]
            end = time.clock()
            print 'Time Spent: ', end - start
            print 'Image Processed: ', idx
        
        print 'Saving VGG-16 Layer Features into ' + os.path.join(args.data_dir, split, split + '_vgg16_fc7.h5')
        hf5_feat = h5py.File(os.path.join(args.data_dir, split, split + '_vgg16_fc7.h5'), 'w')
        hf5_feat.create_dataset('fc7_feature', data=feature)
        hf5_feat.close()

        print 'Saving Image ID List into ' + os.path.join(args.data_dir, split, split + '_img_id_fc7.h5')
        hf5_img_id = h5py.File(os.path.join(args.data_dir, split, split + '_img_id_fc7.h5'), 'w')
        hf5_img_id.create_dataset('img_id', data=img_id_list)
        hf5_img_id.close()
        print 'Image Information Encoding Done'
        return data_loader.load_VGG_feature_fc7(args.data_dir, split)

# VGG-16 Model Quoted From https://github.com/machrisaa/tensorflow-vgg
VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg16:
    def __init__(self, vgg16_npy_path='data/vgg16.npy'):

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("Parameter Npy File Loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("Build VGG-16 Model Started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print("Build Model Finished In: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")