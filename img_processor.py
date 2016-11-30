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
import img_model

def VGG_16_extract(split, data_dir, batch_size):
    images = tf.placeholder("float", [None, 224, 224, 3])
    if split == 'train':
        data = data_loader.load_qa_data['train']
    else:
        data = data_loader.load_qa_data['val']

    img_id_dict = {}
    for qa in data:
        img_id_dict[qa['img_id']] = 1

    img_id_list = [img_id for img_id in img_id_dict]
    num_img = len(img_id_list)
    print 'Total Images: ', num_img

    with tf.Session() as sess:
        fc7_feature = np.ndarray((num_img, 4096))
        idx = 0

        while idx < num_img:
            start = time.clock()
            batch = np.ndarray((batch_size, 224, 224, 3))

            img_counter = 0
            for i in range(batch_size):
                if idx >= num_img:
                    break
                img_file_path = os.path.join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg' % (split, split, img_id_list[idx]))
                batch[i, :, :, :] = utils.load_image(img_file_path)
                idx += 1
                img_counter += 1

            feed_dict = {images : batch[0:img_counter, :, :, :]}
            vgg = img_model.Vgg16()
            with tf.name_scope("img_model"):
                vgg.build(img_batch)
            fc7_feature_batch = sess.run(vgg.fc7, feed_dict=feed_dict)
            fc7_feature[(idx - img_counter):idx, :] = fc7_feature_batch[0:img_counter, :]
            end = time.clock()
            print 'Time Spent: ', end - start
            print 'Image Processed: ', idx
        
        print 'Saving VGG-16 FC7 Layer Features'
        hf5_fc7 = h5py.File(os.path.join(data_dir, split + '_vgg16.h5'), 'w')
        hf5_fc7.create_dataset('fc7_feature', data=fc7_feature)
        hf5_fc7.close()

        print 'Saving Image ID List'
        hf5_img_id = h5py.File(os.path.join(data_dir, split + '_img_id.h5'), 'w')
        hf5_img_id.create_dataset('img_id', data=img_id_list)
        hf5_img_id.close()
        print 'Image Information Encoding Done'