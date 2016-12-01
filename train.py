import tensorflow as tf
import numpy as np
import data_loader
import argparse

import img_processor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--top_num', type=int, default=1000, 'Top Number Answer')
    args = parser.parse_args()


    print 'Reading Question Answer Data'
    data, vocab_data = data_loader.load_question_answer(args.data_dir)
    train_img_feature, train_img_id_list = img_processor.VGG_16_extract(args.data_dir, 'train')
    print "Train Image Features: ", train_img_feature.shape
    print "Train Image Id List: ", train_img_id_list.shape

    train_img_id_map = {}
    for i in xrange(len(train_img_id_list)):
        train_img_id_map[train_img_id_list[i]] = i