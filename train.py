import tensorflow as tf
import numpy as np
import data_loader
import argparse

import img_processor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')
    parser.add_argument('--batch_size', type=int, default=64, help='Image Training Batch Size')
    args = parser.parse_args()


    print 'Reading Question Answer Data'
    qa_data, vocab_data = data_loader.load_question_answer(args.data_dir)
    train_img_feature, train_img_id_list = img_processor.VGG_16_extract('train', args)
    print "Train Image Features: ", train_img_feature.shape
    print "Train Image Id List: ", train_img_id_list.shape