import json
import os
import re
import numpy as np
import pickle
import h5py

word_regex = re.compile(r'\w+')

def load_qa_data(data_dir, top_num):
    questions = None
    answers = None
    assert os.path.isdir(os.path.join(data_dir, 'train'))
    assert os.path.isdir(os.path.join(data_dir, 'val'))
    train_que_json_path = os.path.join(data_dir,'train', 'MultipleChoice_mscoco_train2014_questions.json')
    train_ans_json_path = os.path.join(data_dir,'train', 'mscoco_train2014_annotations.json')
    dev_que_json_path = os.path.join(data_dir,'val', 'MultipleChoice_mscoco_val2014_questions.json')
    dev_ans_json_path = os.path.join(data_dir,'val', 'mscoco_val2014_annotations.json')
    data_file_path = os.path.join(data_dir, 'data_file.pkl')
    vocab_file_path = os.path.join(data_dir, 'vocab_file.pkl')

    # If Data Already Extracted
    if os.path.isfile(data_file_path):
        with open(data_file_path) as f:
            qa_data = pickle.load(f)
        print 'Train Question Answer Data Already Exists and Loaded'
        with open(vocab_file_path) as f:
            vocab_data = pickle.load(f)
        print 'Dev Question Answer Data Already Exists and Loaded'
        return qa_data, vocab_data

    # Extract Data
    print "Loading Training Quesions: %s" % train_que_json_path
    with open(train_que_json_path) as f:
        train_que = json.load(f)

    print "Loading Training Answers: %s" % train_ans_json_path
    with open(train_ans_json_path) as f:
        train_ans = json.load(f)

    print "Loading Dev Questions: %s" % dev_que_json_path
    with open(dev_que_json_path) as f:
        dev_que = json.load(f)

    print "Loading Dev Answers: %s" % dev_ans_json_path
    with open(dev_ans_json_path) as f:
        dev_ans = json.load(f)

    questions = train_que['questions'] + dev_que['questions']
    answers = train_ans['annotations'] + dev_ans['annotations']

    ans_vocab = build_ans_vocab(answers, top_num)
    que_vocab, max_que_length = build_que_vocab(questions, answers, ans_vocab)
    print 'Max Question Length: ', max_que_length

    # Write Training Question Data
    training_data = []
    for i, question in enumerate(train_que['questions']):
        ans = train_ans['annotations'][i]['multiple_choice_answer']
        if ans in ans_vocab:
            training_data.append({
                'image_id': train_ans['annotations'][i]['image_id'],
                'question': np.zeros(max_que_length),
                'answer': ans_vocab[ans]
            })
            que_words = re.findall(word_regex, question['question'])

            padding_num = max_que_length - len(que_words)
            for j in range(len(que_words)):
                training_data[-1]['question'][padding_num + j] = que_vocab[que_words[j]]
    print 'Training Data Extracted: ', len(training_data)

    # Write Validation Question Data
    dev_data = []
    for i, question in enumerate(dev_que['questions']):
        ans = dev_ans['annotations'][i]['multiple_choice_answer']
        if ans in ans_vocab:
            dev_data.append({
                'image_id': dev_ans['annotations'][i]['image_id'],
                'question': np.zeros(max_que_length),
                'answer': ans_vocab[ans]
            })
            que_words = re.findall(word_regex, question['question'])

            padding_num = max_que_length - len(que_words)
            for j in range(len(que_words)):
                dev_data[-1]['question'][padding_num + j] = que_vocab[que_words[j]]
    print 'Dev Data Extracted: ', len(dev_data)

    print 'Saving Data'
    qa_data = {
        'train': training_data,
        'val': dev_data
    }
    with open(data_file_path, 'wb') as f:
        pickle.dump(qa_data, f)

    vocab_data = {
        'ans_vocab': ans_vocab,
        'que_vocab': que_vocab,
        'max_que_length': max_que_length
    }
    with open(vocab_file_path, 'wb') as f:
        pickle.dump(vocab_data, f)

    return qa_data, vocab_data

def build_ans_vocab(answers, top_num):
    # Build Answer Frequency Dictionary
    ans_freq = {}
    for annotation in answers:
        answer = annotation['multiple_choice_answer']
        if answer in ans_freq:
            ans_freq[answer] += 1
        else:
            ans_freq[answer] = 1

    # Sort Answer By Frequency to Get Top Answers
    ans_freq_tuple = [(-freq, ans) for ans, freq in ans_freq.iteritems()]
    del ans_freq
    ans_freq_tuple.sort()
    ans_freq_tuple = ans_freq_tuple[0:top_num - 1]

    # ans_vocab is a Dictionary with Structure (Answer Content -> Frequency Rank)
    ans_vocab = {}
    for freq_rank, ans_freq in enumerate(ans_freq_tuple):
        ans = ans_freq[1]
        ans_vocab[ans] = freq_rank
    ans_vocab['UNK'] = top_num - 1
    return ans_vocab

def build_que_vocab(questions, answers, ans_vocab):
    que_freq = {}
    max_que_length = 0
    # Split question in words if the answer can be found
    for i, question in enumerate(questions):
        answer = answers[i]['multiple_choice_answer']
        que_length_counter = 0
        if answer in ans_vocab:
            que_words = re.findall(word_regex, question['question'])
            for word in que_words:
                if word in que_freq:
                    que_freq[word] += 1
                else:
                    que_freq[word] = 1
                que_length_counter += 1
        if que_length_counter > max_que_length:
            max_que_length = que_length_counter

    qword_freq_threshold = 0
    qword_tuple = [(-frequency, qword) for qword, frequency in que_freq.iteritems()]
    del que_freq

    que_vocab = {}
    for i, qword_freq in enumerate(qword_tuple):
        frequency = -qword_freq[0]
        qword = qword_freq[1]
        if frequency > qword_freq_threshold:
            que_vocab[qword] = i + 1
        else:
            break
    que_vocab['UNK'] = len(que_vocab) + 1

    return que_vocab, max_que_length

def load_VGG_feature_fc7(data_dir, split):
    VGG_feature = None
    img_id_list = None
    with h5py.File(os.path.join(data_dir, split, (split + '_vgg16_fc7.h5')), 'r') as hf:
        VGG_feature = np.array(hf.get('fc7_feature'))
    with h5py.File(os.path.join(data_dir, split, (split + '_img_id_fc7.h5')), 'r') as hf:
        img_id_list = np.array(hf.get('img_id'))
    return VGG_feature, img_id_list

def load_VGG_feature_pool5(data_dir, split, file_code):
    VGG_feature = None
    img_id_list = None
    with h5py.File(os.path.join(data_dir, split, '%s_vgg16_pool5_%d.h5' % (split, file_code)), 'r') as hf:
        VGG_feature = np.array(hf.get('pool5_feature'))
    with h5py.File(os.path.join(data_dir, split, '%s_img_id_pool5_%d.h5' % (split, file_code)), 'r') as hf:
        img_id_list = np.array(hf.get('img_id'))
    return VGG_feature, img_id_list