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
    train_que_json_path = os.path.join(data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
    train_ans_json_path = os.path.join(data_dir, 'mscoco_train2014_annotations.json')
    val_que_json_path = os.path.join(data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
    val_ans_json_path = os.path.join(data_dir, 'mscoco_val2014_annotations.json')
    text_file_path = os.path.join(data_dir, 'data_file.pkl')
    vocab_file_path = os.path.join(data_dir, 'vocab_file.pkl')

    # If Data Already Extracted
    if os.path.isfile(text_file_path):
        with open(text_file_path) as f:
            text_data = pickle.load(f)
        with open(vocab_file_path) as f:
            vocab_data = pickle.load(f)
        return text_data, vocab_data

    # Extract Data
    print "Loading Training Quesions: %s" % train_que_json_path
    with open(train_que_json_path) as f:
        train_que = json.load(f.read())

    print "Loading Training Answers: %s" % train_ans_json_path
    with open(train_ans_json_path) as f:
        train_ans = json.load(f.read())

    print "Loading Validation Questions: %s" % val_que_json_path
    with open(val_que_json_path) as f:
        val_que = json.load(f.read())

    print "Loading Validation Answers: %s" % val_ans_json_path
    with open(val_ans_json_path) as f:
        val_ans = json.load(f.read())

    questions = train_que['questions'] + val_que['questions']
    answers = train_ans['annotations'] + val_ans['annotations']

    ans_vocab = build_answer_vocab(answers, top_num)
    que_vocab, max_que_length = build_question_vocab(questions, answers, ans_vocab)
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
            que_words = re.findall(word_regex, question['questions'])

            padding_num = max_que_length - len(que_words)
            for j in range(len(que_words)):
                training_data[-1]['question'][padding_num + j] = que_vocab[que_words[j]]
    print 'Training Data Extracted: ', len(training_data)

    # Write Validation Question Data
    val_data = []
    for i, question in enumerate(val_que['questions']):
        ans = val_ans['annotations'][i]['multiple_choice_answer']
        if ans in ans_vocab:
            val_data.append({
                'image_id': val_ans['annotations'][i]['image_id'],
                'question': np.zeros(max_que_length),
                'answer': ans_vocab[ans]
            })
            que_words = re.findall(word_regex, question['questions'])

            padding_num = max_que_length - len(que_words)
            for j in range(len(que_words)):
                val_data[-1]['question'][padding_num + j] = que_vocab[que_words[j]]
    print 'Validation Data Extracted: ', len(val_data)

    print 'Saving Data'
    text_data = {
        'train': training_data,
        'val': val_data
    }
    with open(data_file_path, 'wb') as f:
        pickle.dump(text_data, f)

    vocab_data = {
        'ans_vocab': ans_vocab,
        'que_vocab': que_vocab,
        'max_que_length': max_que_length
    }
    with open(vocab_file_path, 'wb') as f:
        pickle.dump(vocab_data, f)

    return text_data, vocab_data

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
        if answer in answer_vocab:
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

def load_VGG_feature(data_dir, split):
    VGG_feature = None
    img_id_list = None
    with h5py.File(os.path.join(data_dir, (split + '_vgg16.h5')), 'r') as hf:
        VGG_feature = np.array(hf.get('fc7_feature'))
    with h5py.File(os.path.join(data_dir, (split + '_img_id.h5')), 'r') as hf:
        img_id_list = np.array(hf.get('img_id'))
    return VGG_feature, img_id_list