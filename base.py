import os
import random
import multiprocessing
import pickle
import sys
import eli5

import numpy as np
import sklearn_crfsuite
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import RepeatedKFold

import math

sys.path.append("..")

import src.utils.utils as utils


# Define feature dictionary.
def word2features(sent, i):
    # obtain some overall information of the point name string
    num_part = 4
    len_string = len(sent)
    mod = len_string % num_part
    part_size = int(math.floor(len_string/num_part))
    # determine which part the current character belongs to
    # larger part will be at the beginning if the whole sequence can't be divided evenly
    size_list = []
    mod_count = 0
    for j in range(num_part):
        if mod_count < mod:
            size_list.append(part_size+1)
            mod_count += 1
        else:
            size_list.append(part_size)
    # for current character
    part_cumulative = [0]*num_part
    for j in range(num_part):
        if j > 0:
            part_cumulative[j] = part_cumulative[j-1] + size_list[j]
        else:
            part_cumulative[j] = size_list[j] - 1   # indices start from 0
    part_indicator = [0]*num_part
    for j in range(num_part):
        if part_cumulative[j] >= i:
            part_indicator[j] = 1
            break
    word = sent[i]
    if word.isdigit():
        itself = 'NUM'
    else:
        itself = word
    features = {
        'word': itself,
        'part0': part_indicator[0] == 1,
        'part1': part_indicator[1] == 1,
        'part2': part_indicator[2] == 1,
        'part3': part_indicator[3] == 1,
    }
    # for previous character
    if i > 0:
        part_indicator = [0] * num_part
        for j in range(num_part):
            if part_cumulative[j] >= i-1:
                part_indicator[j] = 1
                break
        word1 = sent[i-1]
        if word1.isdigit():
            itself1 = 'NUM'
        else:
            itself1 = word1
        features.update({
            '-1:word': itself1,
            '-1:part0': part_indicator[0] == 1,
            '-1:part1': part_indicator[1] == 1,
            '-1:part2': part_indicator[2] == 1,
            '-1:part3': part_indicator[3] == 1,
        })
    else:
        features['BOS'] = True
    # for next character
    if i < len(sent)-1:
        part_indicator = [0] * num_part
        for j in range(num_part):
            if part_cumulative[j] >= i + 1:
                part_indicator[j] = 1
                break
        word1 = sent[i+1]
        if word1.isdigit():
            itself1 = 'NUM'
        else:
            itself1 = word1
        features.update({
            '+1:word': itself1,
            '+1:part0': part_indicator[0] == 1,
            '+1:part1': part_indicator[1] == 1,
            '+1:part2': part_indicator[2] == 1,
            '+1:part3': part_indicator[3] == 1,
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for label in sent]

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

# Active learning using uniform sampling with cross validation.
def base_crf(args):

    # Read the input args.
    train_set = args['train_label']
    train_string = args['train_string']
    test_set = args['test_label']
    test_string = args['test_string']

    # Obtain testing features and labels.
    X_test = [sent2features(s) for s in test_string]
    y_test = [sent2labels(s) for s in test_set]

    # Train a CRF using the current training set.
    X_train = [sent2features(s) for s in train_string]
    y_train = [sent2labels(s) for s in train_set]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        min_freq=1,
        c1=0.1,
        c2=0.1,
        max_iterations=50,
        all_possible_transitions=False
    )
    crf.fit(X_train, y_train)
    # expl = eli5.show_weights(crf, top=30)
    # print(eli5.format_as_text(expl))
    print("Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])

    # Use the estimator.
    y_pred = crf.predict(X_test)
    phrase_count, phrase_correct, out_count, out_correct, all_count, all_correct = utils.phrase_acc(y_test, y_pred)
    phrase_acc = phrase_correct / phrase_count
    out_acc = out_correct / out_count
    all_acc = (phrase_correct + out_correct) / (phrase_count + out_count)
    all_acc2 = all_correct / all_count
    return phrase_acc, out_acc, all_acc, all_acc2

# This is the main function.
if __name__ == '__main__':

    with open("../dataset/train_label_true.bin", "rb") as my_dataset:
        train_label = pickle.load(my_dataset)
    with open("../dataset/train_string.bin", "rb") as my_string:
        train_string = pickle.load(my_string)
    with open("../dataset/test_label.bin", "rb") as my_dataset:
        test_label = pickle.load(my_dataset)
    with open("../dataset/test_string.bin", "rb") as my_string:
        test_string = pickle.load(my_string)

    max_samples_batch = 100
    batch_size = 1

    # print(os.cpu_count()) # It counts for logical processors instead of physical cores.
    tmp_args = {
        'train_label': train_label[:80],
        'train_string': train_string[:80],
        'test_label': test_label,
        'test_string': test_string,
    }

    # Use the same parameters for more than once of uniformly random sampling.
    phrase_acc, out_acc, all_acc, all_acc2 = base_crf(tmp_args)
    print (phrase_acc, out_acc, all_acc, all_acc2)





