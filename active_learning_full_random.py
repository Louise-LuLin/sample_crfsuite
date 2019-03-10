import os
import random
import multiprocessing
import pickle
import sys

import numpy as np
import sklearn_crfsuite
import matplotlib.pyplot as plt
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
    word = sent[i][0]
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
        word1 = sent[i-1][0]
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
        word1 = sent[i+1][0]
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
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]

# Active learning using uniform sampling with cross validation.
def cv_edit_active_learn(args):

    # Read the input args.
    train_idx = args['train_idx']
    test_idx = args['test_idx']
    dataset = args['dataset']
    strings = args['strings']
    max_samples_batch = args['max_samples_batch']
    batch_size = args['batch_size']

    phrase_acc = np.zeros([max_samples_batch+1])
    out_acc = np.zeros([max_samples_batch+1])
    label_count = np.zeros([max_samples_batch+1])

    # Define training set and testing set.
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_string = [strings[i] for i in train_idx]
    test_string = [strings[i] for i in test_idx]

    # Define an initial actual training set from the training pool.
    initial_size = 10
    train_set_current = train_set[:initial_size]
    train_set_new = train_set[initial_size:]
    train_string_current = train_string[:initial_size]
    train_string_new = train_string[initial_size:]
    for i in range(initial_size):
        label_count[0] += len(train_string[i])

    # Obtain testing features and labels.
    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]

    # Train a CRF using the current training set.
    X_train_current = [sent2features(s) for s in train_set_current]
    y_train_current = [sent2labels(s) for s in train_set_current]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train_current, y_train_current)

    # Use the estimator.
    y_pred = crf.predict(X_test)
    phrase_count, phrase_correct, out_count, out_correct = utils.phrase_acc(y_test, y_pred)
    phrase_acc[0] = phrase_correct / phrase_count
    out_acc[0] = out_correct / out_count

    for num_training in range(max_samples_batch):

        # Uniformly take samples from the training pool.
        len_train_new = len(train_set_new)
        sample_idx = []
        for i in range(batch_size):
            rand_tmp = random.randint(0,len_train_new-1)
            while rand_tmp in sample_idx:
                rand_tmp = random.randint(0,len_train_new-1)
            sample_idx.append(rand_tmp)

        label_count[num_training + 1] = label_count[num_training] + len(
            train_set_new[sample_idx[0]])  # assume batch_size = 1
        # update training strings
        string_to_remove = [train_string_new[i] for i in sample_idx[:batch_size]]
        for i in string_to_remove:
            train_string_current.append(i)
            train_string_new.remove(i)
        # update training set
        sample_to_remove = [train_set_new[i] for i in sample_idx[:batch_size]]
        for i in sample_to_remove:
            train_set_current.append(i)
            train_set_new.remove(i)

        # Obtain current training features.
        X_train_current = [sent2features(s) for s in train_set_current]
        y_train_current = [sent2labels(s) for s in train_set_current]

        # # define fixed parameters and parameters to search
        # crf = sklearn_crfsuite.CRF(
        #     algorithm='lbfgs',
        #     max_iterations=100,
        #     all_possible_transitions=True
        # )
        # params_space = {
        #     'c1': scipy.stats.expon(scale=0.5),
        #     'c2': scipy.stats.expon(scale=0.05),
        # }
        #
        # # search
        # rs = RandomizedSearchCV(crf, params_space,
        #                         cv=2,
        #                         verbose=1,
        #                         n_jobs=-1,
        #                         n_iter=5)
        # rs.fit(X_train_current, y_train_current)
        #
        # print('best params:', rs.best_params_)
        # print('best CV score:', rs.best_score_)
        # crf = rs.best_estimator_

        # Train the CRF.
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train_current, y_train_current)

        # Use the estimator.
        y_pred = crf.predict(X_test)
        phrase_count, phrase_correct, out_count, out_correct = utils.phrase_acc(y_test, y_pred)
        # print(phrase_count, phrase_correct, out_count, out_correct)
        phrase_acc[num_training+1] = phrase_correct / phrase_count
        out_acc[num_training+1] = out_correct / out_count

    return phrase_acc, out_acc, label_count

# This is the main function.
if __name__ == '__main__':

    with open("../dataset/sod_label.bin", "rb") as my_dataset:
        dataset = pickle.load(my_dataset)
    with open("../dataset/sod_string.bin", "rb") as my_string:
        strings = pickle.load(my_string)
    # with open("../dataset/sdh_dataset.bin", "rb") as my_dataset:
    #     dataset = pickle.load(my_dataset)
    # with open("../dataset/sdh_string.bin", "rb") as my_string:
    #     strings = pickle.load(my_string)

    # Randomly select test set and training pool in the way of cross validation.
    num_fold = 8
    kf = RepeatedKFold(n_splits=num_fold, n_repeats=1, random_state=666)

    # Define a loop for plotting figures.
    max_samples_batch = 100
    batch_size = 1

    # Shuffle the dataset.
    combined = list(zip(dataset, strings))
    random.seed(666)
    random.shuffle(combined)
    dataset[:], strings[:] = zip(*combined)

    pool = multiprocessing.Pool(os.cpu_count())
    args = []
    # print(os.cpu_count()) # It counts for logical processors instead of physical cores.
    for train_idx, test_idx in kf.split(dataset):
        tmp_args = {
            'train_idx': train_idx,
            'test_idx': test_idx,
            'dataset': dataset,
            'strings': strings,
            'max_samples_batch': max_samples_batch,
            'batch_size': batch_size,
        }
        args.append(tmp_args)

    # Use the same parameters for more than once of uniformly random sampling.
    results = pool.map(cv_edit_active_learn, args)
    phrase_acc = np.array([results[i][0] for i in range(num_fold)])
    out_acc = np.array([results[i][1] for i in range(num_fold)])
    label_count = np.array([results[i][2] for i in range(num_fold)])

    # # Run multiple sampling for each fold.
    # number_iter = 4
    # for i in range(number_iter):
    #     results = pool.map(cv_edit_active_learn, args)
    #     phrase_acc = phrase_acc + np.array([results[i][0] for i in range(num_fold)])
    #     out_acc = out_acc + np.array([results[i][1] for i in range(num_fold)])
    # phrase_acc = phrase_acc/(number_iter+1)
    # out_acc = out_acc/(number_iter+1)

    phrase_acc_av = np.sum(phrase_acc, axis=0)/num_fold
    out_acc_av = np.sum(out_acc, axis=0)/num_fold
    label_count_av = np.sum(label_count, axis=0)/num_fold

    plt.plot(label_count_av, phrase_acc_av, 'r',
             label_count_av, out_acc_av, 'b')
    plt.xlabel('number of training samples')
    plt.ylabel('testing accuracy')
    plt.legend(['phrase accuracy', 'out-of-phrase accuracy'])
    plt.show()

    # Save data for future plotting.
    with open("sod_phrase_acc_uniform.bin", "wb") as phrase_uniform_file:
        pickle.dump(phrase_acc, phrase_uniform_file)
    with open("sod_out_acc_uniform.bin", "wb") as out_uniform_file:
        pickle.dump(out_acc, out_uniform_file)
    with open("sod_label_count_uniform.bin", "wb") as label_count_file:
        pickle.dump(label_count, label_count_file)
