import os
import pickle
import multiprocessing
import sys
import random

import numpy as np
import sklearn_crfsuite
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from scipy import spatial

import math
from src import utils as utils


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

# Active learning using edit distance with cross validation.
def cv_edit_active_learn(args):

    # Read the input args.
    train_idx = args['train_idx']
    test_idx = args['test_idx']
    dataset = args['dataset']
    strings = args['strings']
    max_samples_batch = args['max_samples_batch']
    batch_size = args['batch_size']

    # Initialize arrays to store results.
    phrase_acc = np.zeros([max_samples_batch+1])
    out_acc = np.zeros([max_samples_batch+1])
    label_count = np.zeros([max_samples_batch+1])
    pseudo_acc = np.zeros([max_samples_batch + 1])

    # Define training set and testing set and corresponding original strings.
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_string = [strings[i] for i in train_idx]
    test_string = [strings[i] for i in test_idx]

    # Define an initial actual training set and the training pool (unlabeled data).
    initial_size = 10
    train_set_current = train_set[:initial_size]
    train_set_new = train_set[initial_size:]
    train_string_current = train_string[:initial_size]
    train_string_new = train_string[initial_size:]
    count = 0
    for i in range(initial_size):
        count += len(train_string[i])

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
    label_count[0] = count
    pseudo_acc[0] = 1  # There is no pseudo-label at the beginning.

    # Vectorized and clustered test set.
    total_string = test_string[:]
    total_string.extend(train_string_new)
    vec, _ = utils.string_vectorize(total_string)
    test_vec = vec[:len(test_string)].tolist()
    train_new_vec = vec[len(test_string):].tolist()

    # Pre-calculate similarity.
    # This will be efficient if the number of iterations is large.
    sim_matrix = np.zeros((len(train_new_vec), len(test_vec)))
    for i in range(len(train_new_vec)):
        for j in range(len(test_vec)):
            sim_matrix[i, j] = 1 - spatial.distance.cosine(train_new_vec[i], test_vec[j])

    len_test = len(test_set)

    initial_budget = 100
    if count >= initial_budget:
        print('Error: initial budget is less than initial number of labels.')
    else:
        label_threshold = initial_budget

    new_instance_idx = []   # record the indices of new added instances in the training set
    pseudo_label_idx = []   # record the positions of pseudo labels
    visited_idx = []    # record the indices of visited instances in the unlabeled set

    for num_training in range(max_samples_batch):

        label_list = crf.tagger_.labels()
        # # Want to look at the model confidence on the test set.
        # entropy_list = []
        # for i in test_set:
        #     len_ptname = len(i)
        #     crf.tagger_.set(sent2features(i))
        #     entropy_seq = []
        #     for j in range(len_ptname):
        #         marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
        #         entropy_seq.append(scipy.stats.entropy(marginal_prob))
        #     entropy_list.append(entropy_seq)
        #
        # # Sort the test set based on the average entropy.
        # entropy_sum = [sum(i)/len(i) for i in entropy_list]
        # sort_idx_temp = np.argsort(-np.array(entropy_sum), kind='mergesort').tolist()

        # Calculate the confidence on the test set using the current CRF.
        prob_list = []
        for i in range(len_test):
            # crf.tagger_.set(X_train_new[i])
            y_sequence = crf.tagger_.tag(X_test[i])
            # print(crf.tagger_.probability(y_sequence))
            # normalized sequence probability
            prob_norm = math.exp(math.log(crf.tagger_.probability(y_sequence)) / len(test_string[i]))
            prob_list.append(prob_norm)

        # Sort the test set based on confidence.
        sort_idx_temp = np.argsort(np.array(prob_list), kind='mergesort').tolist()

        # Calculate the average similarity between the unlabeled samples and the selected test samples.
        group_size = 1
        avr_sim = np.sum(sim_matrix[:, sort_idx_temp[:group_size]], axis=1) / group_size
        distance = avr_sim

        # We want to have information weighted by such distance.
        entropy_list = []
        len_train_new = len(train_set_new)
        for i in range(len_train_new):
            crf.tagger_.set(sent2features(train_set_new[i]))
            entropy_seq = []
            len_ptname = len(train_set_new[i])
            if i in visited_idx:
                revisit_idx_re = visited_idx.index(i)
                unlabeled_part = pseudo_label_idx[revisit_idx_re]
                for j in unlabeled_part:
                    marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
                    entropy_seq.append(scipy.stats.entropy(marginal_prob))
            else:
                for j in range(len_ptname):
                    marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
                    entropy_seq.append(scipy.stats.entropy(marginal_prob))
            entropy_list.append(entropy_seq)

        entropy_list_mean = []
        for i in range(len(entropy_list)):
            entropy_list_mean.append(sum(entropy_list[i])/len(entropy_list[i]))

        candidate_score = []
        for i in range(len_train_new):
            if distance[i] == 0:
                candidate_score.append(sys.float_info.max)
            else:
                candidate_score.append(entropy_list_mean[i]*distance[i])

        # Obtain the candidate index.
        sort_idx = np.argsort(candidate_score, kind='mergesort').tolist()
        sort_idx.reverse()
        sort_idx = sort_idx[0]

        # Check if this is revisiting.
        if sort_idx in visited_idx:
            revisit_flag = True
        else:
            revisit_flag = False

        if revisit_flag:
            revisit_idx_un = sort_idx  # the instance index in the unlabeled set
            revisit_idx_re = visited_idx.index(sort_idx)  # the instance index in the tracking record
            revisit_idx_tr = new_instance_idx[revisit_idx_re]  # the instance index in the training set
            # Update the pseudo label to manual label in the training set.
            y_train_current[revisit_idx_tr] = sent2labels(train_set_current[revisit_idx_tr])
            # Update the unlabeled set.
            del train_set_new[revisit_idx_un]
            del train_string_new[revisit_idx_un]
            del train_new_vec[revisit_idx_un]
            sim_matrix = np.delete(sim_matrix, revisit_idx_un, 0)
            # Update the tracking record.
            count += len(pseudo_label_idx[revisit_idx_re])
            del new_instance_idx[revisit_idx_re]
            del pseudo_label_idx[revisit_idx_re]
            del visited_idx[revisit_idx_re]
            for i in range(len(visited_idx)):
                if visited_idx[i] > revisit_idx_un:
                    visited_idx[i] = visited_idx[i] - 1
            label_count[num_training + 1] = count
        else:
            # Apply z-scores to pseudo-label.
            visited_idx.append(sort_idx)
            entropy_tmp = entropy_list[sort_idx]
            len_ptname = len(entropy_tmp)
            y_sequence = crf.tagger_.tag(sent2features(train_set_new[sort_idx]))
            mean_entropy_tmp = np.mean(entropy_tmp)
            std_entropy_tmp = np.std(entropy_tmp)
            z_score = [(entropy_tmp[i] - mean_entropy_tmp) / std_entropy_tmp for i in range(len_ptname)]
            label_index = []
            for i in range(len_ptname):
                if z_score[i] > 0.0:
                    count += 1
                    y_sequence[i] = sent2labels(train_set_new[sort_idx])[i]
                    label_index.append(i)
            pseudo_index = [i for i in range(len_ptname) if i not in label_index]
            pseudo_label_idx.append(pseudo_index)
            label_count[num_training + 1] = count

            # Update training set.
            new_instance_idx.append(len(train_string_current))
            train_set_current.append(train_set_new[sort_idx])
            train_string_current.append(train_string_new[sort_idx])
            # X_train_current.append(sent2features(train_set_new[sort_idx]))
            y_train_current.append(y_sequence)
            X_train_current = [sent2features(s) for s in train_set_current]
            # del train_set_new[sort_idx]
            # del train_string_new[sort_idx]
            # del train_new_vec[sort_idx]
            # sim_matrix = np.delete(sim_matrix, sort_idx, 0)

        # Update the pseudo labels using the current CRF.
        new_instance_count = 0
        for i in new_instance_idx:
            current_label_seq = y_train_current[i]
            new_pseudo_label_seq = crf.tagger_.tag(X_train_current[i])
            for j in pseudo_label_idx[new_instance_count]:
                current_label_seq[j] = new_pseudo_label_seq[j]
            y_train_current[i] = current_label_seq
            new_instance_count += 1

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
        phrase_acc[num_training+1] = phrase_correct / phrase_count
        out_acc[num_training+1] = out_correct / out_count

    return phrase_acc, out_acc, label_count, pseudo_acc

# This is the main function.
if __name__ == '__main__':

    # with open("../dataset/filtered_dataset.bin", "rb") as my_dataset:
    #     dataset = pickle.load(my_dataset)
    # with open("../dataset/filtered_string.bin", "rb") as my_string:
    #     strings = pickle.load(my_string)
    with open("../dataset/sod_label.bin", "rb") as my_dataset:
        dataset = pickle.load(my_dataset)
    with open("../dataset/sod_string.bin", "rb") as my_string:
        strings = pickle.load(my_string)

    # Randomly select test set and training pool in the way of cross validation.
    num_fold = 8
    kf = RepeatedKFold(n_splits=num_fold, n_repeats=1, random_state=666)

    # Define a loop for plotting figures.
    max_samples_batch = 200
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
    results = pool.map(cv_edit_active_learn, args)
    # print(len(results))
    # print(len(results[0]))
    phrase_acc = [results[i][0] for i in range(num_fold)]
    out_acc = [results[i][1] for i in range(num_fold)]
    # print(len(phrase_acc))
    # print(len(phrase_acc[0]))
    label_count = [results[i][2] for i in range(num_fold)]
    pseudo_acc = [results[i][3] for i in range(num_fold)]

    with open("../baseline/ibm_phrase_acc_confidence_edit.bin", "rb") as phrase_confidence_file_temp:
        phrase_acc_confidence_edit = pickle.load(phrase_confidence_file_temp)
    with open("../baseline/ibm_out_acc_confidence_edit.bin", "rb") as out_confidence_file_temp:
        out_acc_confidence_edit = pickle.load(out_confidence_file_temp)
    with open("../baseline/ibm_confidence_edit_num.bin", "rb") as label_count_file_temp:
        label_count_confidence_edit = pickle.load(label_count_file_temp)

    phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0) / num_fold
    phrase_acc_max_confidence_edit = np.max(phrase_acc_confidence_edit, axis=0)
    phrase_acc_min_confidence_edit = np.min(phrase_acc_confidence_edit, axis=0)
    label_count_av_confidence_edit = np.sum(label_count_confidence_edit, axis=0) / num_fold
    out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0) / num_fold

    phrase_acc_av = np.sum(phrase_acc, axis=0)/num_fold
    phrase_acc_max = np.max(phrase_acc, axis=0)
    phrase_acc_min = np.min(phrase_acc, axis=0)

    out_acc_av = np.sum(out_acc, axis=0)/num_fold

    label_count_av = np.sum(label_count, axis=0)/num_fold
    label_count_max = np.max(label_count, axis=0)
    label_count_min = np.min(label_count, axis=0)

    pseudo_acc_av = np.sum(pseudo_acc, axis=0) / num_fold
    pseudo_acc_max = np.max(pseudo_acc, axis=0)
    pseudo_acc_min = np.min(pseudo_acc, axis=0)

    plt.plot(label_count_av, phrase_acc_av, 'r',
             label_count_av_confidence_edit, phrase_acc_av_confidence_edit, 'b',
             label_count_av, phrase_acc_max, '--r',
             label_count_av, phrase_acc_min, '--r',
             label_count_av_confidence_edit, phrase_acc_max_confidence_edit, '--b',
             label_count_av_confidence_edit, phrase_acc_min_confidence_edit, '--b')
    plt.xlabel('number of manual labels')
    plt.ylabel('testing accuracy')
    plt.legend(['partial label', 'full label'])
    plt.show()

    plt.plot(np.arange(1, len(label_count_av) + 1, 1), label_count_av, 'r',
             np.arange(1, len(label_count_av) + 1, 1), label_count_max, '--r',
             np.arange(1, len(label_count_av) + 1, 1), label_count_min, '--r')
    plt.xlabel('number of iterations')
    plt.ylabel('average manual labels')
    plt.show()

    plt.plot(np.arange(1, len(pseudo_acc_av) + 1, 1), pseudo_acc_av, 'r',
             np.arange(1, len(pseudo_acc_av) + 1, 1), pseudo_acc_max, '--r',
             np.arange(1, len(pseudo_acc_av) + 1, 1), pseudo_acc_min, '--r')
    plt.xlabel('number of iterations')
    plt.ylabel('pseudo-label accuracy')
    plt.show()

    # Save data for future plotting.
    with open("ibm_phrase_acc_partial_entropy_sum_edit_z.bin", "wb") as phrase_confidence_file:
        pickle.dump(phrase_acc, phrase_confidence_file)
    with open("ibm_out_acc_partial_entropy_sum_edit_z.bin", "wb") as out_confidence_file:
        pickle.dump(out_acc, out_confidence_file)
    with open("ibm_partial_entropy_sum_edit_num_z.bin", "wb") as label_count_file:
        pickle.dump(label_count, label_count_file)
    with open("ibm_partial_entropy_sum_edit_pseudo_acc_z.bin", "wb") as pseudo_acc_file:
        pickle.dump(pseudo_acc, pseudo_acc_file)
