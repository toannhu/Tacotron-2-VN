from __future__ import print_function
from normalization.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import re as regex


def load_source_vocab():
    return load_vocab(hp.source_vocab)


def load_target_vocab():
    return load_vocab(hp.target_vocab)


def load_vocab(path):
    vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    src2idx, idx2src = load_source_vocab()
    tgt2idx, idx2tgt = load_target_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [src2idx.get(word, 1) for word in (source_sent + u" </s>").split()]  # 1: OOV, </S>: End of Text
        y = [tgt2idx.get(word, 1) for word in (target_sent + u" </s>").split()]
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_train_data():
    src_sents = [line for line in
                 codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line]
    tgt_sents = [line for line in
                 codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line]

    X, Y, Sources, Targets = create_data(src_sents, tgt_sents)
    return X, Y


def load_test_data():

    src_sents = [line.strip() for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    tgt_sents = [line.strip() for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]

    X, Y, Sources, Targets = create_data(src_sents, tgt_sents)
    return X, Sources, Targets  # (1064, 150)


def get_batch_data(data_type=tf.int32):
    # Load data
    X, Y = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, dtype=data_type)
    Y = tf.convert_to_tensor(Y, dtype=data_type)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=hp.batch_size,
                                  capacity=hp.batch_size * 64,
                                  min_after_dequeue=hp.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()
