import numpy as np
import tensorflow as tf
import nltk
import hashlib
import collections

'''

def cache_function(function, *args, **kwargs):
    kwargs_keys = sorted(kwargs.keys())
    h = hashlib.sha224()\
    h.update(function.__qualname__)\
    h.update(str(str(arg) for arg in args).encode())
    h.update(str(str(key) for key in kwargs_keys).encode())
    h.update(str(str(key) for key in kwargs_keys).encode())
    h.update(str(str(kwargs[key]).encode() for key in kwargs_keys))
'''


def count_words(words):
    counts = collections.defaultdict(lambda: 0)
    for word in words:
        counts[word] += 1
    return counts


def filter_vocab(word_counts, min_count):
    return {word for word in word_counts if word_counts[word] >= min_count}


def prepare_words_ids_lookup(vocab):
    ''' założenie jest takie, że pierwszy, zerowy id to nieznane słowo'''
    return {word: i+1 for i, word in enumerate(vocab)}


def prepare_word_lookup(vocab):
    ''' założenie jest takie, że pierwszy, zerowy id to nieznane słowo'''
    ids_dict = prepare_words_ids_lookup(vocab)
    params = tf.constant(np.identity(len(ids_dict)+1), dtype=tf.float32)
    return ids_dict, params

size = 50
n_layers = 2
batch_size = 1000


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(
        size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse
        )

multi_cell = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(n_layers)]
)


def brown_lookup():
    corpus = nltk.corpus.brown
    words = (word.lower() for word in corpus.words())
    filtered_vocab = filter_vocab(count_words(words), 4)
    return prepare_word_lookup(filtered_vocab)


def sents_to_id_lists(sents, ids):
    converted = []
    for sent in sents:
        word_ids = []
        for word in sent:
            try:
                word_ids.append(ids[word.lower()])
            except KeyError:
                word_ids.append(0)
        converted.append(word_ids)
    return converted


'''

tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

def prepare_word_lookup(vocab):
    ids_dict = prepare_words_ids_lookup(vocab)
    lookup = tf.nn.embedding_lookup(
        tf.constant(np.identity(len(ids_dict)), dtype=tf.float32),
        list(range(len(ids_dict))))
    return ids_dict, lookup


cell.zero_state(batch_size, data_type())

def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(
        size, forget_bias=0.0, state_is_tuple=True,
        reuse=tf.get_variable_scope().reuse)

cell = tf.contrib.rnn.MultiRNNCell(
    [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

self._initial_state = cell.zero_state(batch_size, data_type())

embedding = tf.get_variable(
    "embedding", [vocab_size, size], dtype=data_type())
inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
inputs = tf.unstack(inputs, num=num_steps, axis=1)
outputs, state = tf.contrib.rnn.static_rnn(
    cell, inputs, initial_state=self._initial_state)
'''