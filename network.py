import tensorflow as tf
import numpy as np
from ptb_reader import * # temporarly
#voc, _ = read_ptb()


class BasicConfig:
    size = 200
    depth = 2
    batch_size = 20
    learning_rate = 1.0
    max_epoch = 2
    max_grad_norm = 5

class RnnLm:
    """Model języka oparty na sieciach rnn. W pierszej wersji tylko do uczenia i ewaluacji, z komórkami lstm"""
    def __init__(self, train_path, validate_path, test_path, config, vocabulary):
        """Tworzy  trzy grafy obliczeniowe sieci z tymi samymi zmiennymi, taką samą arhitekturą, ale innymi źródłąmi
        danych wejściowych - czytające z train path, validate path i test path"""
        self.vocabulary = vocabulary
        self.config = config

        # below all trainable variables are defined
        self.cell = self.multi_cell(config.size, config.depth)
        self.weights = tf.Variable([[0.0] * self.vocabulary.size()] * self.config.size)
        self.bias = tf.Variable([0.0] * self.vocabulary.size())
        with tf.variable_scope("net", reuse=False):
            self.train_data = self.value_from_file(train_path)
            self.train_graph = self.build_compute_graph(self.train_data)
        with tf.variable_scope("net", reuse=True):
            self.validate_data, self.test_data = self.value_from_file(validate_path), self.value_from_file(test_path)
            self.validate_graph = self.build_compute_graph(self.validate_data)
            self.test_graph = self.build_compute_graph(self.test_data)

        #self.train()

    def value_from_file(self, data_path):
        filename_queue = tf.train.string_input_producer([data_path])
        reader = tf.TFRecordReader()
        key, value = reader.read(filename_queue)
        return value

    def build_compute_graph(self, input_value):
        seq_lengths, sequences = self.batched_input(input_value)
        #sequence_max_len = sequences.shape[1]
        embeddings_flattened = tf.nn.embedding_lookup(self.vocabulary.get_lookup_tensor(), tf.reshape(sequences, (-1,)))
        embeddings_reshaped = tf.reshape(
            embeddings_flattened,
            (self.config.batch_size, -1, self.vocabulary.vector_length())
        )
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(
                self.cell, embeddings_reshaped, sequence_length=seq_lengths, dtype=tf.float32)

        stacked_outs = tf.reshape(outputs, (-1, self.config.size))
        so = tf.matmul(stacked_outs, self.weights) + self.bias
        logits = tf.reshape(so, (self.config.batch_size, -1, self.vocabulary.size())) # -1 replaces unknown max len of sequence in a batch
        predictions = tf.nn.softmax(logits, dim=2)
        cost = self.cost(seq_lengths, sequences, logits) # logits/predictions?
        return {"predictions": predictions, "cost": cost}

    def batched_input(self, input_value):
        batches = tf.train.batch(list(self.get_single_example(input_value)), batch_size=self.config.batch_size,
                                 dynamic_pad=True)
        seq_lengths, sequences = batches
        return seq_lengths, sequences

    def get_single_example(self, value):
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=value,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return context_parsed['length'],sequence_parsed['tokens']

    def lstm_cell(self, size):
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)

    def multi_cell(self, size, layers):
        return tf.contrib.rnn.MultiRNNCell(
            [self.lstm_cell(size) for _ in range(layers)], state_is_tuple=True)

    def cost(self, lengths, sequences, predictions):
        loss = tf.contrib.seq2seq.sequence_loss(
            predictions,
            sequences,
            self.mask(tf.shape(sequences), lengths))
        cost = tf.reduce_sum(loss) / self.config.batch_size
        return cost

    def mask(self, shape, unmasked):    # TODO: tf.sequence_mask
        """
        Returns 2D tensor with ones and zeros. First "unmasked" elements in a row are ones, other are zeros.
        """
        zero_mat = tf.zeros(shape=shape, dtype=tf.int64)
        width = tf.reduce_sum(zero_mat[0, :] + 1)
        range_matrix = zero_mat + tf.range(width, dtype=tf.int64)
        return tf.cast(range_matrix < (tf.zeros(shape=shape, dtype=tf.int64) + tf.reshape(unmasked, (-1, 1))),
                       dtype=tf.float32)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def _run(self, sess, outputs, epoch_size, train_op=None):
        fetches = {k: outputs[k] for k in outputs}
        cost = 0
        if train_op:
            fetches["tain_op"] = train_op
        for step in range(epoch_size):
            results = sess.run(fetches)
            cost += results["cost"]
        perplexity = np.exp(cost/epoch_size)
        print("perplexity: {}".format(perplexity))
        return perplexity

    def train(self):
        #self._lr = tf.Variable(0.0, trainable=False)
        self._lr = tf.constant(self.config.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.train_graph["cost"], tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        '''self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)'''

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
        #with tf.Session() as session:
            for i in range(self.config.max_epoch):
                '''lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)'''
                #init = tf.global_variables_initializer()
                #sess.run(init)
                # Start populating the filename queue.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=session)

                print("Epoch: %d " % (i + 1))
                train_perplexity = self._run(session, self.train_graph, 420*5, self._train_op)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = self._run(session, self.validate_graph, 33*5)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = self._run(session, self.test_graph, 37*5)
            print("Test Perplexity: %.3f" % test_perplexity)

            coord.request_stop()
            coord.join(threads)

#net = RnnLm("train.pb", "validate.pb", "test.pb", BasicConfig, voc) # also temporarly