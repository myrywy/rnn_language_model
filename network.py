import tensorflow as tf
import numpy as np


class BasicConfig:
    size = 200
    depth = 1
    batch_size = 20
    learning_rate = 1.0
    max_epoch = 2
    max_grad_norm = 5

class RnnLm:
    """Model języka oparty na sieciach rnn. W pierszej wersji tylko do uczenia i ewaluacji, z komórkami lstm"""
    def __init__(self, train_set, validate_set, test_set, config, vocabulary):
        """Tworzy  trzy grafy obliczeniowe sieci z tymi samymi zmiennymi, taką samą arhitekturą, ale innymi źródłąmi
        danych wejściowych - czytające z train path, validate path i test path"""
        self.vocabulary = vocabulary
        self.config = config

        self.train_set = train_set
        self.validate_set = validate_set
        self.test_set = test_set

        self._cells_created = 0

        # below all trainable variables are defined
        self.cell = self.multi_cell(config.size, config.depth)
        self.weights = tf.get_variable(shape=(self.config.size, self.vocabulary.size()), name="weights")
        self.bias = tf.get_variable(shape=(self.vocabulary.size(),), name="bias")
        #tf.summary.tensor_summary('weights', self.weights)
        #tf.summary.tensor_summary('bias', self.bias)
        with tf.variable_scope("net", reuse=False):
            self.train_data, self.train_iter = self.get_sentence(train_set)
            self.train_graph = self.build_compute_graph(self.train_data, True)
        with tf.variable_scope("net", reuse=True):
            self.validate_data, self.validate_iter = self.get_sentence(validate_set)
            self.test_data, self.test_iter = self.get_sentence(test_set)
            self.validate_graph = self.build_compute_graph(self.validate_data)
            self.test_graph = self.build_compute_graph(self.test_data)


    def get_sentence(self, dataset):
        """
        Get op that gets next element (sentence and its length) form dataset

        :param dataset: Dataset from which sentences are taken
        :return: tuppe - next element op (element consist of length and 1D vector of ids of words) and iterator itself
            (the iterator object should be initialised at the beggining of a session)
        """
        
        dataset = dataset.map(
            lambda seq_len, word_ids: (seq_len, word_ids, tf.gather(self.vocabulary.get_lookup_tensor(), word_ids))
        )
        dataset = dataset.padded_batch(
            self.config.batch_size,
            padded_shapes=((), (-1,), (-1, self.vocabulary.vector_length()))
        )

        iterator = dataset.make_initializable_iterator()
        sentence = iterator.get_next()
        return sentence, iterator

    def build_compute_graph(self, input_value, get_summary=False):
        seq_lengths, sequences, embeded_seqs = input_value
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(
                self.cell, embeded_seqs, sequence_length=seq_lengths, dtype=tf.float32)

        stacked_outs = tf.reshape(outputs, (-1, self.config.size))
        so = tf.matmul(stacked_outs, self.weights) + self.bias
        logits = tf.reshape(so, (self.config.batch_size, -1, self.vocabulary.size())) # -1 replaces unknown max len of sequence in a batch
        predictions = tf.nn.softmax(logits, dim=2)
        cost = self.cost(seq_lengths, sequences, logits)
        if get_summary:
            tf.summary.scalar('cost', cost)
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
        with tf.variable_scope("cell_" + str(self._cells_created)):
            self._cells_created += 1
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
            self.mask(tf.shape(sequences), lengths),
            average_across_timesteps=False,
            average_across_batch=False,
        )
        cost = tf.reduce_sum(loss) / self.config.batch_size
        return cost

    def mask(self, shape, unmasked):    # TODO: tf.sequence_mask
        """
        Returns 2D tensor with ones and zeros. First ("unmasked") elements in a row are ones, other are zeros.
        """
        zero_mat = tf.zeros(shape=shape, dtype=tf.int64)
        width = tf.reduce_sum(zero_mat[0, :] + 1)
        range_matrix = zero_mat + tf.range(width, dtype=tf.int64)
        return tf.cast(range_matrix < (tf.zeros(shape=shape, dtype=tf.int64) + tf.reshape(unmasked, (-1, 1))),
                       dtype=tf.float32)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def _run(self, sess, outputs, train_op=None):
        fetches = {k: outputs[k] for k in outputs}
        cost = 0
        if train_op:
            fetches["tain_op"] = train_op
            fetches["summary"] = self.summary
        step = 0
        while True:
            step += 1
            try:
                results = sess.run(fetches)
                cost += results["cost"]
                if step % 100 == 0:
                    print("step:", step)
                    print("cost: {}".format(results["cost"]))
                    print("mean cost: {}".format(cost/step))
                    if train_op:
                        #summary = sess.run(self.summary)
                        #results = sess.run({**fetches, "summary": self.summary})
                        self.train_writer.add_summary(results["summary"], step)
                        #self.sv.summary_computed(sess, results["summary"])

            except tf.errors.OutOfRangeError:
                break

        perplexity = cost/step
        print("perplexity: {}".format(perplexity))
        return perplexity

    def train(self):
        self._lr = tf.constant(self.config.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.train_graph["cost"], tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        #self.sv = sv = tf.train.Supervisor(logdir="./logs4/", summary_op=None)
        #with sv.managed_session() as session:
        with tf.Session() as session:
            self.summary = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./train', session.graph)
            tf.global_variables_initializer().run()
            for i in range(self.config.max_epoch):
                session.run(self.train_iter.initializer)
                session.run(self.validate_iter.initializer)
                session.run(self.test_iter.initializer)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=session)

                print("Epoch: %d " % (i + 1))
                train_perplexity = self._run(session, self.train_graph)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = self._run(session, self.validate_graph)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            session.run(self.test_iter.initializer)
            test_perplexity = self._run(session, self.test_graph)
            print("Test Perplexity: %.3f" % test_perplexity)

            coord.request_stop()
            coord.join(threads)


from ptb_reader import *  # temporarly
def ptb_test():
    voc, (tr, val, tst) = read_ptb()
    net = RnnLm(tr.dataset(), val.dataset(), tst.dataset(), BasicConfig, voc) # also temporarly
    return net

if __name__ == "__main__":
    net = ptb_test()
    net.train()
