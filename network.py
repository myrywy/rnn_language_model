from typing import List, Tuple
import tensorflow as tf
import numpy as np
import input_data

verbosity = "normal"

class BasicConfig:
    size = 200
    depth = 1
    batch_size = 20
    learning_rate = 1.0
    max_epoch = 10
    max_grad_norm = 5

class RnnLm:
    """Model języka oparty na sieciach rnn. W pierszej wersji tylko do uczenia i ewaluacji, z komórkami lstm"""
    def __init__(self, train_set, validate_set, test_set, config, vocabulary, model_file=None):
        """Tworzy  trzy grafy obliczeniowe sieci z tymi samymi zmiennymi, taką samą arhitekturą, ale innymi źródłąmi
        danych wejściowych - czytające z train path, validate path i test path"""
        self.vocabulary = vocabulary
        self.config = config
        train_set, validate_set, test_set = [self._normalize_input_type(obj)
                                             for obj in (train_set, validate_set, test_set)]
        #train_set = train_set.take(25)
        #validate_set = validate_set.take(22)
        #test_set = test_set.take(34)
        self.train_set = train_set
        self.validate_set = validate_set
        self.test_set = test_set

        self._cells_created = 0

        self.summaries = {"train": [], "validate": [], "test": [], "production": []}

        # below all trainable variables are defined
        self.cell = self.multi_cell(config.size, config.depth)
        self.weights = tf.get_variable(shape=(self.config.size, self.vocabulary.size()), name="weights", initializer=tf.contrib.layers.xavier_initializer())
        self.bias = tf.get_variable(shape=(self.vocabulary.size(),), name="bias")
        #tf.summary.tensor_summary('weights', self.weights)
        #tf.summary.tensor_summary('bias', self.bias)
        with tf.variable_scope("net", reuse=False):
            self.train_data, self.train_iter = self.get_sentence(train_set)
            self.train_graph = self.build_compute_graph(self.train_data, "train")
        with tf.variable_scope("net", reuse=True):
            self.validate_data, self.validate_iter = self.get_sentence(validate_set)
            self.test_data, self.test_iter = self.get_sentence(test_set)
            self.validate_graph = self.build_compute_graph(self.validate_data, "validate")
            self.test_graph = self.build_compute_graph(self.test_data, "test")
            self.production_input_data = input_data.InputData(vocabulary)
            self.production_input_sequences = tf.placeholder(dtype=tf.int64, name="sequences")
            self.production_input_sequence_lengths = tf.placeholder(dtype=tf.int64, name="sequence_lengths")
            production_input_dataset = self._normalize_input_type((self.production_input_sequence_lengths,
                                                                   self.production_input_sequences))
            self.production_batch_size = tf.placeholder(dtype=tf.int64, shape=())
            self.production_data, self.production_iter = self.get_sentence(production_input_dataset,
                                                                           batch_size=self.production_batch_size)
            self.production_graph = self.build_compute_graph(self.production_data,
                                                             "production",
                                                             batch_size=self.production_batch_size)
        self.production_session = None

    def _normalize_input_type(self, input_object):
        if isinstance(input_object, tf.contrib.data.Dataset):
            return input_object
        return tf.contrib.data.Dataset.from_tensor_slices(input_object)

    def get_sentence(self, dataset, batch_size=None):
        """
        Get op that gets next element (sentence and its length) form dataset

        :param dataset: Dataset from which sentences are taken
        :param batch_size: If None then batch size is read from self.config
        :return: tuppe - next element op (element consist of length and 1D vector of ids of words) and iterator itself
            (the iterator object should be initialised at the beggining of a session)
        """

        dataset = dataset.map(
            lambda seq_len, word_ids: (seq_len, word_ids, tf.gather(self.vocabulary.get_lookup_tensor(), word_ids))
        )
        if batch_size is None:
            batch_size = self.config.batch_size
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=((), (-1,), (-1, self.vocabulary.vector_length()))
        )

        iterator = dataset.make_initializable_iterator()
        sentence = iterator.get_next()
        return sentence, iterator

    def build_compute_graph(self, input_value, phase, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        seq_lengths, sequences, embeded_seqs = input_value
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(
                self.cell, embeded_seqs, sequence_length=seq_lengths, dtype=tf.float32)

        stacked_outs = tf.reshape(outputs, (-1, self.config.size))
        so = tf.matmul(stacked_outs, self.weights) + self.bias
        logits_shape = tf.concat(axis=0, values=[[batch_size], tf.constant([-1, self.vocabulary.size()], dtype=tf.int64)])
        logits_shape = tf.cast(logits_shape, tf.int32)
        logits = tf.reshape(so, logits_shape) # -1 replaces unknown max len of sequence in a batch
        predictions = tf.nn.softmax(logits, dim=2)
        cost = self.cost(seq_lengths, sequences, logits)
        self.summaries[phase].append(tf.summary.scalar('{}_cost'.format(phase), cost))
        return {"predictions": predictions, "cost": cost, "sequences": sequences, "embeded_seqs": embeded_seqs}

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

    def predict_word(self, sentence: List[str], index: int, **kw):
        return self.predict_words_in_sentece(sentence, [index], **kw)[0]

    def predict_words_in_sentece(self, sentence: List[str], indices: List[int], **kw):
        return self.predict_multiple_words([sentence], [indices], **kw)[0]

    def predict_multiple_words(self, sentences: List[List[str]], indices: List[List[int]], sort=False) -> List[List[List[Tuple[int, str, float]]]]:
        """
        Return probabilities of words from vocabulary on positions indicated by indices.
        :param sentences: List of sentences represented as lists of words.
        :param indices: i-th element of indices should be a list of indices of words in i-th sentence that are to be
            predicted.
        :return: Results structured as indices but with lists of words from vocabulary with probabilities instead of
            insices
        """
        if not self.production_session:
            self.production_session = tf.Session()
            self.load_from_files(self.production_session)
        seqs = list(self.production_input_data.sents_to_id_lists(sentences))
        seqlens = [len(s) for s in seqs]
        seqs_array = np.zeros((len(seqs), max(seqlens)), dtype=np.int64)
        for i, seqs in enumerate(seqs):
            seqs_array[i] = np.array(seqs)
        seqlens = np.array(seqlens, dtype=np.int64)
        results = self.predict(self.production_session, feed_dict={self.production_input_sequence_lengths: seqlens,
                                                                   self.production_input_sequences: seqs_array,
                                                                   self.production_batch_size: len(seqlens)})
        probabilities = []
        for sentence_number, word_indices in enumerate(indices):
            probabilities.append([])
            for index in word_indices:
                raw_probabilities = results['predictions'][sentence_number][index]
                probabilities[-1].append(
                    [(id, self.vocabulary.id2word(id), p) for id, p in enumerate(raw_probabilities)]
                )
                if sort:
                    probabilities[-1][-1] = list(sorted(probabilities[-1][-1], key=lambda x: x[2]))
        return probabilities

    def predict(self, session, feed_dict=None):
        session.run(self.train_iter.initializer, feed_dict)
        session.run(self.validate_iter.initializer, feed_dict)
        session.run(self.test_iter.initializer, feed_dict)
        session.run(self.production_iter.initializer, feed_dict)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=session)

        results = session.run(self.production_graph, feed_dict)

        coord.request_stop()
        coord.join(threads)
        return results

    def _run(self, sess, outputs, train_op=None, summary_op=None):
        fetches = {k: outputs[k] for k in outputs}
        cost = 0
        if train_op is not None:
            fetches["tain_op"] = train_op
        if summary_op is not None:
            fetches["summary"] = summary_op
        step = 0
        while True:
            step += 1
            try:
                results = sess.run(fetches)
                cost += results["cost"]
                if step % 1 == 0:
                    print("step:", step)
                    print("cost: {}".format(results["cost"]))
                    print("mean cost: {}".format(cost/step))
                    if verbosity == "insane":
                        print("batch: \n{}".format(results["sequences"]))
                        for i in range(len(results["embeded_seqs"])):
                            sent = results["embeded_seqs"][i]
                            sent_words = []
                            for j in range(len(sent)):
                                word_vec = sent[j]
                                #print("word vec", word_vec)
                                identifier = self.vocabulary.one_hot_to_id(word_vec)
                                sent_words.append(self.vocabulary.id2word(identifier))
                            print(" ".join(sent_words).replace("<UNKNOWN>", ""))
                    if summary_op is not None:
                        #summary = sess.run(self.summary)
                        #results = sess.run({**fetches, "summary": self.summary})
                        self.train_writer.add_summary(results["summary"], step)
                        #self.sv.summary_computed(sess, results["summary"])

            except tf.errors.OutOfRangeError:
                break

            except tf.errors.InvalidArgumentError:
                print("Warn: Batch rejected (probably at the end of a dataset)")
                continue

        perplexity = cost/step
        print("perplexity: {}".format(perplexity))
        return perplexity

    def load_from_files(self, sess):
        new_saver = tf.train.Saver()
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))

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
        saver = tf.train.Saver()
        with tf.Session() as session:
            train_summary = tf.summary.merge(self.summaries["train"])
            validate_summary = tf.summary.merge(self.summaries["validate"])
            test_summary = tf.summary.merge(self.summaries["test"])
            self.train_writer = tf.summary.FileWriter('./train')
            self.train_writer.add_graph(session.graph)
            tf.global_variables_initializer().run()
            for i in range(self.config.max_epoch):
                session.run(self.train_iter.initializer)
                session.run(self.validate_iter.initializer)
                session.run(self.test_iter.initializer)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=session)

                print("Epoch: %d " % (i + 1))
                train_perplexity = self._run(session, self.train_graph, train_op=self._train_op, summary_op=train_summary)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = self._run(session, self.validate_graph, summary_op=validate_summary)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            session.run(self.test_iter.initializer)
            test_perplexity = self._run(session, self.test_graph, summary_op=test_summary)
            print("Test Perplexity: %.3f" % test_perplexity)

            coord.request_stop()
            coord.join(threads)
            saver.save(session, "language_model")


class ProductionRnnLm:
    def __init__(self, sents, config, vocabulary):
        '''self.lengths = tf.placeholder(shape=(-1,), dtype=tf.int32)
        self.train_input = tf.placeholder(shape=(1, -1), dtype=tf.int32)
        self.validate_input = tf.placeholder(shape, dtype)
        self.test_input = tf.placeholder(shape, dtype)
        self.rnn = RnnLm(tf.contrib.data.Dataset.from_tensors(self.train_input),
                         tf.contrib.data.Dataset.from_tensors(self.validate_input),
                         tf.contrib.data.Dataset.from_tensors(self.test_input),
                         config,
                         vocabulary)'''
        self.train_input = input_data.InputData(vocabulary, sents=[[]])
        self.validate_input = input_data.InputData(vocabulary, sents=[[]])
        '''self.test_input = input_data.InputData(vocabulary, sents=sents)'''
        self.input_data = input_data.InputData(vocabulary)
        self.test_input = (tf.placeholder(dtype=tf.int64, name="seqlens"), tf.placeholder(dtype=tf.int32, name="seqs"))
        self.rnn = RnnLm(self.train_input.dataset(),
                         self.validate_input.dataset(),
                         self.test_input,
                         config,
                         vocabulary)
        self.session = None

    def open(self):
        self.session = tf.Session()
        self.session.__enter__()
        self.rnn.load_from_files(self.session)

    def run(self, sentences=None):
        if not self.session:
            raise RuntimeError("You must open a tensorflow session first.")
        if sentences:
            seqs = list(self.input_data.sents_to_id_lists(sentences))
            seqlens = [len(s) for s in seqs]
            seqs_array = np.ndarray((len(seqs), max(seqlens)), dtype=np.int32)
            for i, seqs in enumerate(seqs):
                seqs_array[i] = np.array(seqs)
            seqlens = np.array(seqlens, dtype=np.int64)
            #self.test_input.close()
        results = self.rnn.predict(self.session, feed_dict={self.test_input[0]: seqlens, self.test_input[1]: seqs_array})
        probabilities = results['predictions'][0][-1]
        return [(self.rnn.vocabulary.id2word(id), id, p) for id, p in enumerate(probabilities)]

    def close(self):
        self.session.__exit__(None, None, None)


from ptb_reader import *  # temporarly
def ptb_test():
    voc, (tr, val, tst) = read_ptb()
    net = RnnLm(tr.dataset(), val.dataset(), tst.dataset(), BasicConfig, voc) # also temporarly
    return net

def simple_train_test():
    voc, (tr, val, tst) = input_data.InputData.prepeare_one_hot_input(
        [["ala", "ma", "kota"]]*500,
        [["ala", "ma", "kota"]]*50,
        [["ala", "ma", "kota"]]*50)
    net = RnnLm(tr.dataset(), val.dataset(), tst.dataset(), BasicConfig, voc) # also temporarly
    net.train()
    most_prob = [prediction[-1][1] for prediction in net.predict_words_in_sentece(["ala", "ma", "kota"], [0,1,2], sort=True)]
    assert most_prob == ["ala", "ma", "kota"]
    return net


def random_train_test():
    import random
    words = list("abcdefghaijklmnoprst")

    def get_test_set(n):
        test_sents = [["ala", "ma", "kota"]] * n
        for i, sent in enumerate(test_sents):
            prefix = [words[random.randint(0, len(words)-1)] for _ in range(random.randint(0, 5))]
            sufix = [words[random.randint(0, len(words)-1)] for _ in range(random.randint(0, 5))]
            test_sents[i] = prefix + sent + sufix
        return test_sents

    voc, (tr, val, tst) = input_data.InputData.prepeare_one_hot_input(
        get_test_set(500),
        get_test_set(50),
        get_test_set(50))
    net = RnnLm(tr.dataset(), val.dataset(), tst.dataset(), BasicConfig, voc) # also temporarly
    net.train()
    most_prob = [prediction[-1][1] for prediction in net.predict_words_in_sentece(["ala", "ma", "kota"], [0,1,2], sort=True)]
    #assert most_prob == ["ala", "ma", "kota"]
    return net


def restore_test():
    voc, (tr, val, tst) = read_ptb()
    config = BasicConfig()
    config.batch_size = 1
    net = ProductionRnnLm([["I", "have", "a", "cat"]], config, voc) # also temporarly
    net.open()
    #net.run()
    return net

if __name__ == "__main__" and False:
    net = ptb_test()
    net.train()

if __name__ == "__main__" and False:
    net = restore_test()
