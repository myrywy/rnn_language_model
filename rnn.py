from typing import List, Tuple
import logging
import tensorflow as tf
import numpy as np
import math
import input_data

verbosity = "normal"

default_initializer = tf.random_uniform_initializer(-0.1, 0.1)


class BasicConfig:
    size = 100
    depth = 2
    batch_size = 20
    learning_rate = 1.0
    max_epoch = 30
    max_grad_norm = 5
    init_scale = 0.1
    lr_decay = 1 / 1.15


class NetworkConfig:
    cell_size = 200
    depth = 2
    offset = 0

class NetworkConfigOffset:
    cell_size = 200
    depth = 2
    offset = 1

class BasicTrainingConfig:
    max_epoch = 30
    max_grad_norm = 5
    lr_decay = 1 / 1.15  # not implemented yet
    learning_rate = 1.0


class MockVocab:
    pass


class Network:
    def __init__(self, config, vocabulary_size, variable_scope_name="network"):
        self.config = config
        self.variable_scope_name = variable_scope_name
        self.variable_scope = None
        self._scope_used = False
        with tf.variable_scope(variable_scope_name, reuse=False) as self.variable_scope:
            self.predict_template = tf.make_template("predicting_network",
                                                     sequence_elements_predictor,
                                                     create_scope_now_=True,
                                                     cell_size=config.cell_size,
                                                     depth=config.depth,
                                                     vocabulary_size=vocabulary_size,
                                                     offset=config.offset)
            self.evaluate_template = tf.make_template("evaluation_network",
                                                      sequence_element_predicting_loss,
                                                      create_scope_now_=True,
                                                      offset=config.offset)

    def get_prediction_network(self, input_source):
        with tf.variable_scope(self.variable_scope, reuse=True):
            return self.predict_template(input_source)

    def get_evaluate_network(self, input_source, expected_results):
        with tf.variable_scope(self.variable_scope, reuse=True):
            _, seq_lengths = input_source
            return self.evaluate_template(
                seq_lengths,
                self.get_prediction_network(input_source),
                expected_results)


class Trainer:
    def __init__(self, network, train_data, validation_data, test_data, config=BasicTrainingConfig):
        self._network = network
        self._current_learning_rate = NotImplemented
        self._iteration = 0
        self._in_train, self._in_validate, self._in_test = train_data, validation_data, test_data
        self._pause = False
        self.config = config

        train_iter = train_data.make_initializable_iterator()
        self._next_train, self._init_train = train_iter.get_next(name="next_sentences_train"), train_iter.initializer

        validation_iter = validation_data.make_initializable_iterator()
        self._next_validation, self._init_validation = validation_iter.get_next(name="next_sentences_validation"), validation_iter.initializer

        test_iter = test_data.make_initializable_iterator()
        self._next_test, self._init_test = test_iter.get_next(name="next_sentences_test"), test_iter.initializer

        #with tf.variable_scope("trained_net", reuse=False):
        self.train_cost = self._network.get_evaluate_network(*self._next_train)
        #with tf.variable_scope("trained_net", reuse=True):
        self.validation_cost = self._network.get_evaluate_network(*self._next_validation)
        self.test_cost = self._network.get_evaluate_network(*self._next_test)

        self._train_op = self._get_train_op()

    def _get_train_op(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.train_cost, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        return optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())


    @staticmethod
    def aggregate_loss_value(loss_values):
        """This is used for aggregation of loss values from all bunches in a dataset."""
        return sum(loss_values)/len(loss_values)

    def run_train(self, session):
        """Loops through train dataset and run evaluation and training operation. """
        while True:
            try:
                r = session.run({"train":self._train_op, "loss": self.train_cost})
                logging.info("training loss: {}".format(r["loss"]))
            except tf.OutOfRangeError:
                break

        raise NotImplementedError

    def run_validate(self, session):
        raise NotImplementedError

    def run_test(self, session):
        raise NotImplementedError

    def run_full_training(self, session):
        """Run self.config.max_iterations cycles - train, validate and at the end test it"""
        for i in range(self._iteration, self.config.max_epoch):
            self.run_train()
            self._iteration += 1


def predicting_network(input_batch):
    return predictions


def test_train(session):
    import random
    # inventory
    a, b, c, d, e, f = [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,1,0]
    voc = [a, b, c, d, e, f]
    predictable_subsequence = [a,b,a,b,a,c]
    def mock_dataset(size, max_len, n_classes, prob):
        """ Funkcja generuje dane wejściowe i wyjściowe przy czym z prawdopodobieństwem prob
        podmienia losowy podciąg zdania na predictable_subsequence i dla ostatniego slowa
        tego podciagu zamienia wyjscie na n_classes - 1 (czyli ta ostatnia klasa jest
        zarezerwowana dla tego ciagu)"""
        data = []
        data_output = []
        for i in range(size):
            data.append([])
            data_output.append([])
            for _ in range(max_len):
                data[-1].append(voc[random.randint(0, len(voc) - 1)])
                data_output[-1].append(random.randint(0, n_classes - 2))
            if random.random() < prob:
                rand_start = random.randint(0, max_len - 1 - len(predictable_subsequence))
                data[-1][rand_start: rand_start + len(predictable_subsequence)] = predictable_subsequence
                data_output[-1][rand_start + len(predictable_subsequence) - 1] = n_classes - 1
        return (tf.constant(data, dtype=tf.float32), tf.constant([20]*size)), tf.constant(data_output)

    train_data = mock_dataset(1000, 20, 10, 0.5)
    validate_data = mock_dataset(100, 20, 10, 0.5)
    test_data = mock_dataset(100, 20, 10, 0.5)
    print(train_data)
    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(20)
    print(train_data)
    validate_data = tf.data.Dataset.from_tensor_slices(validate_data).batch(20)
    test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(20)

    manual_test_data1 = [a, b, a, b, a, c, e, e, e,e ,e , e ,e , e, e, e, e, e, e, e]

    '''with tf.Session() as sess:
        it = train_data.make_initializable_iterator()
        sess.run(it.initializer)
        (sentence, length), output = it.get_next()
        while True:
            print(sess.run({"sent": sentence, "length": length, "output": output}), sep="\n")'''

    net1 = Network(NetworkConfig, 10, "net1")
    net2 = Network(NetworkConfigOffset, 10, "net2")
    tr1 = Trainer(net1, train_data, validate_data, test_data)
    tr1.run_full_training()

    #pr1 = net1.get_prediction_network()
    #pr2 = net2.get_prediction_network()
    return net1, net2


#TODO: powienienem to zrefaktoryzować, żeby przyjmować wprost zdania, długości zdań zamiast tupla oinput_value
def sequence_elements_predictor(input_value,
                                cell_size,
                                depth,
                                vocabulary_size,
                                offset=0,
                                initializer=default_initializer):
    """Produkuje sieć RNN zwracającą prawdopodobieństwa(-inf; +inf)
    słów z vocabulary dla kolejnych elementów zdań wejściowych
    Elementy wejściowe powinny być reprezentowane przez wektor.
    :param input_value: tuple 2 tensorów, z których pierwszy to tensor o długości batch_size rzędu 1
        - powinien on zawierać długości kolejych zdań w batchu; drugi to tensor o wymiarach (batch_size,
        max_sentence_size, word_vector_size)
    :param cell_size: rozmiar wyjściowego wektora komórki lstm
    :param depth: liczba warstw lstm
    :param vocabulary_size: liczba słów w słowniku tj. liczba klas, z których można wybierać
    :param offset: wprowadza <offset> początkowych wektorów zerowych dodawanych przed zdaniem
    :param internal_size: rozmiar wektora wyjściowego komórki LSTM"""
    with tf.name_scope("unpack"):
        embedded_seqs, seq_lengths = input_value
        batch_size = tf.shape(embedded_seqs)[0]
    # note - start marks are prepended to INPUTS so they are represented by vectors so their rank is 3 (batch
    # size dimension, max sequence length dimesion, vector size dimension)

    weights = tf.get_variable(shape=(cell_size, vocabulary_size),
                              name="weights",
                              initializer=initializer)
    bias = tf.get_variable(shape=(vocabulary_size,),
                           name="bias",
                           initializer=initializer)
    with tf.name_scope("apply_offset"):
        start_marks_shape = [tf.shape(embedded_seqs)[0], tf.constant(offset), tf.shape(embedded_seqs)[2]]
        start_marks_shape = tf.stack(start_marks_shape, axis=0)
        start_marks = tf.zeros(shape=start_marks_shape,
                               dtype=embedded_seqs.dtype)
        extended_embedded_seqs = tf.concat([start_marks, embedded_seqs], axis=1)
        extended_length = seq_lengths + offset

    cell = multi_cell(cell_size, depth)
    zero_state = cell.zero_state(tf.cast(batch_size, tf.int32), tf.float32)
    with tf.variable_scope("RNN"):
        outputs, state = tf.nn.dynamic_rnn(
            cell,
            extended_embedded_seqs,
            sequence_length=extended_length,
            dtype=tf.float32,
            initial_state=zero_state)

    stacked_outs = tf.reshape(outputs, (-1, cell_size))
    so = tf.matmul(stacked_outs, weights) + bias
    logits_shape = tf.concat(axis=0, values=[[batch_size], tf.constant([-1, vocabulary_size], dtype=tf.int32)])
    logits_shape = tf.cast(logits_shape, tf.int32)
    logits = tf.reshape(so, logits_shape)
    return logits

def lstm_cell(size):
    return tf.contrib.rnn.BasicLSTMCell(
        size, forget_bias=0.0, state_is_tuple=True,
        reuse=tf.get_variable_scope().reuse)

def multi_cell(size, layers):
    return tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(size) for _ in range(layers)], state_is_tuple=True)









def sequence_element_predicting_loss(sequence_lengths, logits, correct, offset):
    # end marks are appended to EXPECTED OUTPUTS so they are represented by id so their rank is 2 (batch size,
    # max sequence length dimension)
    end_marks_shape = tf.stack([tf.shape(correct)[0], tf.constant(offset)], axis=0)
    end_marks = tf.zeros(shape=end_marks_shape,
                         dtype=correct.dtype)
    extended_correct = tf.concat([correct, end_marks], axis=1)

    return cost(sequence_lengths, extended_correct, logits)


def cost(lengths, sequences, predictions):
    """

    :param lengths:
    :param sequences: Tensor of shape (batch size, max sent size, vocab size)
    :param predictions: Tensor of shape (batch size, max sent)
    :return:
    """

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=sequences, logits=predictions)
    target_weights = mask(tf.shape(sequences), lengths)
    target_weights *= mask_unknown(sequences)
    train_loss = tf.reduce_sum(crossent * target_weights)
    train_loss /= tf.cast(tf.shape(sequences)[0], dtype=tf.float32)
    return train_loss

def mask(shape, unmasked):    # TODO: tf.sequence_mask
    """
    Returns 2D tensor with ones and zeros. First ("unmasked") elements in a row are ones, other are zeros.
    """
    zero_mat = tf.zeros(shape=shape, dtype=tf.int64)
    width = tf.reduce_sum(zero_mat[0, :] + 1)
    range_matrix = zero_mat + tf.range(width, dtype=tf.int64)
    return tf.cast(range_matrix < (tf.zeros(shape=shape, dtype=tf.int64) + tf.reshape(unmasked, (-1, 1))),
                   dtype=tf.float32)

def mask_unknown(sequences, masc_type=tf.float32):
    nonzeros = tf.not_equal(sequences, 0, "nonzeros")
    return tf.cast(nonzeros, masc_type, "nonzero_mask")











def cos_do_trainera():
    # end marks are appended to EXPECTED OUTPUTS so they are represented by id so their rank is 2 (batch size,
    # max sequence length dimension)
    end_marks_shape = start_marks_shape[0:2]
    end_marks = tf.zeros(shape=end_marks_shape,
                          dtype=sequences.dtype)
    extended_sequences = tf.concat([sequences, end_marks], axis=1)

    predictions = logits
    cost = self.cost(seq_lengths, extended_sequences, logits)
    self.summaries[phase].append(tf.summary.scalar('{}_cost'.format(phase), cost))
    return {"predictions": predictions, "cost": cost, "sequences": sequences, "embeded_seqs": embeded_seqs, "rnn_outputs": outputs, "stacked_outs": so, "logits_shape": logits_shape, "logits": logits, "zero_state": zero_state, "state": state}











def training_network(predicting_network, expected_batch):
    return loss


class Rnn:
    def __init__(self, train_set, validate_set, test_set, config, vocabulary, model_file=None):
        """Tworzy  trzy grafy obliczeniowe sieci z tymi samymi zmiennymi, taką samą arhitekturą, ale innymi źródłąmi
        danych wejściowych - czytające z train path, validate path i test path"""
        self.vocabulary = vocabulary
        self.config = config
        initializer = tf.random_uniform_initializer(-self.config.init_scale,
                                                    self.config.init_scale)
        train_set, validate_set, test_set = [self._normalize_input_type(obj)
                                             for obj in (train_set, validate_set, test_set)]
        self.train_set = train_set
        self.validate_set = validate_set
        self.test_set = test_set

        self._cells_created = 0

        self.summaries = {"train": [], "validate": [], "test": [], "production": []}

        # below all trainable variables are defined
        self.cell = self.multi_cell(config.size, config.depth)
        self.weights = tf.get_variable(shape=(self.config.size, self.vocabulary.size()), name="weights", initializer=initializer)
        self.bias = tf.get_variable(shape=(self.vocabulary.size(),), name="bias", initializer=initializer)

        with tf.variable_scope("net", reuse=False, initializer=initializer):
            self.train_data, self.train_iter = self.get_sentence(train_set)
            self.train_graph = self.build_compute_graph(self.train_data, "train")
        with tf.variable_scope("net", reuse=True, initializer=initializer):
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

        self._lr = tf.Variable(0.0, trainable=False)
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self.end_of_sentence_id = 0

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
        # note - start marks are prepended to INPUTS so they are represented by vectors so their rank is 3 (batch
        # size dimension, max sequence length dimesion, vector size dimension);
        # end marks are appended to EXPECTED OUTPUTS so they are represented by id so their rank is 2 (batch size,
        # max sequence length dimension)
        start_marks_shape = tf.stack([tf.shape(embeded_seqs)[0], tf.constant(1), tf.shape(embeded_seqs)[2]], axis=0)
        start_marks = tf.zeros(shape=start_marks_shape,
                              dtype=embeded_seqs.dtype)
        end_marks_shape = start_marks_shape[0:2]
        end_marks = tf.zeros(shape=end_marks_shape,
                              dtype=sequences.dtype)
        extended_sequences = tf.concat([sequences, end_marks], axis=1)
        extended_embeded_seqs = tf.concat([start_marks, embeded_seqs], axis=1)
        extended_length = seq_lengths
        zero_state = self.cell.zero_state(tf.cast(batch_size, tf.int32), tf.float32)
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(
                self.cell,
                extended_embeded_seqs,
                sequence_length=extended_length,
                dtype=tf.float32,
                initial_state=zero_state)

        stacked_outs = tf.reshape(outputs, (-1, self.config.size))
        so = tf.matmul(stacked_outs, self.weights) + self.bias
        logits_shape = tf.concat(axis=0, values=[[batch_size], tf.constant([-1, self.vocabulary.size()], dtype=tf.int64)])
        logits_shape = tf.cast(logits_shape, tf.int32)
        logits = tf.reshape(so, logits_shape) # -1 replaces unknown max len of sequence in a batch
        predictions = logits#tf.nn.softmax(logits, dim=2)
        cost = self.cost(seq_lengths, extended_sequences, logits)
        self.summaries[phase].append(tf.summary.scalar('{}_cost'.format(phase), cost))
        return {"predictions": predictions, "cost": cost, "sequences": sequences, "embeded_seqs": embeded_seqs, "rnn_outputs": outputs, "stacked_outs": so, "logits_shape": logits_shape, "logits": logits, "zero_state": zero_state, "state": state}

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
        """

        :param lengths:
        :param sequences: Tensor of shape (batch size, max sent size, vocab size)
        :param predictions: Tensor of shape (batch size, max sent)
        :return:
        """

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sequences, logits=predictions)
        target_weights = self.mask(tf.shape(sequences), lengths)
        target_weights *= self.mask_unknown(sequences)
        train_loss = tf.reduce_sum(crossent * target_weights)
        train_loss /= tf.cast(tf.shape(sequences)[0], dtype=tf.float32)
        return train_loss

    def mask(self, shape, unmasked):    # TODO: tf.sequence_mask
        """
        Returns 2D tensor with ones and zeros. First ("unmasked") elements in a row are ones, other are zeros.
        """
        zero_mat = tf.zeros(shape=shape, dtype=tf.int64)
        width = tf.reduce_sum(zero_mat[0, :] + 1)
        range_matrix = zero_mat + tf.range(width, dtype=tf.int64)
        return tf.cast(range_matrix < (tf.zeros(shape=shape, dtype=tf.int64) + tf.reshape(unmasked, (-1, 1))),
                       dtype=tf.float32)

    def mask_unknown(self, sequences, masc_type=tf.float32):
        nonzeros = tf.not_equal(sequences, 0, "nonzeros")
        return tf.cast(nonzeros, masc_type, "nonzero_mask")

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})







class Predictor:
    def get_sentence_probability(self, sent):
        probs = predict_words_in_sentence(sent, list(range(len(sent))))
        word_probabilities = []
        for word, predictions in zip(sent, probs):
            word_probability = 0
            for predicted_word, prob in reversed(predictions):
                if word == predicted_word:
                    word_probability = prob
                    break
            if word_probability != 0:
                word_probabilities.append(math.log(word_probability))
        return math.exp(sum(word_probabilities))

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
        fetches = {k: outputs[k] for k in outputs if k} # if k in {"sequences", "embeded_seqs", "rnn_outputs", "stacked_outs", "logits_shape", "logits", "predictions"}
        cost = 0
        if train_op is not None:
            fetches["tain_op"] = train_op
        if summary_op is not None:
            pass#fetches["summary"] = summary_op
        step = 0
        while True:
            step += 1
            try:
                results = sess.run(fetches)
                print(*results.keys())
                cost += results["cost"]
                if step % 1 == 0:
                    print("step:", step)
                    print("cost: {}".format(results["cost"]))
                    print("mean cost: {}".format(cost/step))
                    print("logits_shape: {}".format(results["logits_shape"]))
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
                        pass
                        #summary = sess.run(self.summary)
                        #results = sess.run({**fetches, "summary": self.summary})
                        #self.train_writer.add_summary(results["summary"], step)
                        #self.sv.summary_computed(sess, results["summary"])

            except tf.errors.OutOfRangeError:
                break

            except tf.errors.InvalidArgumentError:
                print("Warn: Batch rejected (probably at the end of a dataset)")
                continue

        perplexity = cost/step
        print("perplexity: {}".format(perplexity))
        return perplexity







class RnnTrainer:
    def load_from_files(self, sess):
        new_saver = tf.train.Saver()
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))


    def train(self):
        #self._lr = tf.constant(self.config.learning_rate)
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
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=config) as session:
            train_summary = tf.summary.merge(self.summaries["train"])
            validate_summary = tf.summary.merge(self.summaries["validate"])
            test_summary = tf.summary.merge(self.summaries["test"])
            self.train_writer = tf.summary.FileWriter('./train')
            self.train_writer.add_graph(session.graph)
            tf.global_variables_initializer().run()
            #tf.get_default_graph().finalize()
            for i in range(self.config.max_epoch):
                lr_decay = self.config.lr_decay ** max(i + 1 - self.config.max_epoch, 0.0)
                self.assign_lr(session, self.config.learning_rate * lr_decay)
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
                saver.save(session, "./language_model")

            session.run(self.test_iter.initializer)
            test_perplexity = self._run(session, self.test_graph, summary_op=test_summary)
            print("Test Perplexity: %.3f" % test_perplexity)

            coord.request_stop()
            coord.join(threads)
