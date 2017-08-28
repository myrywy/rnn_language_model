from unittest import TestCase, main, skip
import numpy as np
import tensorflow as tf
import tempfile
from vocabulary import Vocabulary
from input_data import InputData


class TestInputData(TestCase):
    def setUp(self):
        self.words = ["a", "b", "c", "d", "e"]
        self.ids = [1, 3, 5, 7, 9]
        self.vectors = np.array([[0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1]])
        self.vocab = Vocabulary(self.words, self.ids, self.vectors)

    def test_sents_to_id_lists(self):
        sents = [["a", "b", "c"], ["b", "c"], ["a"], []]
        sents_ids = [[1, 3, 5], [3, 5], [1], []]
        input_data = InputData(self.vocab, sents)
        out = input_data.sents_to_id_lists(sents)
        self.assertEqual(sents_ids, list(out))

    def test_write(self):
        sents = [["a", "b", "c"], ["b", "c"], ["a"], []]
        sents_ids = [[1, 3, 5], [3, 5], [1], []]
        input_data = InputData(self.vocab, sents)
        with tempfile.NamedTemporaryFile() as fp:
            input_data.write(fp.name)

            filenames = [fp.name]
            dataset = tf.contrib.data.TFRecordDataset(filenames)
            dataset = dataset.map(input_data.get_single_example)
            iterator = dataset.make_initializable_iterator()
            sentence = iterator.get_next()

            with tf.Session() as sess:
                sentences = []
                sess.run(iterator.initializer)
                while True:
                    try:
                        sentences.append(sess.run(sentence))
                    except tf.errors.OutOfRangeError:
                        break
            self.assertEqual([s[1].tolist() for s in sentences], sents_ids)

    def test_prepeare_one_hot_input(self):
        sents1 = [["a", "b", "c"], ["b", "c"], ["a"], []]
        sents2 = [["d", "e"], ["d"]]
        voc, (input_data_1, input_data_2) = InputData.prepeare_one_hot_input(sents1, sents2, min_word_count=0)
        self.assertEqual(set(voc.ids.keys()), {"a", "b", "c", "d", "e"})
        with tempfile.NamedTemporaryFile() as fp:
            input_data_1.write(fp.name)

            filenames = [fp.name]
            dataset = tf.contrib.data.TFRecordDataset(filenames)
            dataset = dataset.map(input_data_1.get_single_example)
            iterator = dataset.make_initializable_iterator()
            sentence = iterator.get_next()

            with tf.Session() as sess:
                sentences = []
                sess.run(iterator.initializer)
                while True:
                    try:
                        sentences.append(sess.run(sentence))
                    except tf.errors.OutOfRangeError:
                        break
            self.assertEqual([[voc.id2word(w) for w in s[1].tolist()] for s in sentences], sents1)


if __name__ == "__main__":
    main()
