from unittest import TestCase, main, skip
import numpy as np
import tensorflow as tf
from vocabulary import Vocabulary

class TestVocabulary(TestCase):
    def setUp(self):
        self.words = ["a", "b", "c", "d", "e", "f"]
        self.ids = [i+1 for i in range(len(self.words))]
        self.vectors = np.array([
            [0,0,0,0,0,0],
            [1,2,3,4,5,6],
            [0,1,2,3,4,5],
            [0,0,1,2,3,4],
            [0,0,0,1,2,3],
            [0,0,0,0,1,2],
            [0,0,0,0,0,1],
        ])
        self.voc1 = Vocabulary(self.words, self.ids, self.vectors)

    def test_word2id(self):
        self.assertEqual(self.ids, [self.voc1.word2id(w) for w in self.words])
        self.assertEqual(0, self.voc1.word2id("newword"))

    def test_id2word(self):
        self.assertEqual(self.words, [self.voc1.id2word(i) for i in self.ids])

    def test_word2vec(self):
        self.assertTrue((self.voc1.word2vec("a") == np.array([1,2,3,4,5,6])).all())
        self.assertTrue((self.voc1.word2vec("b") == np.array([0,1,2,3,4,5])).all())
        self.assertTrue((self.voc1.word2vec("f") == np.array([0,0,0,0,0,1])).all())
        self.assertTrue((self.voc1.word2vec("noword") == np.array([0,0,0,0,0,0])).all())

    def test_id2vec(self):
        self.assertTrue((self.voc1.id2vec(1) == np.array([1,2,3,4,5,6])).all())
        self.assertTrue((self.voc1.id2vec(2) == np.array([0,1,2,3,4,5])).all())
        self.assertTrue((self.voc1.id2vec(6) == np.array([0,0,0,0,0,1])).all())
        self.assertTrue((self.voc1.id2vec(0) == np.array([0,0,0,0,0,0])).all())

    def test_get_lookup_tensor(self):
        with tf.Session() as sess:
            self.assertTrue((sess.run(self.voc1.get_lookup_tensor()) == self.vectors).all())

    def test_one_hot_to_id(self):
        self.assertEqual(self.voc1.one_hot_to_id(np.array([0,0,1,0])), 2)
        self.assertEqual(self.voc1.one_hot_to_id(np.array([0,0.75,0.25,0])), 1)
        self.assertEqual(self.voc1.one_hot_to_id(np.array([0.51,0.0,0.25,0.24])), 0)

    def test_create_vocabulary(self):
        voc2 = Vocabulary.create_vocabulary(["a", "b", "a", "c"], 0, create_one_hot_embeddings=True)
        vectors2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        with tf.Session() as sess:
            self.assertTrue((sess.run(voc2.get_lookup_tensor()) == vectors2).all())
        self.assertTrue((voc2.word2vec("newword") == np.array([1, 0, 0, 0])).all())

if __name__ == '__main__':
    main()
