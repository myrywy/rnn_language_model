from unittest import TestCase, main
import numpy as np
import tensorflow as tf
import network
import vocabulary
import input_data as in_da
from ptb_reader import read_ptb


fullcheck = False


class RnnLmTestCase(TestCase):
    def tearDown(self):
        tf.reset_default_graph()

    def test_get_sentence(self):
        with tf.variable_scope("test_get_sentence"):
            voc, (tr, val, tst) = read_ptb()
            net = network.RnnLm(tr.dataset(), val.dataset(), tst.dataset(), network.BasicConfig, voc)

            sentence, iter = net.get_sentence(net.train_set)
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                sess.run(iter.initializer)
                # Start populating the filename queue.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                try:
                    #while not coord.should_stop():
                    batch_lens, batch_ids, batch_vecs = sess.run(sentence)
                    with open("ptb/data/ptb.train.txt") as train_file:
                        for line, (ids, length, vecs) in zip(train_file, zip(batch_ids, batch_lens, batch_vecs)):
                            from_net_sent = [voc.id2word(i) for i in ids][:length]
                            if fullcheck:
                                vecs_from_id = [voc.id2vec(i) for i in ids][:length]
                                vecs_from_net = vecs[:length]
                                assert (vecs_from_id == vecs_from_net).all()
                            from_file_sent = line.lower().split() + ["</snt>"]
                            assert len(from_net_sent) == len(from_file_sent)
                            assert len(set(from_net_sent).difference({"<UNKNOWN>"}))    # check if there are any non-UNKNOW words in a sentence
                            for w1, w2 in zip(from_net_sent, from_file_sent):
                                assert w1 == w2 or w1 == "<UNKNOWN>"    # check that all that are not UNKNONW are the same

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    coord.request_stop()

                coord.join(threads)

    def test_cost(self):
        with tf.variable_scope("test_cost"):
            vec = np.array([
                [0, 0, 0],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
            ])
            '''voc = vocabulary.Vocabulary(["a", "b", "c"], [1, 2, 3], vec)
            with tf.variable_scope("train"):
                tr = in_da.InputData(voc, [["a", "b", "c"]])
                tr_set = tr.dataset()
            with tf.variable_scope("validate"):
                val = in_da.InputData(voc, [["a", "b", "c"]])
                val_set = val.dataset()
            with tf.variable_scope("test"):
                tst = in_da.InputData(voc, [["a", "b", "c"]])
                tst_set = tst.dataset()'''
            conf = network.BasicConfig()
            #voc, (tr, val, tst) = read_ptb()
            #net = network.RnnLm(tr.dataset(), val.dataset(), tst.dataset(), network.BasicConfig, voc)
            conf.batch_size = 1
            voc, (tr_set, val_set, tst_set) = in_da.InputData.prepeare_one_hot_input([["a","b","c"]], [["a","b","c"]], [["a","b","c"]], min_word_count=0)
            net = network.RnnLm(tr_set.dataset(), val_set.dataset(), tst_set.dataset(), conf, voc)

            '''
            expected = tf.constant([[1, 2, 3]], dtype=tf.int64) # as ids
            #results = tf.constant([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=tf.float32) # as onehots
            #results = tf.constant([[[0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]], dtype=tf.float32) # as onehots
            results = tf.constant([[[0.0, 0.25, 0.25, 0.25], [0.0, 0.25, 0.25, 0.25], [0.0, 0.25, 0.25, 0.25]]], dtype=tf.float32) # as onehots
            lengths = tf.constant([3], dtype=tf.int64)
            '''
            expected1 = tf.constant([[1]], dtype=tf.int64) # as ids
            results1 = tf.constant([[[0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32) # as onehots
            lengths1 = tf.constant([1], dtype=tf.int64)
            expected2 = tf.constant([[1, 1]], dtype=tf.int64) # as ids
            results2 = tf.constant([[[0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 10.0, 0.0]]], dtype=tf.float32) # as onehots
            lengths2 = tf.constant([1], dtype=tf.int64)
            expected3 = tf.constant([[1, 1]], dtype=tf.int64) # as ids
            results3 = tf.constant([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32) # as onehots
            lengths3 = tf.constant([2], dtype=tf.int64)

            net.config.batch_size = 1
            cost1 = net.cost(lengths1, expected1, results1)
            cost2 = net.cost(lengths2, expected2, results2)
            cost3 = net.cost(lengths3, expected3, results3)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                self.assertEqual(sess.run(cost1), sess.run(cost2))
                self.assertEqual(np.exp(sess.run(cost1)), 4)
                self.assertEqual(np.exp(sess.run(cost3)), 16)

    def test_mask(self):
        with tf.variable_scope("test_cost"):
            conf = network.BasicConfig()
            # voc, (tr, val, tst) = read_ptb()
            # net = network.RnnLm(tr.dataset(), val.dataset(), tst.dataset(), network.BasicConfig, voc)
            conf.batch_size = 1
            voc, (tr_set, val_set, tst_set) = in_da.InputData.prepeare_one_hot_input([["a", "b", "c"]],
                                                                                     [["a", "b", "c"]],
                                                                                     [["a", "b", "c"]],
                                                                                     min_word_count=0)
            net = network.RnnLm(tr_set.dataset(), val_set.dataset(), tst_set.dataset(), conf, voc)
            lengths1 = tf.constant([1], dtype=tf.int64)
            expected1 = tf.constant([[1, 1]], dtype=tf.int64) # as ids
            mask1 = net.mask(tf.shape(expected1), lengths1)
            lengths2 = tf.constant([2], dtype=tf.int64)
            expected2 = tf.constant([[1, 1]], dtype=tf.int64) # as ids
            mask2 = net.mask(tf.shape(expected2), lengths2)
            lengths3 = tf.constant([2, 1], dtype=tf.int64)
            expected3 = tf.constant([[1, 1], [1, 1]], dtype=tf.int64) # as ids
            mask3 = net.mask(tf.shape(expected3), lengths3)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                self.assertTrue((sess.run(mask1) == np.array([[1.0, 0.0]])).all())
                self.assertTrue((sess.run(mask2) == np.array([[1.0, 1.0]])).all())
                self.assertTrue((sess.run(mask3) == np.array([[1.0, 1.0], [1.0, 0.0]])).all())


if __name__ == '__main__':
    main()