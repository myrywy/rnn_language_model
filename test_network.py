import tensorflow as tf
import network
import vocabulary
from ptb_reader import read_ptb


voc, (tr, val, tst) = read_ptb()
net = network.RnnLm(tr.dataset(), val.dataset(), tst.dataset(), network.BasicConfig, voc) # also temporarly

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
            for line, (ids, length) in zip(train_file, zip(batch_ids, batch_lens)):
                from_net_sent = [voc.id2word(i) for i in ids][:length]
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
