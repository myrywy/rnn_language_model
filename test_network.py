import tensorflow as tf
import network
import vocabulary
from ptb_reader import read_ptb

voc, _ = read_ptb()

net = network.RnnLm("train.pb", "validate.pb", "test.pb", network.BasicConfig, voc)

readed = net.batched_input(net.train_data)
read_value = net.get_single_example(net.value_from_file("train.pb"))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        #while not coord.should_stop():
        # Run training steps or whatever
        r = sess.run([readed,read_value])

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    '''
    #import pdb;pdb.set_trace()
    for i in range(1):
        # Retrieve a single instance:
        print(sess.run())
    coord.request_stop()'''
    coord.join(threads)

'''
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(sess.run(net.batched_input(net.train_data)))
    coord.request_stop()
    coord.join(threads)
'''