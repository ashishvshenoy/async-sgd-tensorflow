import tensorflow as tf
import os

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

g = tf.Graph()

with g.as_default():

    # creating a model variable on task 0. This is a process running on node vm-48-1
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([10, 1]), name="model")

    # creating only reader and gradient computation operator
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        reader = tf.ones([10, 1], name="operator_%d" % FLAGS.task_index)
        # not the gradient compuation here is a random operation. You need
        # to use the right way (as described in assignment 3 desc).
        # we use this specific example to show that gradient computation
        # requires use of the model
        local_gradient = tf.mul(reader, tf.matmul(tf.transpose(w), reader))

    with tf.device("/job:worker/task:0"):
        assign_op = w.assign_add(tf.mul(local_gradient, 0.001))


    with tf.Session("grpc://vm-4-%d:2222" % (FLAGS.task_index+1)) as sess:

        # only one client initializes the variable
        if FLAGS.task_index == 0:
            sess.run(tf.initialize_all_variables())
        for i in range(0, 1000):
            sess.run(assign_op)
            print w.eval()
        sess.close()
