import tensorflow as tf
import os
import time

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

print "Before creating the graph"

g = tf.Graph()

with g.as_default():

    def read_and_test_single() :
        with tf.device("/job:worker/task:%d" % 0):
            filename_queue = tf.train.string_input_producer([absolute_path+"22"],num_epochs=None)
            reader = tf.TFRecordReader()
            _, serialized_data = reader.read(filename_queue)
            features = tf.parse_single_example(serialized_data,
                                                       features={
                                                            'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                                            'index' : tf.VarLenFeature(dtype=tf.int64),
                                                            'value' : tf.VarLenFeature(dtype=tf.float32),
                                                       }
                                                      )

            label = features['label']
            index = features['index']
            value = features['value']

            dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),[num_features,],tf.sparse_tensor_to_dense(value))

            predictions = tf.reduce_sum(tf.mul(w,dense_feature))
            signs = tf.sign(predictions)
            signs = tf.cast(signs,tf.int64)
            ones = tf.constant([1], dtype=tf.int64)
            temp =  tf.sub(ones,tf.mul(signs,label))
            temp = tf.cast(temp, tf.float32)
            this_is_error = tf.mul(0.5,temp)
        return this_is_error


    with tf.device("/job:worker/task:0"):
        num_features = 33762578
        w = tf.Variable(tf.ones([num_features]), name="model")

    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        num_features = 33762578
        absolute_path = "/home/ubuntu/big_data/tfrecords"
        filepaths = [["00","01","02","03","04"],["05","06","07","08","09"],["10","11","12","13","14"],["15","16","17","18","19"],["20","21"]]
        filename_queue = tf.train.string_input_producer([absolute_path+path for path in filepaths[FLAGS.task_index]],num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_data = reader.read(filename_queue, name='reader_%d'%FLAGS.task_index)
        feature = tf.parse_single_example(serialized_data,
                                                   features={
                                                        'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                                        'index' : tf.VarLenFeature(dtype=tf.int64),
                                                        'value' : tf.VarLenFeature(dtype=tf.float32),
                                                   }
                                                  )

        label = feature['label']
        index = feature['index']
        value = feature['value']
        num_features = 33762578
        eta = -0.01
        filtered_w = tf.gather(w, feature['index'].values)
        p = tf.reduce_sum(tf.mul(filtered_w, feature['value'].values))
        label = tf.cast(label, tf.float32)
        q = tf.mul(label,p)
        s = tf.sigmoid(q)-1
        #r = tf.mul(s,feature['value'].values)
        r = tf.mul(label, feature['value'].values)

        filtered_local_gradient = tf.mul(s,r)
        #filtered_local_gradient = tf.mul(filtered_local_gradient, eta)

        sparse_filtered_local_gradient = tf.SparseTensor(shape=[num_features], indices=[feature['index'].values], values=filtered_local_gradient)

    with tf.device("/job:worker/task:0"):
        #assign_op = tf.scatter_add(w, tf.reshape(sparse_filtered_local_gradient.indices, [-1]) ,  tf.reshape(sparse_filtered_local_gradient.values, [-1]))
        dense_gradient = tf.sparse_to_dense(tf.transpose(sparse_filtered_local_gradient.indices),[num_features],sparse_filtered_local_gradient.values )
        assign_op = w.assign_add(tf.mul(dense_gradient, eta))
        tested_error = read_and_test_single()


    with tf.Session("grpc://vm-4-%d:2222" % (FLAGS.task_index+1)) as sess:
        print "Starting the computation now"
        if FLAGS.task_index == 0:
            sess.run(tf.initialize_all_variables())
        print "Initializing the variables done."
        tf.train.start_queue_runners(sess=sess)
        count = 0
        for i in range(0, 10000):
            if FLAGS.task_index == 0 :
                print "This is zero task hence checking for test"
                if(count!=0 and count%50 == 0 ):
                        errors = []
                        test = 0
                        try:
                            while test!=1000:
                                result = sess.run(tested_error)
                                print tested_error.eval()
                                errors.append(tested_error.eval()[0])
                                test = test +1
                                print "Test Sample : "+str(test)
                                print "Error : "+str(tested_error.eval()[0])
                        except tf.errors.OutOfRangeError :
                            print "EOF in test file"
                        error_rate = ((sum(errors)*1.0)/1000)*100
                        print "Error Rate : "+str(error_rate)
                        with open("error_file", "a+") as error_file :
                            error_file.write("\nError Rate after "+str(count)+" iteration "+str(error_rate))
            count+=1
            print "Samples Read : "+str(count)
            start_time = time.time()
            sess.run(assign_op)
            duration = time.time() - start_time
            print "Duration of assigning : "+str(duration)
            print w.eval()
        sess.close()