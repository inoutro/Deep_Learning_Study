import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error


def reset_graph():
    tf.reset_default_graph()


def max_norm_regularizer(threshold, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None  # there is no regularization loss term

    return max_norm


### Input Data
file_name = "BRI641_fMRI_data_new_sets.mat"
m = loadmat(file_name)
# print(m)
# data exploration
# I decide to Use"data_1d_x" for data analysis

X_data = m["data_1d_x"] # Input data
Y_data = m["data_y"]    # labels X_data.shape
print("Input data shape:", X_data.shape ,"/////" ,"Data label shape: ", Y_data.shape)



### model
reset_graph()

# Data set is 1-dimension, so i use 1-dimension CNN
height = 1
width = 74484
channels = 1
n_inputs = height * width


# First Convolution layer information
conv1_fmaps = 32   # Filter size
conv1_ksize = 10   # Number of filter
conv1_stride = 5   # Stride
conv1_pad = "SAME"

# Second Convolution layer information
conv2_fmaps = 64   # Filter size
conv2_ksize = 10   # Number of filter
conv2_stride = 5   # Stride
conv2_pad = "SAME"

# Third Convolution layer information(+ dropout rate)
conv3_fmaps = 64   # Filter size
conv3_ksize = 10   # Number of filter
conv3_stride = 5   # Stride
conv3_pad = "SAME"
conv3_dropout_rate = 0.5

# In 2nd pooling layer, to make flat layer
pool2_fmaps = conv3_fmaps

# Neural Network layer information(+ dropout rate)
n_fc1 = 128
fc1_dropout_rate = 0.5

# output node: output class is 4 (LH / RH / AD / VS)
n_outputs = 4


with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=(None,74484), name="X")
    # Conv1d function's shape is required [batch, in_width, in_channels]
    # so, reshape the input X like [-1, width, channels]
    X_reshaped = tf.reshape(X, shape=[-1, width, channels])
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    # To use dropout, make this training part
    training = tf.placeholder_with_default(False, shape=[], name='training')

# Convolutional neural network design
# activation function is relu

conv1 = tf.layers.conv1d(X_reshaped,
                         filters=conv1_fmaps,
                         kernel_size=conv1_ksize,
                         strides=conv1_stride,
                         padding=conv1_pad,
                         activation=tf.nn.relu,
                         name="conv1")

conv2 = tf.layers.conv1d(conv1,
                         filters=conv2_fmaps,
                         kernel_size=conv2_ksize,
                         strides=conv2_stride,
                         padding=conv2_pad,
                         activation=tf.nn.relu,
                         name="conv2")

with tf.name_scope("pool1"):
    pool1 = tf.layers.max_pooling1d(conv2,
                                    pool_size=2,
                                    strides=2,
                                    padding="VALID")

conv3 = tf.layers.conv1d(pool1,
                         filters=conv3_fmaps,
                         kernel_size=conv3_ksize,
                         strides=conv3_stride,
                         padding=conv3_pad,
                         activation=tf.nn.relu,
                         name="conv3")

with tf.name_scope("pool2"):
    pool2 = tf.layers.max_pooling1d(conv3,
                                    pool_size=2,
                                    strides=2,
                                    padding="VALID")
    # pool2 layer form is (?, 149, 64) so, pool2_fmaps(64)*149 to flatten
    pool2_flat = tf.reshape(pool2,
                            shape=[-1, pool2_fmaps * 149])
    pool2_flat_drop = tf.layers.dropout(pool2_flat,
                                        conv3_dropout_rate,
                                        training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool2_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    # Computes sparse softmax cross entropy between logits and labels
    loss = tf.reduce_mean(xentropy)
    # Computes the mean of elements across "xentropy"
    optimizer = tf.train.AdamOptimizer()
    # I implement Adam Optimizer
    training_op = optimizer.minimize(loss)
    # minimize loss to train

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # Computes accuracy

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

## This is same with Deep Neural Network session part

# Same with DNN part
n_epochs = 20
batch_size = 30

# Leave one out cross validation - group making
groups = []
for i in range(1,13):
    group = [i] * 120
    for i in group:
        groups.append(i)
logo = LeaveOneGroupOut()
logo.get_n_splits(X_data, Y_data, groups)

looop = []
times = 1
for train_index, test_index in logo.split(X_data, Y_data, groups):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = Y_data[train_index], Y_data[test_index]
    with tf.Session() as sess:
        init.run()
        accuracy_test = []
        for epoch in range(n_epochs):
            i = 0
            for batch in range(len(X_train) // batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                i += batch_size
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch.reshape(-1)})
            acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train.reshape(-1)})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test.reshape(-1)})
            accuracy_test.append(acc_test)
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        looop.append(max(accuracy_test))
        save_path = saver.save(sess, "./my_CNN_model_final.ckpt")
    print(times," times cross validation End")
    times += 1
    print("*"*30)
print("="*70)
print("Final CNN test accuracy result each cross validation")
print(looop)
print("LOOCV Test accuracy mean in CNN: ", np.mean(looop))
