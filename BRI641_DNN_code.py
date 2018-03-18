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

X_data = m["data_1d_x"] # Input data
Y_data = m["data_y"]    # labels X_data.shape
print("Input data shape:", X_data.shape ,"/////" ,"Data label shape: ", Y_data.shape)


## model
reset_graph()

# Hyperparameter

n_inputs = 74484     # input data variables node
n_hidden1 = 300      # hidden layer 1 node
n_hidden2 = 100      # hidden layer 2 node
n_outputs = 4        # output node: output class is 4 (LH / RH / AD / VS)

learning_rate = 0.01 # dropout rate
dropout_rate = 0.5   # dropout rate


n_epochs = 20        # epoch
batch_size = 30      # batch size

# input, output placeholder setting
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# Dropout
training = tf.placeholder_with_default(False, shape=(), name='training')
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

# Max-Norm Regularization
max_norm_reg = max_norm_regularizer(threshold=1.0)

# Deep neural network design
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, name="hidden1",
                              activation=tf.nn.relu, kernel_regularizer=max_norm_reg)
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2",
                              activation=tf.nn.relu, kernel_regularizer=max_norm_reg)
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # Computes sparse softmax cross entropy between logits and labels
    loss = tf.reduce_mean(xentropy, name="loss")
    # Computes the mean of elements across "xentropy"

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # I implement Gradient Descent Optimizer
    training_op = optimizer.minimize(loss)
    # minimize loss to train

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # Computes accuracy

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# run the weights clipping operations after each training operation
clip_all_weights = tf.get_collection("max_norm")

# Leave one out cross validation - group making
groups = []
for i in range(1, 13):
    group = [i] * 120
    for i in group:
        groups.append(i)
logo = LeaveOneGroupOut()
logo.get_n_splits(X_data, Y_data, groups)

looop = []
times = 1
for train_index, test_index in logo.split(X_data, Y_data, groups):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = Y_data[train_index], Y_data[test_index]
    # LOOCV by indexing

    with tf.Session() as sess:
        init.run()
        accuracy_test = []
        for epoch in range(n_epochs):
            # Slicing train data by batch size(30)
            i = 0
            for batch in range(len(X_train) // batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                i += batch_size
                # Training the weights, biases - Deep Neural Networks
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch.reshape(-1)})
            # Computes accuracy training set
            acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train.reshape(-1)})
            # Computes accuracy test set
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test.reshape(-1)})
            # append the test set accuracy in epoch
            accuracy_test.append(acc_test)
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        # append the biggets test set accuracy in each epoch
        looop.append(max(accuracy_test))
        save_path = saver.save(sess, "./my_model_final.ckpt")
    print(times, " times cross validation End")
    times += 1
    print("*" * 30)
print("=" * 70)
print("Final DNN test accuracy result each cross validation")
print(looop)
print("LOOCV Test accuracy mean in DNN: ", np.mean(looop))