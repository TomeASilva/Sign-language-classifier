import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


# Flatten the trainning and test images

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten/255
X_test = X_test_flatten/255
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# print("number of training examples = " + str(X_train.shape[1]))
# print("number of test examples = " + str(X_test.shape[1]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))


def create_placeholders(n_x, n_y):

    X = tf.placeholder(shape=[n_x, None], dtype=tf.float32, name='features')
    Y = tf.placeholder(shape=[n_y, None], dtype=tf.float32, name='labels')

    return X, Y


def initialize_parameters():

    w1 = tf.get_variable(
        "w1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    w2 = tf.get_variable(
        "w2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    w3 = tf.get_variable(
        "w3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    b2 = tf.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())
    b3 = tf.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())

    parameters = {"w1": w1,
                  'w2': w2,
                  'w3': w3,
                  'b1': b1,
                  'b2': b2,
                  'b3': b3}

    return parameters


def forward_propagation(X, parameters):

    activations = [tf.nn.relu, tf.nn.relu, None]
    weights = []
    biases = []
    layer = X
    for key, value in parameters.items():
        if key[0] == 'w':
            weights.append(value)
        else:
            biases.append(value)

    for weight, bias, activation in zip(weights, biases, activations):
        layer = tf.matmul(weight, layer) + bias
        if activation is not None:
            layer = activation(layer)

    z3 = layer

    return layer


def compute_cost(Z3, Y):

    Z3 = tf.transpose(Z3)
    Y = tf.transpose(Y)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=21000, minibatch_size=32, print_cost=True):

    costs = []

    tf.reset_default_graph()
    tf.set_random_seed(1)

    features, labels = create_placeholders(X_train.shape[0], Y_train.shape[0])
    parameters = initialize_parameters()

    output_pred = forward_propagation(features, parameters)

    cost = compute_cost(output_pred, labels)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_epochs):
            indices = np.random.randint(
                0, X_train.shape[1]-1, size=minibatch_size)
            minibatch = X_train[:, indices]
            output = Y_train[:, indices]

            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={
                                         features: minibatch, labels: output})

            if print_cost == True and i % 1000 == 0:
                print("Cost at iteration {} is {}".format(
                    i, minibatch_cost))
            if i % 100 == 0:
                costs.append(minibatch_cost)

        fig, ax = plt.subplots()

        ax.plot(costs)
        ax.set_ylabel("cost")
        ax.set_xlabel("Iteration per (100)")
        ax.set_title("Learning rate" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(
            tf.argmax(output_pred), tf.argmax(labels))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train, Accuracy:", accuracy.eval(
            {features: X_train, labels: Y_train}))
        print("Test, Accuracy:", accuracy.eval(
            {features: X_test, labels: Y_test}))

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test)
