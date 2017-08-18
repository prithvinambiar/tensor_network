import tensorflow as tf
import numpy as np


def initialize_weight(input_count, output_count):
    return tf.Variable(tf.truncated_normal([input_count, output_count]))


def initialize_bias(output):
    return tf.Variable(tf.truncated_normal([output]))


def initialize_network(input_count, list_of_neurons):
    list1 = [input_count] + list_of_neurons
    # list2 = list_of_neurons + [output_count]
    list2 = list_of_neurons

    def func(d): return initialize_weight(d[0], d[1]), initialize_bias(d[1])
    return map(func, list(zip(list1, list2)))


def tensor_network(first_layer_input, list_of_neurons):
    return initialize_network(first_layer_input, list_of_neurons)


def evaluate_network(network, input_data):
    res = input_data
    for _tuple in network:
        w = _tuple[0]
        b = _tuple[1]
        res = tf.nn.softmax(tf.matmul(res, w) + b)
    return res


def run_feed_forward_neural_net(train_input_data, train_output_data, neurons_list):
    (_, number_of_features) = np.shape(train_input_data)
    (_, number_of_output) = np.shape(train_output_data)
    x = tf.placeholder(tf.float32, [None, number_of_features])
    y_ = tf.placeholder(tf.float32, [None, number_of_output])
    network = list(tensor_network(number_of_features, neurons_list))
    y = tf.nn.sigmoid(evaluate_network(network, x))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        weights = map(lambda t: t[0], network)
        bias = map(lambda t: t[1], network)
        sess.run(tf.variables_initializer(weights))
        sess.run(tf.variables_initializer(bias))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
        sess.run(train_step, feed_dict={x: train_input_data, y_: train_output_data})
