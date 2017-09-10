import tempfile

from tensor_network.neural_net import fully_connected
import tensorflow as tf
import numpy as np


def test_train_neural_net_for_or_gate():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [1], [1], [1]])

    with fully_connected.FullyConnected(2, 1, [100, 100, 1]) as fc:
        learning_rate = tf.train.exponential_decay(0.1, fc.global_step,
                                                   50, 0.96, staircase=True)
        fc.train((input_data, output_data), folder=tempfile.gettempdir() + "/tensorflow/folder/1"
                 , optimiser=tf.train.AdamOptimizer(learning_rate=learning_rate), iterations=10000)

        assert fc.predict([[0, 0]]) < [[0.25]]
        assert fc.predict([[1, 0]]) > [[0.75]]
        assert fc.predict([[0, 1]]) > [[0.75]]
        assert fc.predict([[1, 1]]) > [[0.75]]
        assert fc.cost(input_data, output_data) < 0.10
