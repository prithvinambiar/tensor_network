import logging
import tempfile
import numpy as np
import tensorflow as tf


def initialize_network(input, arch, is_training):
    hidden_layer_output = input
    for output in arch[0:-1]:
        h2 = tf.contrib.layers.batch_norm(hidden_layer_output,
                                          center=True, scale=True,
                                          is_training=is_training)
        hidden_layer_output = tf.contrib.layers.fully_connected(h2, output, activation_fn=tf.nn.relu
                                                                , weights_regularizer=tf.contrib.layers.l2_regularizer)

    last_layer_output = tf.contrib.layers.fully_connected(hidden_layer_output, arch[-1], activation_fn=tf.nn.sigmoid
                                                          , weights_regularizer=tf.contrib.layers.l2_regularizer)
    return last_layer_output


def log_loss(actual_y, predicted_y):
    return tf.reduce_mean(-tf.reduce_sum(
        (actual_y * tf.log(predicted_y + 1e-10) + ((1 - actual_y) * tf.log(1 - predicted_y + 1e-10))), axis=[1]))


class FullyConnected:
    def __init__(self, number_of_features, number_of_output, neurons_list):
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, number_of_features])
        self.y = tf.placeholder(tf.float32, [None, number_of_output])
        self.is_training = tf.placeholder(tf.bool)
        self.network = initialize_network(self.x, neurons_list, self.is_training)
        self.cost_function = log_loss(self.y, self.network)
        self.global_step = tf.Variable(0, trainable=False)

    def __del__(self):
        self.session.close()

    def __enter__(self):
        self.session.__enter__()
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        self.session.__exit__(exec_type, exec_value, exec_tb)
        tf.reset_default_graph()

    def train(self, train_data, validation_data=None, iterations=10000,
              optimiser=tf.train.GradientDescentOptimizer(learning_rate=0.05), import_prev_model=False,
              model_save_frequency=0, log_frequency=10, folder=tempfile.gettempdir() + "/tensorflow",
              reg_lambda=0.0001, batch_size_pct=0.2):
        (train_input, train_output) = train_data
        (validation_input, validation_output) = train_data if validation_data is None else validation_data
        tensorflow_dir = folder
        log_dir = tensorflow_dir + "/log"
        model_file = tensorflow_dir + "/model/model_data"
        logging.info("Logging TensorFlow data to %s " % log_dir)
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(self.session.graph)
        with tf.name_scope("train"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = optimiser.minimize(self.cost_function, name="train_step", global_step=self.global_step)
        saver = tf.train.Saver(max_to_keep=1)

        if import_prev_model:
            saver.restore(self.session, model_file)
        else:
            self.session.run(tf.global_variables_initializer())

        tf.summary.scalar('cost', self.cost_function)
        merged_summary = tf.summary.merge_all()

        if len(train_input) * .2 < 1:
            batch_size = len(train_input)
        else:
            batch_size = int(len(train_input) * batch_size_pct)

        print("Number of rows in train data = ", len(train_input))
        print("Number of rows in batch data = ", batch_size)

        for i in range(iterations):
            j = i + 1
            indices = np.random.choice(range(len(train_input)), size=batch_size, replace=False)
            batch_input = train_input[indices]
            batch_output = train_output[indices]

            if (log_frequency != 0 and j % log_frequency == 0) or j == 1:
                validation_accuracy = self.cost(validation_input, validation_output)
                train_accuracy = self.cost(train_input, train_output)
                cost = self.session.run(self.cost_function,
                                        feed_dict={self.x: batch_input, self.y: batch_output, self.is_training: 0})
                print("Iterations = %s and Cost = %s and Train accuracy = %s and Validation accuracy = %s" % (
                    j, cost, train_accuracy, validation_accuracy))

            self.session.run(train_step,
                             feed_dict={self.x: batch_input, self.y: batch_output, self.is_training: 1})

            if model_save_frequency != 0 and j % model_save_frequency == 0:
                summary = self.session.run(merged_summary, feed_dict={self.x: train_input, self.y: train_output, self.is_training: 1})
                writer.add_summary(summary, j)
                print("Saving model")
                saver.save(self.session, model_file)

    def predict(self, test_input_data):
        return self.session.run(self.network, feed_dict={self.x: test_input_data, self.is_training: 0})

    def cost(self, input_data, output_data):
        with tf.name_scope("validation"):
            return self.session.run(self.cost_function,
                                    feed_dict={self.y: output_data, self.x: input_data, self.is_training: 0})
