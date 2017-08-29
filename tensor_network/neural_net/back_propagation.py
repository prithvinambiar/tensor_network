import logging
import tempfile

import tensorflow as tf


def initialize_weight(input_count, output_count):
    return tf.Variable(tf.truncated_normal([input_count, output_count]), name='W')


def initialize_bias(output):
    return tf.Variable(tf.zeros([output]), name='B')


def initialize_layer(layer_info, name='layer'):
    with tf.name_scope(name):
        return initialize_weight(layer_info[0], layer_info[1]), initialize_bias(layer_info[1])


def initialize_network(input_count, list_of_neurons):
    list1 = [input_count] + list_of_neurons
    # list2 = list_of_neurons + [output_count]
    list2 = list_of_neurons

    layers = []
    i = 1
    for layer_info in list(zip(list1, list2)):
        layers.append(initialize_layer(layer_info, name='layer' + str(i)))
        i += 1
    return layers


def tensor_network(first_layer_input, list_of_neurons):
    return initialize_network(first_layer_input, list_of_neurons)


def evaluate_network(network, input_data):
    res = input_data
    for _tuple in network:
        w = _tuple[0]
        b = _tuple[1]
        res = tf.nn.sigmoid(tf.matmul(res, w) + b)
    return res


def cost_function(actual_y, predicted_y, weights, beta):
    l2_regularization = sum(tf.nn.l2_loss(weight) for weight in weights)
    loss = log_loss(actual_y, predicted_y)
    return tf.reduce_mean(loss + beta * l2_regularization)


def log_loss(actual_y, predicted_y):
    return -tf.reduce_sum(
        (actual_y * tf.log(predicted_y + 1e-10) + ((1 - actual_y) * tf.log(1 - predicted_y + 1e-10))), axis=[1])


def accuracy(actual_y, predicted_y):
    loss = log_loss(actual_y, predicted_y)
    return tf.reduce_mean(loss)


def get_weights(network):
    weights = []
    for layer in network:
        weight, _ = layer
        weights.append(weight)
    return weights


class BackPropagation:
    def __init__(self, number_of_features, number_of_output, neurons_list):
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, number_of_features])
        self.y = tf.placeholder(tf.float32, [None, number_of_output])
        self.beta = tf.placeholder(tf.float32)
        self.validation_x = tf.placeholder(tf.float32, [None, number_of_features])
        self.validation_y = tf.placeholder(tf.float32, [None, number_of_output])
        self.network = list(tensor_network(number_of_features, neurons_list))
        self.model = evaluate_network(self.network, self.x)
        self.validation_y_pred = evaluate_network(self.network, self.validation_x)
        self.cost_function = cost_function(self.y, self.model, get_weights(self.network), self.beta)
        self.accuracy = accuracy(self.validation_y, self.validation_y_pred)

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
              frequency=10, folder=tempfile.gettempdir() + "/tensorflow", reg_lambda=0.0001):
        (train_input, train_output) = train_data
        (validation_input, validation_output) = train_data if validation_data is None else validation_data
        tensorflow_dir = folder
        log_dir = tensorflow_dir + "/log"
        model_file = tensorflow_dir + "/model/model_data"
        logging.info("Logging TensorFlow data to %s " % log_dir)
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(self.session.graph)
        tf.summary.scalar('cost', self.cost_function)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary = tf.summary.merge_all()
        with tf.name_scope("train"):
            train_step = optimiser.minimize(self.cost_function, name="train_step")
        saver = tf.train.Saver(max_to_keep=1)

        if import_prev_model:
            saver.restore(self.session, model_file)
        else:
            self.session.run(tf.global_variables_initializer())

        for i in range(iterations):
            if frequency != 0 and i % (iterations / frequency) == 0:
                accuracy = self.validation(validation_input, validation_output)
                cost = self.session.run(self.cost_function,
                                        feed_dict={self.x: train_input, self.y: train_output, self.beta: reg_lambda})
                print("Iterations = %s and Cost = %s and accuracy = %s" % (i, cost, accuracy))
                summary = self.session.run(merged_summary, feed_dict={self.x: train_input, self.y: train_output,
                                                                      self.validation_x: validation_input,
                                                                      self.validation_y: validation_output,
                                                                      self.beta: reg_lambda})
                writer.add_summary(summary, i)
                saver.save(self.session, model_file)

            self.session.run(train_step,
                             feed_dict={self.x: train_input, self.y: train_output, self.validation_x: validation_input,
                                        self.validation_y: validation_output, self.beta: reg_lambda})

    def predict(self, test_input_data):
        return self.session.run(self.model, feed_dict={self.x: test_input_data})

    def validation(self, validation_input_data, validation_output_data):
        with tf.name_scope("validation"):
            predictions = self.predict(validation_input_data)
            return self.session.run(self.accuracy,
                                    feed_dict={self.validation_y: validation_output_data,
                                               self.validation_y_pred: predictions})
