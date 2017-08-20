import logging
import tempfile

import tensorflow as tf


def initialize_weight(input_count, output_count):
    return tf.Variable(tf.truncated_normal([input_count, output_count]), name='W')


def initialize_bias(output):
    return tf.Variable(tf.truncated_normal([output]), name='B')


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


class BackPropagation:
    def __init__(self, number_of_features, number_of_output, neurons_list):
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, number_of_features])
        self.y = tf.placeholder(tf.float32, [None, number_of_output])
        self.network = list(tensor_network(number_of_features, neurons_list))
        self.model = evaluate_network(self.network, self.x)
        with tf.name_scope("cost"):
            self.cost_function = tf.reduce_mean(
                -tf.reduce_sum((self.y * tf.log(self.model + 1e-10) + ((1 - self.y) * tf.log(1 - self.model + 1e-10))),
                               axis=[1]))

    def __del__(self):
        self.session.close()

    def train(self, train_input_data, train_output_data, iterations=10000,
              optimiser=tf.train.GradientDescentOptimizer(learning_rate=0.05)):
        temp_dir = tempfile.gettempdir() + "/tensorflow"
        logging.info("Logging TensorFlow data to %s " % temp_dir)
        writer = tf.summary.FileWriter(temp_dir)
        writer.add_graph(self.session.graph)
        tf.summary.scalar('cost', self.cost_function)
        merged_summary = tf.summary.merge_all()

        with tf.name_scope("train"):
            train_step = optimiser.minimize(self.cost_function)
        self.session.run(tf.global_variables_initializer())

        for i in range(iterations):
            if i % 10 == 0:
                print("Iterations = ", i, " and Cost = ", self.session.run(self.cost_function,
                                                                           feed_dict={self.x: train_input_data,
                                                                                      self.y: train_output_data}))
                # for j in self.network:
                #     print(self.session.run(j))

            self.session.run(train_step, feed_dict={self.x: train_input_data, self.y: train_output_data})
            s = self.session.run(merged_summary, feed_dict={self.x: train_input_data, self.y: train_output_data})
            writer.add_summary(s, i)

    def predict(self, test_input_data):
        return self.session.run(self.model, feed_dict={self.x: test_input_data})

    def validation(self, validation_input_data, validation_output_data):
        with tf.name_scope("validation"):
            predictions = self.predict(validation_input_data)
            return self.session.run(self.cost_function,
                                    feed_dict={self.y: validation_output_data, self.model: predictions})
