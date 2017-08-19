import tensorflow as tf


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
        res = tf.nn.sigmoid(tf.matmul(res, w) + b)
    return res


class BackPropagation:
    def __init__(self, number_of_features, number_of_output, neurons_list):
        self.__session = tf.Session()
        self.__x = tf.placeholder(tf.float32, [None, number_of_features])
        self.__y = tf.placeholder(tf.float32, [None, number_of_output])
        self.__network = list(tensor_network(number_of_features, neurons_list))
        self.model = evaluate_network(self.__network, self.__x)

    def __del__(self):
        self.__session.close()

    def train(self, train_input_data, train_output_data, iterations=10000):
        self.__session.run(tf.global_variables_initializer())
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum((self.__y * tf.log(self.model) + ((1 - self.__y) * tf.log(1 - self.model))), axis=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
        for i in range(iterations):
            if i % 1000 == 0:
                print(self.__session.run(cross_entropy,
                                         feed_dict={self.__x: train_input_data, self.__y: train_output_data}))
                for j in self.__network:
                    print(self.__session.run(j))
            self.__session.run(train_step, feed_dict={self.__x: train_input_data, self.__y: train_output_data})

    def predict(self, test_input_data):
        return self.__session.run(self.model, feed_dict={self.__x: test_input_data})
