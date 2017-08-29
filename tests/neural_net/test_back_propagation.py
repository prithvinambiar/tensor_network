import tempfile

from tensor_network.neural_net import back_propagation


def test_train_neural_net_for_or_gate():
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_data = [[0], [1], [1], [1]]

    with back_propagation.BackPropagation(2, 1, [10, 5, 1]) as bp:
        bp.train((input_data, output_data), frequency=10, folder=tempfile.gettempdir() + "/tensorflow/folder/1")
        assert bp.predict([[0, 0]]) < [[0.25]]
        assert bp.predict([[1, 0]]) > [[0.75]]
        assert bp.predict([[0, 1]]) > [[0.75]]
        assert bp.predict([[1, 1]]) > [[0.75]]
        assert bp.validation(input_data, output_data) < 0.10
