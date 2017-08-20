from tensor_network.neural_net import back_propagation


def test_train_neural_net_for_or_gate():
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_data = [[0], [1], [1], [1]]

    bp = back_propagation.BackPropagation(2, 1, [1])
    bp.train(input_data, output_data)
    assert bp.predict([[0, 0]]) < [[0.25]]
    assert bp.predict([[1, 0]]) > [[0.75]]
    assert bp.predict([[0, 1]]) > [[0.75]]
    assert bp.predict([[1, 1]]) > [[0.75]]


def test_validation_log_loss_should_be_below_10():
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_data = [[0], [1], [1], [1]]

    bp = back_propagation.BackPropagation(2, 1, [1])
    bp.train(input_data, output_data)
    print(bp.validation(input_data, output_data))
    assert bp.validation(input_data, output_data) < 0.10
