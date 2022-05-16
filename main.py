import math
from random import seed
from random import random
import numpy
from sklearn import datasets


def new_neural_network(n_features, n_input, n_output, n_hidden_list, randomize):
    network = list()
    n_hidden_layer = len(n_hidden_list)
    input_layer = [{'weights': [random() if randomize else 0 for _ in range(n_features + 1)]} for _ in range(n_input)]
    network.append(input_layer)
    n_weights = n_input + 1
    for j in range(0, n_hidden_layer):
        n_hidden = n_hidden_list[j]
        hidden_layer = [{'weights': [random() if randomize else 0 for _ in range(n_weights)]} for _ in range(n_hidden)]
        network.append(hidden_layer)
        n_weights = n_hidden + 1
    output_layer = [{'weights': [random() if randomize else 0 for _ in range(n_weights)]} for _ in range(n_output)]
    network.append(output_layer)
    return network


def compute_weighted_sum(sample, weights):
    weighted_sum = weights[len(weights) - 1]
    for i in range(0, len(weights) - 1):
        weighted_sum += weights[i] * sample[i]
    return weighted_sum


def forward_propagate(neural_network, sample, function):
    inputs = sample[:len(sample) - 1]
    for layer in neural_network:
        new_inputs = []
        for neuron in layer:
            weighted_sum = compute_weighted_sum(neuron['weights'], inputs)
            neuron['output'] = apply_activation(weighted_sum, function)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def apply_activation(weighted_sum, function):
    return function(weighted_sum)


def apply_activation_derivative(output, derivative_function):
    return apply_activation(output, derivative_function)


def relu(x):
    return x if x > 0 else 0


def relu_derivative(x):
    return 0 if x < 0 else 1


def backpropagation(network, expected, derivative_function):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['correction'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['correction'] = errors[j] * apply_activation_derivative(neuron['output'], derivative_function)


def train_network(neural_network, training_dataset, learning_rate, decadence,
                  n_max_epoch, function, derivative_function, precision, early_termination):
    min_error = float('inf')
    min_epoch = 0
    for epoch in range(n_max_epoch):
        error = 0
        for sample in training_dataset:
            predicted = forward_propagate(neural_network, sample, function)
            expected = sample[len(sample) - 1]
            error += math.pow((expected - predicted), 2)
            backpropagation(neural_network, expected, derivative_function)
            new_learning_rate = learning_rate / (decadence * (epoch + 1))
            update_weights(neural_network, sample, new_learning_rate)
        error /= (len(training_dataset))
        if epoch == 0:
            min_epoch = epoch + 1
            min_error = error
        else:
            if error < min_error:
                min_epoch = epoch + 1
                min_error = error
        # print('epoch: ' + str(epoch) + ', learning rate: ' + str(round(new_learning_rate, precision))
        #       + ', error: ' + str(round(error, precision)))
        if early_termination and error > min_error:
            # print("min error: " + str(round(min_error, precision)) + ", epoch: " + str(min_epoch))
            break
    # print("min error: " + str(round(min_error, precision)) + ", epoch: " + str(min_epoch))


def update_weights(neural_network, sample, learning_rate):
    for i in range(len(neural_network)):
        if i == 0:
            inputs = sample[0:len(sample) - 1]
        else:
            inputs = [neuron['output'] for neuron in neural_network[i - 1]]
        for neuron in neural_network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['correction'] * inputs[j]
            neuron['weights'][len(inputs)] += learning_rate * neuron['correction']


def predict(neural_network, sample, activation_function):
    predicted = forward_propagate(neural_network, sample, activation_function)
    return predicted


def main():
    iris = datasets.load_iris()
    n_features = 4
    dataset = iris.data[:, :n_features]
    labels = iris.target
    learning_rate = 0.0001
    precision = 2
    decadence = 1 + learning_rate
    n_max_epoch = 1000
    normalized_dataset = dataset.copy()
    n_sample = len(dataset)
    early_termination = True
    normalize = True
    seed(5)
    n_output = 1
    n_input = 5
    n_hidden_list = [5, 5, 5]
    function = relu
    function_derivative = relu_derivative

    # computation of min and max values for each feature
    mins_maxs = []
    for i in range(0, n_features):
        mins_maxs.insert(i, (dataset[:, i].min(), dataset[:, i].max()))
    means = []
    std_devs = []
    for i in range(0, n_features):
        mean = 0
        std_dev = 0
        for sample in dataset:
            mean += sample[i]
        mean /= n_sample
        means.insert(i, mean)
        for sample in dataset:
            std_dev += math.pow(sample[i] - mean, 2)
        std_dev /= n_sample
        std_devs.insert(i, std_dev)

    # dataset normalization
    if normalize is True:
        for i in range(0, n_sample):
            for j in range(0, n_features):
                # z-mean normalization
                normalized_dataset[i][j] -= means[j]
                normalized_dataset[i][j] /= std_devs[j]

    # training and test sets definition
    tmp_training_dataset = numpy.concatenate((numpy.concatenate((normalized_dataset[0:40], normalized_dataset[50:90]),
                                                                axis=0), normalized_dataset[100:140]), axis=0)
    n_training = len(tmp_training_dataset)
    tmp_training_dataset_labels = numpy.concatenate((numpy.concatenate((labels[0:40], labels[50:90]),
                                                                       axis=0), labels[100:140]), axis=0)
    tmp_test_dataset = numpy.concatenate((numpy.concatenate((normalized_dataset[40:50], normalized_dataset[90:100]),
                                                            axis=0), normalized_dataset[140:150]), axis=0)
    n_test = len(tmp_test_dataset)
    tmp_test_dataset_labels = numpy.concatenate((numpy.concatenate((labels[40:50], labels[90:100]),
                                                                   axis=0), labels[140:150]), axis=0)

    training_dataset = []
    for i in range(0, n_training):
        sample = []
        for j in range(0, n_features):
            sample.insert(j, tmp_training_dataset[i][j])
        sample.insert(n_features, tmp_training_dataset_labels[i])
        training_dataset.insert(i, sample)

    test_dataset = []
    for i in range(0, n_test):
        sample = []
        for j in range(0, n_features):
            sample.insert(j, tmp_test_dataset[i][j])
        sample.insert(n_features, tmp_test_dataset_labels[i])
        test_dataset.insert(i, sample)

    network = new_neural_network(n_features, n_input, n_output, n_hidden_list, True)
    train_network(network, training_dataset, learning_rate, decadence, n_max_epoch,
                  function, function_derivative, precision, early_termination)
    # for layer in network:
    #     for neuron in layer:
    #         print("[", end=" ")
    #         for weight in neuron['weights']:
    #             print(round(weight, precision), end=" ")
    #         print("]")

    min_0, min_1, min_2 = float('inf'), float('inf'), float('inf')
    max_0, max_1, max_2 = 0, 0, 0
    mean_0, mean_1, mean_2 = 0, 0, 0
    n_0, n_1, n_2 = 0, 0, 0
    for i in range(0, n_training):
        sample = training_dataset[i]
        expected = training_dataset[i][len(sample) - 1]
        predicted = predict(network, sample, relu)[0]
        # print("expected: " + str(expected) + " predicted: " + str(round(predicted, precision)))
        if expected == 0:
            mean_0 += predicted
            n_0 += 1
            if predicted > max_0:
                max_0 = predicted
            if predicted < min_0:
                min_0 = predicted
        elif expected == 1:
            mean_1 += predicted
            n_1 += 1
            if predicted > max_1:
                max_1 = predicted
            if predicted < min_1:
                min_1 = predicted
        else:
            mean_2 += predicted
            n_2 += 1
            if predicted > max_2:
                max_2 = predicted
            if predicted < min_2:
                min_2 = predicted
    mean_0, mean_1, mean_2 = mean_0 / n_0, mean_1 / n_1, mean_2 / n_2
    threshold_1, threshold_2 = (max_0 + min_1) / 2, (max_1 + min_2) / 2

    accuracy = 0
    for sample in test_dataset:
        expected = sample[len(sample) - 1]
        predicted = predict(network, sample, function)[0]
        if predicted < threshold_1:
            predicted = 0
        elif predicted < threshold_2:
            predicted = 1
        else:
            predicted = 2
        if expected == predicted:
            accuracy += 1
        # print('expected: ' + str(round(expected, precision)) + ' predicted: ' + str(round(predicted, precision)))
    accuracy /= n_test
    accuracy *= 100

    print("NEURAL NETWORK | accuracy on test set: " + str(round(accuracy, precision)) + "%")
    print("labels 0's mean: " + str(round(mean_0, precision)) +
          ", labels 1's mean: " + str(round(mean_1, precision)) +
          ", labels 2's mean: " + str(round(mean_2, precision)) +
          ", \nlabels 0's min: " + str(round(min_0, precision)) +
          ", labels 1's min: " + str(round(min_1, precision)) +
          ", labels 2's min: " + str(round(min_2, precision)) +
          ", \nlabels 0's max: " + str(round(max_0, precision)) +
          ", labels 1's max: " + str(round(max_1, precision)) +
          ", labels 2's max: " + str(round(max_2, precision)) +
          ",\nthreshold_1: " + str(round(threshold_1, precision)) +
          ", threshold_2: " + str(round(threshold_2, precision)) + ".")


main()
