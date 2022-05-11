import copy
import math
from random import seed
from random import random
import numpy
from sklearn import datasets


def new_nn(n_input, n_output, n_hidden, n_hidden_layer, randomize):
    network = list()
    for j in range(n_hidden_layer):
        hidden_layer = [{'weights': [random() if randomize else 0 for i in range(n_input + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'weights': [random() if randomize else 0 for i in range(n_hidden + 1)]} for i in range(n_output)]
    network.append(output_layer)
    return network


def activate(sample, weights):
    activation = weights[len(weights)-1]
    for i in range(1, len(weights) - 1):
        activation += weights[i] * sample[i-1]
    return activation


def forward_propagate(neural_network, sample, function):
    inputs = sample[:len(sample)-1]
    for layer in neural_network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation, function)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    n_input = len(inputs)
    sum = 0
    for input in inputs:
        sum += input
    if sum != 0:
        for i in range(n_input):
            inputs[i] = inputs[i] / sum
    return inputs


def transfer(activation, function):
    return function(activation)


def transfer_derivative(output, derivative_function):
    return derivative_function(output)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    return x if x > 0 else 0


def relu_derivative(x):
    return 0 if x < 0 else 1


def tanh(x):
    return math.tanh(x)


def backpropagation(network, expected, derivative_function):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['correction'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['correction'] = errors[j] * transfer_derivative(neuron['output'], derivative_function)


def train_network(neural_network, training_dataset, learning_rate, decadence,
                  n_max_epoch, n_output, function, derivative_function, precision):
    error = 0
    for epoch in range(n_max_epoch):
        prev_error = error
        error = 0
        for sample in training_dataset:
            outputs = forward_propagate(neural_network, sample, function)
            # print(outputs)
            expected = [0 for i in range(n_output)]
            expected[sample[len(sample)-1]] = 1
            for i in range(n_output):
                # error += math.pow((expected[i]-outputs[i]), 2)
                # error += -(expected[i]*math.log(outputs[i], math.e) + (1-expected[i])*math.log(1-outputs[i], math.e))
                error += bce(expected[i], outputs[i])
            backpropagation(neural_network, expected, derivative_function)
            update_weights(neural_network, sample, learning_rate / (decadence * (epoch+1)))
        error /= (len(training_dataset)*n_output)
        # print('epoch: ' + str(epoch) + ' learning rate: ' + str(round(learning_rate, precision))
        #       + ' error: ' + str(round(error, precision)))
        if epoch != 0 and error > prev_error:
            break


def bce(expected, predicted):
    res = 0
    # to avoid math exceptions
    if predicted == 0:
        a = float('-inf')
        b = math.log(1 - predicted, math.e)
    elif predicted == 1:
        a = math.log(predicted, math.e)
        b = float('-inf')
    else:
        a = math.log(predicted, math.e)
        b = math.log(1 - predicted, math.e)
    res += -(expected * a + (1 - expected) * b)
    return res


def update_weights(neural_network, sample, learning_rate):
    for i in range(len(neural_network)):
        if i == 0:  # input layer (first layer)
            inputs = sample[0:len(sample)-1]
        else:  # hidden/output layer
            inputs = [neuron['output'] for neuron in neural_network[i - 1]]
        for neuron in neural_network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learning_rate * neuron['correction'] * inputs[j]
            neuron['weights'][len(inputs)] -= learning_rate * neuron['correction']


def predict(neural_network, sample, activation_function):
    predicted = forward_propagate(neural_network, sample, activation_function)
    return predicted.index(max(predicted))


def main():
    # numpy.seterr(all="ignore")
    iris = datasets.load_iris()
    n_features = 4
    dataset = iris.data[:, :n_features]
    labels = iris.target
    learning_rate = 0.001
    precision = 3
    decadence = 1 + learning_rate
    n_max_epoch = 1000
    normalized_dataset = dataset.copy()
    n_sample = len(dataset)
    early_termination = True
    normalize = True

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
    training_dataset = numpy.concatenate((numpy.concatenate((normalized_dataset[0:40], normalized_dataset[50:90]),
                                                            axis=0), normalized_dataset[100:140]), axis=0)
    n_training = len(training_dataset)
    training_dataset_labels = numpy.concatenate((numpy.concatenate((labels[0:40], labels[50:90]),
                                                                   axis=0), labels[100:140]), axis=0)
    test_dataset = numpy.concatenate((numpy.concatenate((normalized_dataset[40:50], normalized_dataset[90:100]),
                                                        axis=0), normalized_dataset[140:150]), axis=0)
    n_test = len(test_dataset)
    test_dataset_labels = numpy.concatenate((numpy.concatenate((labels[40:50], labels[90:100]),
                                                               axis=0), labels[140:150]), axis=0)

    tmp_training_dataset = []
    for i in range(0, n_training):
        tmp_sample = []
        for j in range(0, n_features):
            tmp_sample.insert(j, training_dataset[i][j])
        tmp_sample.insert(n_features, training_dataset_labels[i])
        tmp_training_dataset.insert(i, tmp_sample)

    tmp_test_dataset = []
    for i in range(0, n_test):
        tmp_sample = []
        for j in range(0, n_features):
            tmp_sample.insert(j, test_dataset[i][j])
        tmp_sample.insert(n_features, test_dataset_labels[i])
        tmp_test_dataset.insert(i, tmp_sample)

    training_dataset1 = copy.deepcopy(tmp_training_dataset)
    for i in range(0, n_training):  # 010
        if training_dataset1[i][n_features] == 2:
            training_dataset1[i][n_features] = 0
    test_dataset1 = copy.deepcopy(tmp_test_dataset)
    for i in range(0, n_test):  # 010
        if test_dataset1[i][n_features] == 2:
            test_dataset1[i][n_features] = 0

    training_dataset2 = copy.deepcopy(tmp_training_dataset)
    for i in range(0, n_training):  # 100
        if training_dataset2[i][n_features] == 0:
            training_dataset2[i][n_features] = 1
        else:
            training_dataset2[i][n_features] = 0
    test_dataset2 = copy.deepcopy(tmp_test_dataset)
    for i in range(0, n_test):  # 100
        if test_dataset2[i][n_features] == 0:
            test_dataset2[i][n_features] = 1
        else:
            test_dataset2[i][n_features] = 0

    training_dataset3 = copy.deepcopy(tmp_training_dataset)
    for i in range(0, n_training):  # 001
        if training_dataset3[i][n_features] == 2:
            training_dataset3[i][n_features] = 1
        else:
            training_dataset3[i][n_features] = 0
    test_dataset3 = copy.deepcopy(tmp_test_dataset)
    for i in range(0, n_test):  # 001
        if test_dataset3[i][n_features] == 2:
            test_dataset3[i][n_features] = 1
        else:
            test_dataset3[i][n_features] = 0

    training_datasets = [training_dataset1, training_dataset2, training_dataset3]
    test_datasets = [test_dataset1, test_dataset2, test_dataset3]

    for i in range(0, 3):
        seed(1)
        n_input = n_features
        n_output = 2
        network = new_nn(n_input, n_output, 4, 2, True)
        train_network(network, training_datasets[i], learning_rate, decadence, n_max_epoch, n_output,
                      relu, relu_derivative, precision)
        # for layer in network:
        #     for neuron in layer:
        #         print(neuron['weights'])
        #     print()

        accuracy = 0
        for sample in test_datasets[i]:
            expected = sample[len(sample)-1]
            predicted = predict(network, sample, relu)
            if expected == predicted:
                accuracy += 1
            print('expected: ' + str(round(expected, precision)) + ' predicted: ' + str(round(predicted, precision)))
        accuracy /= n_test
        accuracy *= 100
        print("accuracy: " + str(round(accuracy, precision)))


main()
