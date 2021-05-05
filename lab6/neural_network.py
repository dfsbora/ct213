import numpy as np
from utils import sigmoid, sigmoid_derivative


class NeuralNetwork:
    """
    Represents a two-layers Neural Network (NN) for multi-class classification.
    The sigmoid activation function is used for all neurons.
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs, alpha):
        """
        Constructs a three-layers Neural Network.

        :param num_inputs: number of inputs of the NN.
        :type num_inputs: int.
        :param num_hiddens: number of neurons in the hidden layer.
        :type num_hiddens: int.
        :param num_outputs: number of outputs of the NN.
        :type num_outputs: int.
        :param alpha: learning rate.
        :type alpha: float.
        """
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.alpha = alpha
        self.weights = [None] * 3
        self.biases = [None] * 3
        self.weights[1] = 0.001 * np.random.randn(num_hiddens, num_inputs)
        self.weights[2] = 0.001 * np.random.randn(num_outputs, num_hiddens)
        self.biases[1] = np.zeros((num_hiddens, 1))
        self.biases[2] = np.zeros((num_outputs, 1))

    def forward_propagation(self, inputs):
        """
        Executes forward propagation.
        Notice that the z and a of the first layer (l = 0) are equal to the NN's input.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :return z: values computed by applying weights and biases at each layer of the NN.
        :rtype z: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        :return a: activations computed by applying the activation function to z at each layer.
        :rtype a: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        """
        z = [None] * 3
        a = [None] * 3
        z[0] = inputs
        a[0] = inputs
        # Add logic for neural network inference
        z[1] = self.weights[1]@a[0] + self.biases[1]
        a[1] = sigmoid(z[1])
        z[2] = self.weights[2]@a[1] + self.biases[2]
        a[2] = sigmoid(z[2])

        #a[2] = 0.001 * np.ones((self.num_outputs, inputs.shape[1]))  # Change this line
        return z, a

    def compute_cost(self, inputs, expected_outputs):
        """
        Computes the logistic regression cost of this network.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: list of numpy matrices.
        :return: logistic regression cost.
        :rtype: float.
        """
        z, a = self.forward_propagation(inputs)
        y = expected_outputs
        y_hat = a[-1]
        cost = np.mean(-(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))
        return cost

    def compute_gradient_back_propagation(self, inputs, expected_outputs):
        """
        Computes the gradient with respect to the NN's parameters using back propagation.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (num_outputs, num_samples) numpy array.
        :return weights_gradient: gradients of the weights at each layer.
        :rtype weights_gradient: 3-dimensional list of numpy arrays.
        :return biases_gradient: gradients of the biases at each layer.
        :rtype biases_gradient: 3-dimensional list of numpy arrays.
        """
        weights_gradient = [None] * 3
        biases_gradient = [None] * 3
        delta = [None] * 3
        # Add logic to compute the gradients

        z, a = self.forward_propagation(inputs)


        delta[2] = (a[2]-np.transpose(expected_outputs)) #* sigmoid_derivative(z[1])
        print("d2: ", np.shape(delta[2]), "a1: ", np.shape(a[1]))
        weights_gradient[2] = delta[2] * a[1]
        print("wg2: ", np.shape(weights_gradient[2]))
        biases_gradient[2] = delta[2]
        print("b2: ", np.shape(biases_gradient[2]))

        print("------")
        delta[1] = (weights_gradient[2]@np.transpose(delta[2])) * sigmoid_derivative(z[1])
        print("d1: ", np.shape(delta[1]), "a0: ", np.shape(inputs))
        weights_gradient[1] = inputs@np.transpose(delta[1])
        print("wg1: ", np.shape(weights_gradient[1]))
        biases_gradient[1] = np.transpose(delta[1])


        return weights_gradient, biases_gradient

    def back_propagation(self, inputs, expected_outputs):
        """
        Executes the back propagation algorithm to update the NN's parameters.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (num_outputs, num_samples) numpy array.
        """
        weights_gradient, biases_gradient = self.compute_gradient_back_propagation(inputs, expected_outputs)
        # Add logic to update the weights and biases

        #weights_gradient[2] = np.array(weights_gradient[2])
        print(np.shape(weights_gradient[2]), np.shape(self.weights[2]))

        #weights_gradient[2] = weights_gradient.reshape((weights_gradient[2].shape[0], 1))

        print(np.shape(self.weights[2]), np.shape(self.weights[1]), np.shape(self.biases[2]), np.shape(self.biases[1]))
        self.weights[2] = self.weights[2] - np.mean(self.alpha * weights_gradient[2], axis=1)
        self.weights[1] = self.weights[1] -  np.mean(self.alpha * weights_gradient[1], axis=1)
        self.biases[2] = self.biases[2] - np.mean(self.alpha * biases_gradient[2], axis=1)
        self.biases[1] = self.biases[1] - np.mean(self.alpha * biases_gradient[1], axis=1)
        print("1: ", self.biases[1])
        #print("0: ", np.mean(biases_gradient[1], axis=0))

        print(np.shape(self.weights[2]), np.shape(self.weights[1]), np.shape(self.biases[2]), np.shape(self.biases[1]))
        #pass # Remove this line
