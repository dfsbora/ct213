import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from utils import sum_gt_zero, xor


num_cases = 200  # number of auto-generated cases
num_epochs = 1000  # number of epochs for training
#classification_function = sum_gt_zero  # selects sum_gt_zero as the classification function
classification_function = xor  # selects xor as the classification function
# Figure format used for saving figures
fig_format = 'png'  # for Word users
# fig_format = 'svg'
# fig_format = 'eps'  # for LaTeX users

# Setting the random seed of numpy's random library for reproducibility reasons
np.random.seed(0)

# Creating the dataset
inputs = 5.0 * (-1.0 + 2.0 * np.random.rand(num_cases, 2))
expected_outputs = np.array([classification_function(x) for x in inputs])

# Separating the dataset into positive and negative samples
positives_indices = np.where(expected_outputs >= 0.5)
negatives_indices = np.where(expected_outputs < 0.5)
positives = inputs[positives_indices]
negatives = inputs[negatives_indices]

# Creating and training the neural network
neural_network = NeuralNetwork(2, 10, 1, 6.0)
costs = np.zeros(num_epochs)
inputs_nn = inputs.T
for i in range(num_epochs):
    neural_network.back_propagation(inputs_nn, expected_outputs)
    costs[i] = neural_network.compute_cost(inputs_nn, expected_outputs)
    print('epoch: %d; cost: %f' % (i + 1, costs[i]))

# Plotting cost function convergence
plt.plot(costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.grid()
plt.savefig('cost_function_convergence.' + fig_format, format=fig_format)

# Plotting positive and negative samples
plt.figure()
plt.plot(positives[:, 0], positives[:, 1], '+r')
plt.plot(negatives[:, 0], negatives[:, 1], 'x')
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dataset')
plt.savefig('dataset.' + fig_format, format=fig_format)


# Plotting the decision regions of the neural network
plt.figure()
x = np.arange(-5.0, 5.05, 0.05)
y = np.arange(-5.0, 5.05, 0.05)
xx, yy = np.meshgrid(x, y)
inputs_region = np.array([xx.flatten(), yy.flatten()])
_, a = neural_network.forward_propagation(inputs_region)
z = a[-1].reshape(len(x), len(y))
plt.contourf(x, y, z)
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
plt.plot(positives[:, 0], positives[:, 1], '+', color='tab:orange')
plt.plot(negatives[:, 0], negatives[:, 1], 'x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Neural Network Classification')
plt.savefig('neural_net_classification.' + fig_format, format=fig_format)
plt.show()
