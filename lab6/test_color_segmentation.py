import numpy as np
import matplotlib.pyplot as plt
import math
from neural_network import NeuralNetwork
from matplotlib.image import imread

threshold = 0.5  # threshold used for discretizing the output
num_iterations = 200  # number of epochs for training
mini_batch_samples_per_class = 100  # number of samples per class used in the mini-batch
# Figure format used for saving figures
fig_format = 'png'  # for Word users
# fig_format = 'svg'
# fig_format = 'eps'  # for LaTeX users


def encode_class(c):
    """
    Transforms a class representation as a number into an array of 0's and 1's.
    1 -> [1 0]; 2 -> [0 1]; otherwise -> [0 0].

    :param c: class as a number.
    :type c: int.
    :return: class as an array of 0's and 1's.
    :rtype: numpy matrix.
    """
    if c == 1:
        return np.array([1, 0])
    elif c == 2:
        return np.array([0, 1])
    else:
        return np.array([0, 0])


def decode_class(c):
    """
    Transforms a class representation as an array of 0's and 1's into a number.

    :param c: class as an array of 0's and 1's.
    :type c: numpy matrix.
    :return: class as a number.
    :rtype: int.
    """
    index = np.argmax(c)
    if c[index] < threshold:
        return 0
    return index + 1


# Loading the image
image = imread('nao1.jpg')

# Setting the random seed of numpy's random library for reproducibility reasons
np.random.seed(0)

# Loading the dataset
data = np.loadtxt('nao1.txt')
num_cases = data.shape[0]

# 70 percent of the samples of each class are selected randomly for making the training set.
# Then, the remaining samples are used for the test set.
greens = []
whites = []
others = []
for i in range(num_cases):
    c = data[i, 3]
    if abs(c - 1) < 1.0e-3:  # if the pixel is in class 1
        greens.append(i)
    elif abs(c - 2) < 1.0e-3:  # if the pixel is in class 2
        whites.append(i)
    else:  # if the pixel isn't in class 1 or 2, then we say its class is undefined (class 0)
        others.append(i)
num_greens = len(greens)
num_whites = len(whites)
num_others = len(others)
num_greens_training = math.floor(0.7 * num_greens)
num_whites_training = math.floor(0.7 * num_whites)
num_others_training = math.floor(0.7 * num_others)
np.random.shuffle(greens)
np.random.shuffle(whites)
np.random.shuffle(others)
greens_training = greens[0:num_greens_training]
whites_training = whites[0:num_whites_training]
others_training = others[0:num_others_training]
greens_test = greens[num_greens_training:-1]
whites_test = greens[num_whites_training:-1]
others_test = greens[num_others_training:-1]
test_set = greens_test + whites_test + others_test

# Training the neural network
print('Training the neural network...')
neural_network = NeuralNetwork(3, 20, 2, 6.0)
costs = np.zeros(num_iterations)
for i in range(num_iterations):
    np.random.shuffle(greens_training)
    np.random.shuffle(whites_training)
    np.random.shuffle(others_training)
    mini_batch = greens_training[0:mini_batch_samples_per_class] + whites_training[0:mini_batch_samples_per_class] + \
                   others_training[0:mini_batch_samples_per_class]
    inputs = (1.0 / 255.0) * data[mini_batch, 0:3].T
    expected_outputs = np.array([encode_class(x) for x in data[mini_batch, 3]]).T
    neural_network.back_propagation(inputs, expected_outputs)
    costs[i] = neural_network.compute_cost(inputs, expected_outputs)
    print('iteration: %d; cost: %f' % (i + 1, costs[i]))

# Plotting cost function convergence
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence (Segmentation)')
plt.grid()
plt.savefig('cost_function_convergence_segmentation.' + fig_format, format=fig_format)
plt.show()

# Evaluating the neural network on the test set
print('Evaluating the neural network on the test set...')
inputs_test = (1.0 / 255.0) * np.array(data[test_set, 0:3]).T
expected_outputs_test = np.array([encode_class(x) for x in data[test_set, 3]]).T
cost_test = neural_network.compute_cost(inputs_test, expected_outputs_test)
print('Cost on the test set: %f' % cost_test)

# Segmenting an actual image
height = image.shape[0]
width = image.shape[1]
num_channels = image.shape[2]

colors = [(0, 0, 0), (0, 255, 0), (255, 255, 255)]
print('Segmenting image...')
_, a = neural_network.forward_propagation((1.0 / 255.0) * image.reshape(image.shape[0] * image.shape[1], image.shape[2]).T)
c = np.array([colors[decode_class(y)] for y in a[-1].T])
segmented_image = c.reshape(height, width, num_channels)

plt.figure()
plt.imshow(image)
plt.title('Original Image')
plt.savefig('original_image.' + fig_format, format=fig_format)
plt.figure()
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.savefig('segmented_image.' + fig_format, format=fig_format)
plt.show()
