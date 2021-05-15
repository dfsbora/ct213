import os
from utils import read_mnist, load_model_from_json, display_image
import random
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

NUM_IMAGES_RANDOM = 5
NUM_IMAGES_MISCLASSIFICATION = 5

test_features, test_labels = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

model = load_model_from_json('lenet5')
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

predicted_labels = model.predict(test_features)
score = model.evaluate(test_features, to_categorical(test_labels))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

for i in range(NUM_IMAGES_RANDOM):
    index = random.randint(0, test_features.shape[0])
    display_image(test_features[index],
                  'Example: %d. Expected Label: %d. Predicted Label: %d.' %
                  (index, test_labels[index], np.argmax(predicted_labels[index, :])))
    plt.savefig('test_image_%d.png' % index, format='png')


count = 0
for i in range(test_features.shape[0]):
    if count == NUM_IMAGES_MISCLASSIFICATION:
        break
    if np.argmax(predicted_labels[i, :]) != test_labels[i]:
        display_image(test_features[i],
                      'Example: %d. Expected Label: %d. Predicted Label: %d.' %
                      (i, test_labels[i], np.argmax(predicted_labels[i, :])))
        plt.savefig('misclassified_image_%d.png' % i, format='png')
        count += 1

plt.show()
