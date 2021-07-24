from tensorflow.keras.layers import Input, BatchNormalization, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import os


# Uncomment this to disable your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def make_detector_network(img_cols, img_rows):
    """
    Makes the convolutional neural network used in the object detector.

    :param img_cols: number of columns of the input image.
    :param img_rows: number of rows of the input image.
    :return: Keras' model of the neural network.
    """
    # Input layer
    input_image = Input(shape=(img_cols, img_rows, 3))

    # Layer 1
    layer = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    layer = BatchNormalization(name='norm_1')(layer)
    layer = LeakyReLU(alpha=0.1, name='leaky_relu_1')(layer)

    # Todo: Implement layers 2 to 5

    # Layer 6
    layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(layer)
    layer = BatchNormalization(name='norm_6')(layer)
    layer = LeakyReLU(alpha=0.1, name='leaky_relu_6')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', name='max_pool_6')(layer)

    skip_connection = layer

    # Todo: Implement layers 7A, 8, and 7B

    # Concatenating layers 7B and 8
    layer = concatenate([skip_connection, layer], name='concat')

    # Layer 9 (last layer)
    layer = Conv2D(10, (1, 1), strides=(1, 1), padding='same', name='conv_9', use_bias=True)(layer)

    model = Model(inputs=input_image, outputs=layer, name='ITA_YOLO')

    return model


model = make_detector_network(120, 160)
model.summary()  # prints the network summary
