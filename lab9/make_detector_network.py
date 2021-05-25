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
    input_1 = Input(shape=(img_cols, img_rows, 3))

    # Layer 1
    conv_1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_1)
    norm_1 = BatchNormalization(name='norm_1')(conv_1)
    leaky_relu_1 = LeakyReLU(alpha=0.1, name='leaky_relu_1')(norm_1)

    # Todo: Implement layers 2 to 5

    # Layer 2
    conv_2 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(leaky_relu_1)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    leaky_relu_2 = LeakyReLU(alpha=0.1, name='leaky_relu_2')(norm_2)

    # Layer 3
    conv_3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(leaky_relu_2)
    norm_3 = BatchNormalization(name='norm_3')(conv_3)
    leaky_relu_3 = LeakyReLU(alpha=0.1, name='leaky_relu_3')(norm_3)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_3')(leaky_relu_3)

    # Layer 4
    conv_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(max_pool_3)
    norm_4 = BatchNormalization(name='norm_4')(conv_4)
    leaky_relu_4 = LeakyReLU(alpha=0.1, name='leaky_relu_4')(norm_4)
    max_pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_4')(leaky_relu_4)

    # Layer 5
    conv_5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(max_pool_4)
    norm_5 = BatchNormalization(name='norm_5')(conv_5)
    leaky_relu_5 = LeakyReLU(alpha=0.1, name='leaky_relu_5')(norm_5)
    max_pool_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool_5')(leaky_relu_5)

    # Layer 6
    conv_6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(max_pool_5)
    norm_6 = BatchNormalization(name='norm_6')(conv_6)
    leaky_relu_6 = LeakyReLU(alpha=0.1, name='leaky_relu_6')(norm_6)
    max_pool_6 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', name='max_pool_6')(leaky_relu_6)


    # Todo: Implement layers 7A, 8, and 7B

    # Layer 7A
    conv_7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(max_pool_6)
    norm_7 = BatchNormalization(name='norm_7')(conv_7)
    leaky_relu_7 = LeakyReLU(alpha=0.1, name='leaky_relu_7')(norm_7)

    # Layer 7B
    conv_skip = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv_skip', use_bias=False)(max_pool_6)
    norm_skip = BatchNormalization(name='norm_skip')(conv_skip)
    leaky_relu_skip = LeakyReLU(alpha=0.1, name='leaky_relu_skip')(norm_skip)
    skip_connection = leaky_relu_skip

    # Layer 8
    conv_8 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(leaky_relu_7)
    norm_8 = BatchNormalization(name='norm_8')(conv_8)
    leaky_relu_8 = LeakyReLU(alpha=0.1, name='leaky_relu_8')(norm_8)


    # Concatenating layers 7B and 8
    concat = concatenate([skip_connection, leaky_relu_8], name='concat')

    # Layer 9 (last layer)
    conv_9 = Conv2D(10, (1, 1), strides=(1, 1), padding='same', name='conv_9', use_bias=True)(concat)

    model = Model(inputs=input_1, outputs=conv_9, name='ITA_YOLO')

    return model


model = make_detector_network(120, 160)
model.summary()  # prints the network summary
