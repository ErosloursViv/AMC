import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Dense, Dropout, ReLU, Add
from keras.backend import get_uid

# SBCNN
class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, name='', **kwargs):
        self.custom_name = name+'_'+str(get_uid(name))
        super(ConvBlock, self).__init__(**kwargs)
        self.conv_layer = Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation=None,
                                 name = self.custom_name if name else None)
        self.norm_layer = BatchNormalization()
        self.relu_layer = ReLU()

    def __call__(self, inputs, **kwargs):
        x = self.conv_layer(inputs)
        x = self.norm_layer(x, **kwargs)
        x = self.relu_layer(x)
        return x


class ABlock(Layer):
    def __init__(self):
        super().__init__()
        self.conv_b1 = ConvBlock(32, (1, 8))
        self.conv_b2 = ConvBlock(32, (3, 1))
        self.max_pool1 = MaxPooling2D(pool_size=(1, 2))
        self.conv_b3 = ConvBlock(32, (1, 8))
        self.conv_b4 = ConvBlock(32, (3, 1))
        self.max_pool2 = MaxPooling2D(pool_size=(1, 2))

    def __call__(self, inputs, **kwargs):
        x = self.conv_b1(inputs, **kwargs)
        x = self.conv_b2(x, **kwargs)
        x = self.max_pool1(x)
        x = self.conv_b3(x, **kwargs)
        x = self.conv_b4(x, **kwargs)
        x = self.max_pool2(x)
        return x


class BBlock(Layer):
    def __init__(self):
        super().__init__()
        self.conv_b1 = ConvBlock(32, (1, 1), name='res_layer')  # hard-coded
        self.conv_b2 = ConvBlock(24, (1, 8))  # hard-coded
        self.conv_b3 = ConvBlock(24, (3, 1))  # hard-coded
        self.conv_b4 = ConvBlock(32, (1, 1), name='res_layer')  # hard-coded
        self.avg_pool = AveragePooling2D(pool_size=(2, 1))  # hard-coded
        self.add = Add()

    def __call__(self, inputs, **kwargs):
        x = self.conv_b1(inputs, **kwargs)
        s = x  # skip connection
        x = self.conv_b2(x, **kwargs)
        x = self.conv_b3(x, **kwargs)
        x = self.conv_b4(x, **kwargs)
        # x = x + s
        x = self.add([x, s])
        x = self.avg_pool(x)
        return x


class CBlock(Layer):
    def __init__(self):
        super().__init__()
        self.conv_b1 = ConvBlock(64, (1, 1), name='res_layer')
        self.conv_b2 = ConvBlock(32, (1, 8))
        self.conv_b3 = ConvBlock(64, (1, 1), name='res_layer')
        self.add = Add()

    def __call__(self, inputs, **kwargs):
        x = self.conv_b1(inputs, **kwargs)
        s = x  # skip connection
        x = self.conv_b2(x, **kwargs)
        x = self.conv_b3(x, **kwargs)
        # x = x + s
        x = self.add([x, s])
        return x


class SBCNN(Layer):
    def __init__(self):
        super().__init__()
        # self.input_layer = InputLayer(input_shape=(2, 1024, 1)) # hard-coded
        self.a_block = ABlock()
        self.b_block = BBlock()
        self.c_block1 = CBlock()
        self.c_block2 = CBlock()
        self.gavg_pool = GlobalAveragePooling2D()
        self.dense_layer = Dense(24)  # hard-coded

    def __call__(self, inputs, **kwargs):
        # x = self.input_layer(inputs)
        x = self.a_block(inputs, **kwargs)
        x = self.b_block(x, **kwargs)
        x = self.c_block1(x, **kwargs)
        x = self.c_block2(x, **kwargs)
        x = self.gavg_pool(x)
        x = self.dense_layer(x)
        return x


# IBCNN
class ABlockI(Layer):
    def __init__(self):
        super(Layer).__init__()
        self.conv_b1 = ConvBlock(32, (3, 1))
        self.conv_b2 = ConvBlock(64, (1, 1))
        self.conv_b3 = ConvBlock(32, (3, 1))
        self.max_pool1 = MaxPooling2D(pool_size=(2, 1))

    def __call__(self, inputs, **kwargs):
        x = self.conv_b1(inputs, **kwargs)
        x = self.conv_b2(x, **kwargs)
        x = self.conv_b3(x, **kwargs)
        x = self.max_pool1(x)
        return x


class BBlockI(Layer):
    def __init__(self):
        super(Layer).__init__()
        self.conv_b1 = ConvBlock(32, (1, 1), name='res_layer')
        self.conv_b2 = ConvBlock(24, (3, 1))
        self.conv_b3 = ConvBlock(32, (1, 1), name='res_layer')
        self.max_pool1 = MaxPooling2D(pool_size=(2, 1))
        self.add = Add()

    def __call__(self, inputs, **kwargs):
        x = self.conv_b1(inputs, **kwargs)
        s = x  # skip
        x = self.conv_b2(x, **kwargs)
        x = self.conv_b3(x, **kwargs)
        # x = x + s
        x = self.add([x, s])
        x = self.max_pool1(x)
        return x


class CBlockI(Layer):
    def __init__(self, n_kernels):
        super(Layer).__init__()
        self.conv_b1 = ConvBlock(int(n_kernels), (1, 1), name='res_layer')
        self.conv_b2 = ConvBlock(int(n_kernels / 2), (3, 1))
        self.conv_b3 = ConvBlock(int(n_kernels), (1, 1), name='res_layer')
        self.add = Add()

    def __call__(self, inputs, **kwargs):
        x = self.conv_b1(inputs, **kwargs)
        s = x  # skip
        x = self.conv_b2(x, **kwargs)
        x = self.conv_b3(x, **kwargs)
        # x = x + s
        x = self.add([x, s])
        return x


class IBCNN(Layer):
    def __init__(self):
        super(Layer).__init__()
        self.a_block = ABlockI()
        self.b_block = BBlockI()
        self.c_block1 = CBlockI(n_kernels=64)
        self.c_block2 = CBlockI(n_kernels=128)
        self.gavg_pool = GlobalAveragePooling2D()
        self.drop_out = Dropout(rate=.2)
        self.dense_layer = Dense(24)

    def __call__(self, inputs, **kwargs):
        x = self.a_block(inputs, **kwargs)
        x = self.b_block(x, **kwargs)
        x = self.c_block1(x, **kwargs)
        x = self.c_block2(x, **kwargs)
        x = self.gavg_pool(x)
        x = self.drop_out(x, **kwargs)
        x = self.dense_layer(x)
        return x


def build_sbcnn_model():
    input_signal = keras.Input((2, 1024, 1))
    output_logit = SBCNN()(input_signal)
    sbcnn = keras.Model(input_signal, output_logit)

    return sbcnn


def build_ibcnn_model(sbcnn, trainable=True):
    # image converting, scaling and reshaping
    quant_layer = keras.layers.Lambda(
        lambda x: tf.quantization.fake_quant_with_min_max_args(x, min=-128, max=127, num_bits=8),
        output_shape=None)

    min_max_layer = keras.layers.Lambda(
        lambda x: (x + 128.) / 255.)

    reshape_layer = tf.keras.layers.Reshape((24, 1, 1))

    sbcnn.trainable = trainable  # disable training in the top layers of the model
    sbcnn.name = 'SBCNN'

    input_signal = keras.Input((2, 1024, 1))
    x = sbcnn(input_signal)
    x = quant_layer(x)  # quantization layer
    x = min_max_layer(x)  # output values [0, 1]
    x = reshape_layer(x)  # reshape to enable 2d convolution
    output_logit_ibcnn = IBCNN()(x)  # pass through the IBCNN layer
    ibcnn = keras.Model(input_signal, output_logit_ibcnn)  # create model
    return ibcnn
