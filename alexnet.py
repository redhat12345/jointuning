# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras import backend as K
from keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D,
    Flatten, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dropout, Lambda, concatenate)
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file


WEIGHTS_PATH_TH = 'https://dl.dropboxusercontent.com/s/3594hzu89qasw7n/alexnet_weights_th.h5?dl=0'
WEIGHTS_PATH_TF = 'https://dl.dropboxusercontent.com/s/g9nxu4fbaug8xh8/alexnet_weights_tf.h5?dl=0'
MD5_HASH_TH = '43d2508b3ac2173680ed1b99953b6535'
MD5_HASH_TF = '117beeb60b5daa36d9bc654a2c0404f9'


def preprocess_input(x, data_format=None, mode='caffe'):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
        mode: One of "caffe", "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
    # Returns
        Preprocessed tensor in 'rgb' mode.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if mode == 'tf':
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    if data_format == 'channels_first':
        if x.ndim == 3:
            # Zero-center by mean pixel
            x[0, :, :] -= 123.68
            x[1, :, :] -= 116.779
            x[2, :, :] -= 103.939
        else:
            x[:, 0, :, :] -= 123.68
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 103.939
    else:
        # Zero-center by mean pixel
        x[..., 0] -= 123.68
        x[..., 1] -= 116.779
        x[..., 2] -= 103.939
    return x


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(
            K.permute_dimensions(square, (0, 2, 3, 1)), ((0, 0), (half, half)))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split
        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')
        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def AlexNet(include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            weights_path=None):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(
                tensor=input_tensor, shape=input_shape, name='data')
        else:
            img_input = input_tensor

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4),
                    activation='relu', name='conv_1')(img_input)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = concatenate([Conv2D(
        128, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
        splittensor(ratio_split=2, id_split=i)(conv_2)) for i in range(2)
    ], axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = concatenate([Conv2D(
        192, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
        splittensor(ratio_split=2, id_split=i)(conv_4)) for i in range(2)
    ], axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = concatenate([Conv2D(
        128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
        splittensor(ratio_split=2, id_split=i)(conv_5)) for i in range(2)
    ], axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    if include_top:
        # classification block
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, name='dense_3')(dense_3)
        output = Activation('softmax', name='softmax')(dense_3)
    else:
        if pooling == 'avg':
            output = GlobalAveragePooling2D()(dense_1)
        elif pooling == 'max':
            output = GlobalMaxPooling2D()(dense_1)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=output)

    if weights_path:
        model.load_weights(weights_path)

    # load weights
    if weights == 'imagenet':
        filename = 'alexnet_weights_{}.h5'.format(K.image_dim_ordering())
        if K.backend() == 'theano':
            path = WEIGHTS_PATH_TH
            md5_hash = MD5_HASH_TH
        else:
            path = WEIGHTS_PATH_TF
            md5_hash = MD5_HASH_TF
        weights_path = get_file(
            fname=filename,
            origin=path,
            cache_subdir='models',
            md5_hash=md5_hash,
            hash_algorithm='md5')
        model.load_weights(weights_path, by_name=True)

        if (K.image_data_format() == 'channels_first' and
                K.backend() == 'tensorflow'):
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    return model
