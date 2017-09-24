# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

from keras import backend as K


PRETRAINED_MODEL_LIST = {'inception_v3', 'mobilenet', 'resnet50', 'resnet101',
                         'resnet152', 'vgg16', 'vgg19', 'xception', 'alexnet'}

TARGET_SIZE_DICT = {
    'inception_v3': (299, 299),
    'mobilenet': (224, 224),
    'resnet50': (224, 224),
    'resnet101': (224, 224),
    'resnet152': (224, 224),
    'vgg16': (224, 224),
    'vgg19': (224, 224),
    'xception': (299, 299),
    'alexnet': (227, 227)
}

PRETRAINED_MODEL_DICT = {
    'inception_v3': 'InceptionV3',
    'mobilenet': 'MobileNet',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'resnet152': 'ResNet152',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
    'xception': 'Xception',
    'alexnet': 'AlexNet'
}


IMAGE_EXTENSIONS = {
    'rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff',
    'rast', 'xbm', 'jpg', 'jpeg', 'bmp', 'png'
}


def set_image_format():
    if K.backend() == 'theano':
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')


set_image_format()
