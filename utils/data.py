# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os
import importlib
import time
import tables
import numpy as np

from .. import config
from . import directory
from . import stats

from functools import partial
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array


def ceildiv(dividend, divisor):
    """ceiling-division for two integers
    """
    return -(-dividend // divisor)


def path_to_tensor(image_path, target_size, grayscale=False, data_format=None):
    """
    Read an image from its path, resize it to a specified size (height, width),
    and return a numpy array that is ready to be passed to the `predict` method
    of a trained model

    Parameters
    ----------
    image_path: string
        the path of an image
    target_size: tuple/list
        (height, width) of the image
    grayscale: bool
        whether to load the image as grayscale
    data_format: str
        one of `channels_first`, `channels_last`

    Returns
    -------
        a numpy array that is to be readily passed to the `predict` method of
        a trained model
    """
    assert os.path.exists(os.path.abspath(image_path))
    if data_format is None:
        data_format = K.image_data_format()
    image = load_img(image_path, grayscale=grayscale, target_size=target_size)
    tensor = img_to_array(image, data_format=data_format)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def is_keras_pretrained_model(model):
    """
    Check if a model is on the keras pre-trained model list, i.e.,
    'inception_v3', 'mobilenet', 'resnet50', 'resnet101', 'resnet152', 'vgg16',
    'vgg19', 'xception'

    Parameters
    ----------
    model: string
        name of a model

    Returns
    -------
        boolean. `True` is the model is on the keras pre-trained model list and
        `False` otherwise
    """
    return model in config.PRETRAINED_MODEL_LIST


def get_pretrained_model(model, *args, **kwargs):
    """
    Return pre-trained model instance

    Parameters
    ----------
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'resnet101'
            'resnet152'
            'vgg16',
            'vgg19',
            'xception'
    *args: positioned arguments passed to pre-trained model class
    **kwargs: key-word arguments passed to pre-trained model class
    """
    assert is_keras_pretrained_model(model)
    if model in {'resnet101', 'resnet152'}:
        module = importlib.import_module('resnet')
    elif model == 'alexnet':
        module = importlib.import_module('jointuning.alexnet')
    else:
        module = importlib.import_module('keras.applications.{}'.format(model))
    model_class = getattr(module, config.PRETRAINED_MODEL_DICT[model])
    return model_class(*args, **kwargs)


def preprocess_an_image(image, model):
    """
    Wrapper around `keras.applications.{model}.preprocess_input()`

    Parameters
    ----------
    image: a 3D numpy array
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'resnet101',
            'resnet152',
            'vgg16',
            'vgg19',
            'xception'

    Returns
    -------
    A 3D numpy array (preprocessed image)
    """
    assert is_keras_pretrained_model(model) and image.ndim in {3, 4}
    if model in {'resnet101', 'resnet152'}:
        model = 'resnet50'
    module = importlib.import_module('keras.applications.{}'.format(model))
    preprocess_input = module.preprocess_input
    if image.ndim == 3:
        return preprocess_input(np.expand_dims(image, axis=0))[0]
    else:
        return preprocess_input(image)


def preprocess_input_wrapper(model):
    """
    Return a function that does input preprocess for pre-trained model and is
    compatible for use with `keras.preprocessing.image.ImageDataGenerator`'s
    `preprocessing_function` argument

    Parameters
    ----------
    model: model name, being one of
            'inception_v3',
            'mobilenet',
            'resnet50',
            'resnet101',
            'resnet152',
            'vgg16',
            'vgg19',
            'xception'
    """
    assert is_keras_pretrained_model(model)
    return partial(preprocess_an_image, model=model)


class LayerFilter(object):
    """Extract filters from a neural network layer

    Parameters
    ----------
    model: str/keras.models.Model
        name of a pre-trained model or a keras.models.Model instance
    layer_name: str
        name of a layer
    layer_index: int
        index of a layer
    """

    def __init__(self, model, layer_name=None, layer_index=None):
        if not isinstance(model, (Model, str)):
            raise ValueError('`model` must be either name of a pre-trained '
                             'model or a keras.models.Model instance')
        if isinstance(model, str):
            if not is_keras_pretrained_model(model):
                raise ValueError('`model` must be on the keras pre-trained ',
                                 'model list')
            self.model_name = model
            self.is_keras_pretrained_model = True
            model = get_pretrained_model(model)
        else:
            self.model_name = model.name
            self.is_keras_pretrained_model = False
        self.model = model
        self.layer = model.get_layer(name=layer_name, index=layer_index)
        uses_learning_phase = self.layer.output._uses_learning_phase
        input = model.input
        if uses_learning_phase and not isinstance(K.learning_phase(), int):
            input = [input, K.learning_phase()]
        else:
            input = [input]
        self.uses_learning_phase = uses_learning_phase
        self.input = model.input
        self.output = self.layer.output
        self.predict_function = K.function(input, [self.output])

    def predict(self, x):
        """Return the prediction of the output layer on input `x`

        Parameters
        ----------
        x: numpy array
            the input data

        Returns
        -------
            numpy array of predictions
        """
        assert isinstance(x, np.ndarray)
        if self.uses_learning_phase:
            x = [x, 0.]
        else:
            x = [x]
        return self.predict_function(x)[0]

    def predict_from_path(self, path,
                          target_size=None,
                          preprocessing_function=None):
        """Return the prediction of the output layer on an image given its path

        Parameters
        ----------
        path: str
            path to the input image
        target_size: tuple/list
            the target size when reading in the impage and perform resizing. If
            model is a pre-trained keras model, it could be left as `None`;
            otherwise it should be specified
        preprocessing_function: function
            input preprocess function. If model is a pre-trained keras model
            and it is left as `None`, the correct input preprocess function
            will be automatically retrieved to perform pre-processing.

        Returns
        -------
            numpy array of predictions
        """
        if not self.is_keras_pretrained_model:
            assert target_size is not None
        else:
            target_size = config.TARGET_SIZE_DICT[self.model_name]
        tensor = path_to_tensor(path, target_size=target_size)
        if preprocessing_function is None and self.is_keras_pretrained_model:
            preprocessing_function = preprocess_input_wrapper(self.model_name)
        if preprocessing_function is not None:
            tensor = preprocessing_function(tensor)
        return self.predict(tensor)


def filters_to_hist_features(filters, nbins=10,
                             data_format=None,
                             density=False):
    """Convolutional kernels or filters -> histogram features.
    See 'Image Descriptor' in Ge and Yu 2017 Borrowing Treasures from the
    Wealthy: Deep Transfer Learning through Selective Joint Fine-tuning

    Parameters
    ----------
    filters: a 4D numpy array
        if backend == 'tensorflow', (nsamples, kheight, kwidth, nkernels)
        if backend == 'theano', (nsamples, nkernels, kheight, kwidth)
    bins: int
        number of bins in each kernel. The location of the bins are set so that
        each of them contains a roughly equal percentage of pixels in that
        kernel over all the samples. See 'Image Descriptor' in Ge and Yu 2017
        paper for more details
    data_format: str
        either 'channels_first' (theano backend) or 'channels_last'
        (tensorflow)
    density: boolean (optional)
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function at
        the bin, normalized such that the integral over the range is 1

    Returns
    -------
        a 3D numpy array, (nsamples, nkernels, histogram)
    """
    assert isinstance(nbins, int)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_first', 'channels_last'}
    if data_format == 'channels_last':
        filters = np.moveaxis(filters, -1, 1)
    nsamples, nkernels, _, _ = filters.shape
    filters = filters.reshape(nsamples, nkernels, -1)
    collapsed = np.swapaxes(filters, 0, 1).reshape(nkernels, -1)
    percentiles = np.linspace(0, 100, num=nbins + 1)
    print('before getting bins')
    bin_edges = np.transpose(np.percentile(collapsed, percentiles, axis=1))
    print('after getting bins')
    print('before getting hist')
    hist = stats.histogram_varying_bins(filters, bin_edges, density=density)
    print('after getting hist')
    return hist


def get_image_descriptor(x, model,
                         layer_names=None,
                         layer_indices=None,
                         data_format=None,
                         nbins=100):
    """Extract image descriptor from input data

    See 'Image Descriptor' in Ge and Yu 2017 "Borrowing Treasures from the
    Wealthy: Deep Transfer Learning through Selective Joint Fine-tuning" for
    more details on how this feature extraction takes place

    Parameters
    ----------
    x: numpy array
        the input data to forward through a trained model to get the output of
        specified layers
    model: str/keras.models.Model
        name of a pre-trained model or a keras.models.Model instance
    layer_names: tuple/list
        layer name(s). If there is only one layer to extract features from,
        `layer_names` should be a tuple/list of length 1. If `layer_names` is
        provided, `layer_indices` should be left to default `None`
    layer_indices: tuple/list
        layer index/indices. If there is only one layer to extract features
        from, `layer_indices` should be a tuple/list of length 1. If
        `layer_indices` is provided, `layer_names` should be left to default
        `None`
    data_format: str
        either 'channels_first' (theano backend) or 'channels_last'
        (tensorflow)
    nbins: int
        number of bins in building histogram for each kernel

    Returns
    -------
        a numpy array of shape N x M, where N stands for number of images in
        the passed-in data and M = nbins x nkernels (nkernels is the total
        number of kernels in all the output layers)
    """
    assert isinstance(x, np.ndarray)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_first', 'channels_last'}
    if layer_names is None and layer_indices is None:
        raise ValueError('either `layer_names` or `layer_indices` needs to',
                         ' be specified')
    if layer_names is not None:
        layers = [LayerFilter(model, layer_name=name) for name in layer_names]
    else:
        layers = [LayerFilter(model, layer_index=idx) for idx in layer_indices]
    features = []
    for layer in layers:
        layer_output = layer.predict(x=x)
        hist_feat = filters_to_hist_features(
            layer_output, nbins, data_format=data_format, density=True)
        nkernels = hist_feat.shape[1]
        hist_feat = hist_feat.reshape(len(x), -1) / nkernels
        features.append(hist_feat)
    features = np.hstack(features)
    return features


def get_image_descriptor_from_dir(data_dir, model,
                                  layer_names=None,
                                  layer_indices=None,
                                  nbins=100,
                                  target_size=(224, 224),
                                  batch_size=100,
                                  classes=None,
                                  shuffle=False,
                                  seed=None,
                                  data_format=None,
                                  followlinks=False,
                                  preprocessing_function=None,
                                  image_ext_list=config.IMAGE_EXTENSIONS,
                                  save_to_pytables=True,
                                  pytables_path=None,
                                  group_name=None,
                                  feature_table_name='features',
                                  path_table_name='paths'):
    """Extract image descriptor from data directory with option to save to a
    pytables dataset on hard drive

    See 'Image Descriptor' in Ge and Yu 2017 "Borrowing Treasures from the
    Wealthy: Deep Transfer Learning through Selective Joint Fine-tuning" for
    more details on how this feature extraction takes place

    Parameters
    ----------
    data_dir: str
        path to the directory to read images from. Each subdirectory in
        this directory will be considered to contain images from one class, or
        alternatively you could specify class subdirectories via the `classes`
        argument.
    model: str/keras.models.Model
        name of a pre-trained model or a keras.models.Model instance
    layer_names: tuple/list
        layer name(s). If there is only one layer to extract features from,
        `layer_names` should be a tuple/list of length 1. If `layer_names` is
        provided, `layer_indices` should be left to default `None`
    layer_indices: tuple/list
        layer index/indices. If there is only one layer to extract features
        from, `layer_indices` should be a tuple/list of length 1. If
        `layer_indices` is provided, `layer_names` should be left to default
        `None`
    nbins: int
        number of bins in building histogram for each kernel
    target_size: tuple/list of length 2
        dimensions to resize input images to
    batch_size: int (optional)
        size of a batch to be forwarded through trained model to extract
        features at one time. Default is set to 1000
    classes: Optional list of strings, names of sudirectories
        containing images from each class (e.g. `["dogs", "cats"]`).
        It will be computed automatically if not set.
    shuffle: bool
        whether to shuffle the data
    seed: Random seed for data shuffling
    data_format: str (optional)
        either 'channels_first' (theano backend) or 'channels_last'
        (tensorflow). If left `None` (default), data format is guessed from
        `keras.backend.image_data_format()`
    followlinks: bool (optional)
        followlinks to `True` to visit directories pointed to by symlinks, on
        systems that support them
    preprocessing_function: callable
        input preprocess function. If model is a pre-trained keras model
        and it is left as `None`, the correct input preprocess function
        will be automatically retrieved to perform pre-processing
    image_ext_list: set/tuple/list
        set of strings containing common image extensions
    save_to_pytables: bool (optional)
        if `True` (default), extracted features are saved into a pytables
        dataset along with its path; otherwise the extracted features and its
        paths get returned
    pytables_path: str
        path to save the pytables dataset when `save_to_pytables=True`. Must be
        in hdf5 file extension
    group_name: str (optional)
        name of a group under which the features get saved. If left `None`
        (default), no group is created
    feature_table_name: str
        name of a table in which the features get saved. It needs to be
        specified when `save_to_pytables=True`
    path_table_name: str
        name of a table in which the paths get saved. It needs to be
        specified when `save_to_pytables=True`

    Returns
    -------
        if `save_to_pytables=False`, a tuple of following elements is returned
        (paths, features)
            - paths: images paths
            - features: a numpy array of shape N x M, N = nsamples,
              M = nbins x nkernels
        if `save_to_pytables=True`, features are saved in feature table and
        paths are saved in path table
    """
    dir_iter = directory.DirIterator(data_dir,
                                     target_size=target_size,
                                     classes=classes,
                                     class_mode='sparse',
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     seed=seed,
                                     data_format=data_format,
                                     followlinks=followlinks,
                                     image_ext_list=image_ext_list)
    start_time = time.time()
    nsamples = len(dir_iter.filepaths)
    if save_to_pytables:
        assert directory.file_of_extensions(pytables_path, {'hdf5'})
        hdf5_file = tables.open_file(pytables_path, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        if isinstance(group_name, str):
            location = hdf5_file.create_group(hdf5_file.root, group_name)
        else:
            location = hdf5_file.root
        path_storage = hdf5_file.create_earray(
            location, path_table_name,
            tables.Atom.from_dtype(dir_iter.filepaths.dtype),
            shape=(0,),
            filters=filters,
            expectedrows=nsamples)
    else:
        batch_features_list = []
    for idx in range(ceildiv(nsamples, batch_size)):
        print('---------------- New iteration -------------------')
        _, batch_paths, batch_x, _ = next(dir_iter)
        print('Got batch')
        if (preprocessing_function is not None and
                callable(preprocessing_function)):
            batch_x = preprocessing_function(batch_x)
        batch_features = get_image_descriptor(batch_x, model,
                                              layer_names=layer_names,
                                              layer_indices=layer_indices,
                                              data_format=data_format,
                                              nbins=nbins)
        print('Got features')
        if save_to_pytables:
            print('Started writing batch of features')
            if idx == 0:
                features_storage = hdf5_file.create_earray(
                    location, feature_table_name,
                    tables.Atom.from_dtype(batch_features.dtype),
                    shape=(0, batch_features.shape[-1]),
                    filters=filters,
                    expectedrows=nsamples)
            features_storage.append(batch_features)
            path_storage.append(batch_paths)
            print('Finished writing batch of features')
        else:
            batch_features_list.append(batch_features)
    if save_to_pytables:
        hdf5_file.close()
        print('It took {} seconds'.format(time.time() - start_time))
    else:
        features = np.vstack(batch_features_list)
        return dir_iter.filepaths, features
