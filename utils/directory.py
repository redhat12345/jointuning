# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os
import errno
import imghdr
import multiprocessing.pool
import numpy as np
from functools import partial

from .. import config
from . import data

from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import (
    Iterator, _count_valid_files_in_directory, array_to_img)


def create_dir(dirpath):
    """
    Create a directory if it does not exist
    """
    try:
        os.makedirs(os.path.expanduser(dirpath))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        elif not os.path.isdir(dirpath):
            print(('Warning: `' + dirpath + '` already exists '
                   'and is not a directory!'))


def file_of_extensions(filepath, ext_list):
    """
    Check the extension of a file is one of given list of extensions
    """
    _, ext = os.path.splitext(filepath)
    return (ext[1:]).lower() in ext_list


def files_under_dir(dirpath, followlinks=False):
    """Return a list of all files in a given directory and its subdirectories
    """
    abs_path = os.path.abspath(os.path.expanduser(dirpath))
    files = [os.path.join(root, file)
             for root, _, files in os.walk(abs_path, followlinks=followlinks)
             for file in files]
    return sorted(files)


def files_under_subdirs(dirpath, subdirs=None, followlinks=False):
    """
    Return a list of all files in a given directory and specified subdirs
    """
    abs_path = os.path.abspath(os.path.expanduser(dirpath))
    if not subdirs:
        subdirs = os.listdir(abs_path)
    subdirs = [os.path.basename(subdir) for subdir in subdirs]
    files = [file for subdir in subdirs
             for file in files_under_dir(os.path.join(abs_path, subdir),
                                         followlinks=followlinks)]
    return sorted(files)


def images_under_dir(dirpath,
                     examine_by='extension',
                     ext_list=config.IMAGE_EXTENSIONS,
                     followlinks=False):
    """
    Return a list of image files in a given dir and its subdirs

    Parameters
    ----------
    dirpath: string or unicode
        path to the directory
    examine_by: string (default='extension')
        method of examining image file, either 'content' or 'extension'
        If 'content', `imghdr.what()` is used
        If 'extension', the file extension is compared against a list of common
        image extensions
    ext_list: list, optional (used only when examine_by='extension')
        If examine_by='extension', the file extension is compared against
        ext_list
    """
    files = files_under_dir(dirpath, followlinks=followlinks)
    assert examine_by in ['content', 'extension']
    if examine_by == 'content':
        images = [file for file in files if imghdr.what(file)]
    else:
        images = [file for file in files if file_of_extensions(file, ext_list)]
    return images


def images_under_subdirs(dirpath,
                         subdirs=None,
                         examine_by='extension',
                         ext_list=config.IMAGE_EXTENSIONS,
                         followlinks=False):
    """
    Return a list of image files in a given dir and specified subdirs
    """
    files = files_under_subdirs(dirpath, subdirs, followlinks=followlinks)
    assert examine_by in ['content', 'extension']
    if examine_by == 'content':
        images = [file for file in files if imghdr.what(file)]
    else:
        images = [file for file in files if file_of_extensions(file, ext_list)]
    return images


def list_valid_filepaths_in_directory(dirpath, white_list_formats,
                                      class_indices, followlinks=False):
    """List absolute paths of files in `directory`

    Parameters
    -----------
    dirpath: absolute path to a directory containing the files to list
        The directory name is used as class label and must be a key of
        `class_indices`
    white_list_formats: set of strings containing allowed extensions for
        the files to be counted
    class_indices: dictionary mapping a class name to its index

    Returns
    -------
        classes: a list of class indices
        filepaths: the absolute path of valid files in a directory (e.g.,
            if `dirpath` is "path_to_dataset/class1", the filepaths will be
            ["path_to_dataset/class1/file1.jpg", ...])
    """
    filepaths = images_under_dir(
        dirpath, ext_list=white_list_formats, followlinks=followlinks)
    dirname = os.path.basename(dirpath)
    classes = [class_indices[dirname]] * len(filepaths)
    return classes, filepaths


class DirIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    Modifies keras.preprocessing.image.DirectoryIterator with following changes
        - outputs a batch of (index, path, x, y) instead of (x, y)
        - `image_data_generator` can be `None`, in which case no random
          transformations and normalization are applied

    Parameters
    ----------
    directory: Path to the directory to read images from. Each subdirectory in
        this directory will be considered to contain images from one class, or
        alternatively you could specify class subdirectories via the `classes`
        argument.
    image_data_generator: Instance of `ImageDataGenerator`
        to use for random transformations and normalization.
    target_size: tuple of integers, dimensions to resize input images to.
    color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
    classes: Optional list of strings, names of sudirectories
        containing images from each class (e.g. `["dogs", "cats"]`).
        It will be computed automatically if not set.
    class_mode: Mode for yielding the targets:
        `"binary"`: binary targets (if there are only two classes),
        `"categorical"`: categorical targets,
        `"sparse"`: integer targets,
        `"input"`: targets are images identical to input images (mainly
            used to work with autoencoders),
        `None`: no targets get yielded (only input images are yielded).
    batch_size: Integer, size of a batch.
    shuffle: Boolean, whether to shuffle the data between epochs.
    seed: Random seed for data shuffling.
    data_format: String, one of `channels_first`, `channels_last`.
    save_to_dir: Optional directory where to save the pictures
        being yielded, in a viewable format. This is useful
        for visualizing the random transformations being
        applied, for debugging purposes.
    save_prefix: String prefix to use for saving sample
        images (if `save_to_dir` is set).
    save_format: Format to use for saving sample images
        (if `save_to_dir` is set).
    """

    def __init__(self, directory,
                 image_data_generator=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 followlinks=False,
                 image_ext_list=config.IMAGE_EXTENSIONS):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=image_ext_list,
                                   follow_links=followlinks)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found {0} images belonging to {1} classes.'.format(
            self.samples, self.num_class))

        # second, build an index of the images in different class subfolders
        results = []
        self.filepaths = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(list_valid_filepaths_in_directory,
                                            (dirpath, image_ext_list,
                                             self.class_indices,
                                             followlinks)))
        for res in results:
            classes, filepaths = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filepaths += filepaths
            i += len(classes)
        self.filepaths = np.array(self.filepaths)
        pool.close()
        pool.join()
        super(DirIterator, self).__init__(self.samples, batch_size,
                                          shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        batch_filepaths = self.filepaths[index_array]
        batch_x_list = [data.path_to_tensor(path, self.target_size,
                                            grayscale, self.data_format)
                        for path in batch_filepaths]
        if self.image_data_generator is not None:
            batch_x_list = [self.image_data_generator.standardize(
                            self.image_data_generator.random_transform(x))
                            for x in batch_x_list]
        batch_x = np.vstack(batch_x_list)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=current_index + i,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = self.classes[index_array]
            batch_y = np_utils.to_categorical(batch_y, self.num_class)
        else:
            return batch_x
        return range(current_batch_size), batch_filepaths, batch_x, batch_y
