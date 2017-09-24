# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import numpy as np


def integer_replace(int_array, old, new):
    """Replace old value(s) in an array of integers

    Parameters
    ----------
    int_array: tuple/list/np.ndarray
        an array of integers
    old: int/tuple/list/np.ndarray
        old values to be replaced
    new: int/tuple/list/np.ndarray
        new values to replace

    Returns
    -------
    an array with values replaced
    """
    assert isinstance(int_array, (tuple, list, np.ndarray))
    if not isinstance(int_array, np.ndarray):
        int_array = np.array(int_array)
    indices = np.arange(int_array.max() + 1)
    indices[old] = new
    return indices[int_array]


def bincount_nd(x, minlength=None):
    """Multi-dimensional version of `numpy.bincount` with shared `minlength`
    """
    x_max = x.max()
    shape = x.shape
    higher_dim = np.prod(shape[:-1], dtype=np.int32)
    if minlength is None or minlength <= x_max:
        minlength = x_max + 1
    offset = (minlength * np.arange(higher_dim)).reshape(shape[:-1])[..., None]
    x_offset = x + offset
    counts = np.bincount(x_offset.ravel(),
                         minlength=(minlength * higher_dim))
    counts.shape = shape[:-1] + (minlength, )
    return counts


def histogram_2d_array(array, bin_edges, axis=1, density=False):
    """Histogram for 2D array along axis

    Parameters
    ----------
    array: 2D numpy array
    bin_edges: tuple/list
        an increasing sequence of bin edges. Values in `array` must be within
        the interval spanned by the left-most and right-most edges
    axis: int (optional)
        along which axis the histogram is computed. Either 0 or 1
    density: bool (optional)
        If `False`, the result will contain the number of samples in each bin.
        If `True`, the result is the value of the probability density function
        at the bin. Default is set to `False`

    Returns:
    --------
    A numpy array
        if `axis=0`, returned array has shape of (array.shape[1], nbins)
        if `axis=1`, returned array has shape of (len(array), nbins)
    """
    assert (isinstance(bin_edges, (tuple, list, np.ndarray)) and
            isinstance(axis, int) and
            isinstance(density, bool))
    if axis == 0:
        array = np.swapaxes(array, 0, 1)
    nrows, ncols = array.shape
    nbins = len(bin_edges) - 1
    indices = np.searchsorted(bin_edges, array, side='right') - 1
    indices[indices == nbins] = nbins - 1
    indices_offset = nbins * np.arange(nrows)[:, None] + indices
    counts = np.bincount(indices_offset.ravel(), minlength=(nbins * nrows))
    if density:
        counts = counts / ncols
    counts.shape = (nrows, nbins)
    return counts


def histogram_2d_array_varying_bins(array, bin_edges, axis=1, density=False):
    """Histogram of 2D array with varying bin edges

    Both `array` and `bin_edges` are 2D arrays and match their dimension on
    the axis other than `axis`, values of which will be used to calculate
    histogram. Histogram calculation is conducted over the matched dimension

    Parameters
    ----------
    array: 2D numpy array
    bin_edges: 2D numpy array
        increasing bin edge sequence along `axis`
    axis: int (optional)
        along which axis the histogram is computed. Either 0 or 1. Default is
        set to 1 (row dimension). Each bin edge sequence are of same length.
        Values in `array` along `axis` will be within the range of its
        corresponding bin edges
    density: bool (optional)
        If `False`, the result will contain the number of samples in each bin.
        If `True`, the result is the value of the probability density function
        at the bin. Default is set to `False`

    Returns
    -------
    A numpy array
        if `axis=0`, returned array has shape of (array.shape[1], nbins)
        if `axis=1`, returned array has shape of (len(array), nbins)
    """
    assert (isinstance(array, np.ndarray) and
            isinstance(bin_edges, np.ndarray) and
            isinstance(axis, int) and
            isinstance(density, bool))
    if axis == 0:
        array = np.swapaxes(array, 0, 1)
        bin_edges = np.swapaxes(bin_edges, 0, 1)
    nrows, ncols = array.shape
    nbins = bin_edges.shape[1] - 1
    rng = bin_edges.max() - bin_edges.min() + 1
    offset = rng * np.arange(nrows)[:, None]
    indices = np.searchsorted((bin_edges + offset).ravel(),
                              (array + offset).ravel(), side='right')
    right_most_idx = np.arange(nbins + 1, nrows * nbins + nrows + 1, nbins + 1)
    indices = integer_replace(indices, right_most_idx, right_most_idx - 1)
    indices = indices - np.repeat(np.arange(1, nrows + 1), ncols)
    counts = np.bincount(indices, minlength=(nbins * nrows))
    if density:
        counts = counts / ncols
    counts.shape = (nrows, nbins)
    return counts


def histogram_varying_bins(array, bin_edges,
                           varying_bin_axis=1, density=False):
    """Histogram of array of 2D or higher with varying bin edges

    Histograms are calculated along the last axis of `array`

    Parameters
    ----------
    array: numpy array of 2D or higher
    bin_edges: 2D numpy array
        rows of `bin_edges` are increasing bin edge sequences.
    varying_bin_axis: int (optional)
        `array.shape[varying_bin_axis]` == `bin_edges.shape[0]`
    density: bool (optional)
        If `False`, the result will contain the number of samples in each bin.
        If `True`, the result is the value of the probability density function
        at the bin. Default is set to `False`

    Returns
    -------
    A numpy array
        returned array will match the dimensions of `array` except the last,
        where the returned array have dimension of nbins. That is, the returned
        array will have shape of `array.shape[:-1] + (nbins, )`
    """
    assert (isinstance(array, np.ndarray) and
            isinstance(bin_edges, np.ndarray) and
            isinstance(density, bool))
    array = np.swapaxes(array, 0, varying_bin_axis)
    array_shape = tuple(array.shape)
    array_reshaped = array.reshape((len(array), -1))
    nrows, ncols = array_reshaped.shape
    nbins = bin_edges.shape[1] - 1
    rng = bin_edges.max() - bin_edges.min() + 1
    offset = rng * np.arange(nrows)[:, None]
    indices = np.searchsorted((bin_edges + offset).ravel(),
                              (array_reshaped + offset).ravel(), side='right')
    right_most_idx = np.arange(nbins + 1, nrows * nbins + nrows + 1, nbins + 1)
    indices = integer_replace(indices, right_most_idx, right_most_idx - 1)
    indices = indices - np.repeat(np.arange(1, nrows + 1), ncols)
    indices.shape = array_shape
    correction = nbins * np.arange(nrows)
    correction.shape = (nrows, ) + (1, ) * (len(array_shape) - 1)
    indices = indices - correction
    indices = np.swapaxes(indices, 0, varying_bin_axis)
    counts = bincount_nd(indices, minlength=nbins)
    if density:
        counts = counts / array_shape[-1]
    return counts
