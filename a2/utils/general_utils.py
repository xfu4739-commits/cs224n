#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2021-2022: Homework 3
general_utils.py: 通用工具函数
Sahil Chopra <schopra8@stanford.edu>
"""

import numpy as np


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    将数据按 minibatch 迭代返回。你可以用该函数以 minibatch 的方式遍历数据，例如：

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    也可以用于多个数据源同时迭代：

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: 两种形式之一：
            - list 或 numpy array
            - list，其中每个元素本身是 list 或 numpy array（用于多数据源）
        minibatch_size: 每个 minibatch 最多包含的样本数
        shuffle: 是否在返回前打乱数据顺序
    Returns:
        minibatches: 返回值取决于 `data` 的形式：
            - 如果 `data` 是 list/array，则每次 yield 一份数据的下一个 minibatch。
            - 如果 `data` 是「list of lists/arrays」，则每次返回该 list 中每个元素的下一个 minibatch。
              这可以用于同时遍历多个数据源（例如特征与标签）。

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [_minibatch(d, minibatch_indices) for d in data] if list_data \
            else _minibatch(data, minibatch_indices)


def _minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def test_all_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("{:} failed, expected output to have shape {:} but has shape {:}"
                         .format(name, expected.shape, actual.shape))
    if np.amax(np.fabs(actual - expected)) > 1e-6:
        raise ValueError("{:} failed, expected {:} but value is {:}".format(name, expected, actual))
    else:
        print(name, "passed!")
