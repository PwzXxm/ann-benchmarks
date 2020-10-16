#!/usr/bin/env python

from __future__ import absolute_import
import h5py
import os
import numpy as np
from ann_benchmarks import datasets


def load_npy(file_radix):
    file_name=file_radix+'.npy'
    if os.path.exists(file_name):
        arr = np.load(file_name)
        return arr


def load_ans(file):
    ground = open(file, 'r')
    ground_list = []
    lines = ground.readlines()
    for l in lines:
        line = l.strip().split(',')
        for u in range(len(line)):
            line[u] = int(line[u])
        ground_list.append(line)
    return ground_list


TRAIN_SIZE = 0
QUERY_NUM = 1000
array = load_npy("/Users/yuxingyuan/query")
answer = load_ans("/Users/yuxingyuan/top_50_select_1.txt")


def my_write_output(train, test, out_fn, distance, point_type='float', count=50):
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
    n = 0
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = distance
    f.attrs['point_type'] = point_type
    print('train size: %9d * %4d' % train.shape)
    print('test size:  %9d * %4d' % test.shape)
    f.create_dataset('train', (QUERY_NUM, len(
        train[0])), dtype=train.dtype)[:] = train[:QUERY_NUM]
    f.create_dataset('test', (QUERY_NUM, len(
        test[0])), dtype=test.dtype)[:] = test[:QUERY_NUM]
    neighbors = f.create_dataset('neighbors', (QUERY_NUM, count), dtype='i')
    distances = f.create_dataset('distances', (QUERY_NUM, count), dtype='f')
    bf = BruteForceBLAS(distance, precision=train.dtype)
    train = datasets.dataset_transform[distance](train)
    test = datasets.dataset_transform[distance](test)
    bf.fit(train[:TRAIN_SIZE])
    neighbors = answer
    distances= []
    f.close()



my_write_output(array, array, "query.h5", "euclidean")