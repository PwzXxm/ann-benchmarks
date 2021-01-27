#!/usr/bin/env python

from __future__ import absolute_import
import h5py
import numpy
from ann_benchmarks.algorithms.vearch_320 import VearchIVFFLAT


def compute_recall(std, answer):
    hit_nums = 0.0
    for neighbor in answer:
        if neighbor in std:
            hit_nums += 1
    return hit_nums / len(answer)


dataset = 'sift-10000-10'


# dataset = 'sift-10m'


def test_ivfflat():
    ncentroids = 128
    client = VearchIVFFLAT(ncentroids)
    base = []
    for i in range(8192):
        base.append([1, 1 + i])
    vectors = numpy.asarray(base)
    # client.done()
    client.fit(vectors)

    client.set_query_arguments(10)

    querys1 = numpy.asarray([[1, 2]])
    querys2 = numpy.asarray([[1, 99]])
    n = 100

    client.batch_query1(querys1, querys2, n)
    ids = client.get_batch_results()
    print(ids)
    print(len(ids[0]))

    client.done()


if __name__ == "__main__":
    test_ivfflat()
