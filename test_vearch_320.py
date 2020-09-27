#!/usr/bin/env python

from __future__ import absolute_import
import h5py
import numpy
from ann_benchmarks.algorithms.vearch_320 import VearchIVFPQ, VearchHNSW, VearchIVFFLAT


def compute_recall(std, answer):
    hit_nums = 0.0
    for neighbor in answer:
        if neighbor in std:
            hit_nums += 1
    return hit_nums / len(answer)


dataset = 'sift-10000-10'


def test_hnsw():
    nlinks = 32
    efConstruction = 40
    client = VearchHNSW(nlinks, efConstruction)
    f = h5py.File('data/' + dataset + '.hdf5', 'r')
    vectors = numpy.array(f['train'])
    client.fit(vectors)
    qs = numpy.array(f['test'])
    topk = 10
    efSearch = 64
    client.set_query_arguments(efSearch)
    client.batch_query(qs, topk)
    ids = client.get_batch_results()
    # print(ids)
    stds = numpy.array(f['neighbors'])
    recall = 0.0
    for i in range(len(ids)):
        print("recall: ", compute_recall(stds[i], ids[i]))
        recall += compute_recall(stds[i], ids[i])
    print("average recall: ", recall / len(ids))
    client.done()
    f.close()


def test_ivfpq():
    ncentroids = 8192
    client = VearchIVFPQ(ncentroids)
    f = h5py.File('data/' + dataset + '.hdf5', 'r')
    vectors = numpy.array(f['train'])
    client.fit(vectors)
    qs = numpy.array(f['test'])
    topk = 10
    nprobe = 200
    client.set_query_arguments(nprobe)
    client.batch_query(qs, topk)
    ids = client.get_batch_results()
    # print(ids)
    stds = numpy.array(f['neighbors'])
    recall = 0.0
    for i in range(len(ids)):
        print("recall: ", compute_recall(stds[i], ids[i]))
        recall += compute_recall(stds[i], ids[i])
    print("average recall: ", recall / len(ids))
    client.done()
    f.close()


def test_ivfflat():
    ncentroids = 8192
    client = VearchIVFFLAT(ncentroids)
    f = h5py.File('data/' + dataset + '.hdf5', 'r')
    vectors = numpy.array(f['train'])
    client.fit(vectors)
    qs = numpy.array([f['test'][0]])
    topk = 100
    nprobe = 200
    client.set_query_arguments(nprobe)
    client.batch_query(qs, topk)
    ids = client.get_batch_results()
    # print(ids)
    stds = numpy.array(f['neighbors'])
    recall = 0.0
    for i in range(len(ids)):
        print("recall: ", compute_recall(stds[i], ids[i]))
        recall += compute_recall(stds[i], ids[i])
    print("average recall: ", recall / len(ids))
    client.done()
    f.close()


if __name__ == "__main__":
    # test_hnsw()
    # test_ivfpq()
    test_ivfflat()
