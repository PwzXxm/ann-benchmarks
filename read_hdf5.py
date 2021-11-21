import numpy as np 
import h5py
import os

fname_h5 = 'data/deep-96-angular.hdf5'

def simple_test_hdf5(fname_h5):
    with h5py.File(fname_h5, 'a') as f:
        # train = np.array(f['train'])
        # print(train.shape)
        # test = np.array(f['test'])
        # print(test.shape)
        # neighbors = np.array(f['neighbors'])
        # print(neighbors.shape)
        # distance = np.array(f['distance'])
        # print(distance.shape)

        f.attrs['distance'] = 'angular'
        f.attrs['point_type'] = 'float'

        print(f.attrs)
        print(dict(f.attrs))
        print(list(f.attrs.keys()))

simple_test_hdf5(fname_h5)