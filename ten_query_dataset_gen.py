import tarfile
import h5py
from ann_benchmarks import datasets

TRAIN_SIZE = 10000
QUERY_NUM = 10


def my_write_output(train, test, out_fn, distance, point_type='float', count=100):
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
    n = 0
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = distance
    f.attrs['point_type'] = point_type
    print('train size: %9d * %4d' % train.shape)
    print('test size:  %9d * %4d' % test.shape)
    # f.create_dataset('train', (len(train), len(
    #     train[0])), dtype=train.dtype)[:] = train
    # f.create_dataset('test', (len(test), len(
    #     test[0])), dtype=test.dtype)[:] = test
    # neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    # distances = f.create_dataset('distances', (len(test), count), dtype='f')
    f.create_dataset('train', (TRAIN_SIZE, len(
        train[0])), dtype=train.dtype)[:] = train[:TRAIN_SIZE]
    f.create_dataset('test', (QUERY_NUM, len(
        test[0])), dtype=test.dtype)[:] = test[:QUERY_NUM]
    neighbors = f.create_dataset('neighbors', (QUERY_NUM, count), dtype='i')
    distances = f.create_dataset('distances', (QUERY_NUM, count), dtype='f')
    bf = BruteForceBLAS(distance, precision=train.dtype)
    train = datasets.dataset_transform[distance](train)
    test = datasets.dataset_transform[distance](test)
    bf.fit(train[:TRAIN_SIZE])
    queries = []
    for i, x in enumerate(test[:QUERY_NUM]):
        if i % 1000 == 0:
            print('%d/%d...' % (i, len(test[:QUERY_NUM])))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()


def my_sift(fn, out_fn):
    with tarfile.open(fn, 'r:gz') as t:
        train = datasets._get_irisa_matrix(t, 'sift/sift_base.fvecs')
        test = datasets._get_irisa_matrix(t, 'sift/sift_query.fvecs')
        my_write_output(train, test, out_fn, 'euclidean')


def main():
    fn = 'data/sift.tar.gz'
    out_fn = 'data/sift-%d-%d.hdf5' % (TRAIN_SIZE, QUERY_NUM)
    print("fn: ", fn, ", out_fn: ", out_fn)
    my_sift(fn, out_fn)


if __name__ == "__main__":
    main()
