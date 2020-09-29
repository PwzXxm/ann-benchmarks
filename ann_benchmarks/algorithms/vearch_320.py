from __future__ import absolute_import
import json
import time
import numpy
import requests
from ann_benchmarks.algorithms.base import BaseANN


def _check_response(response):
    if response.status_code != 200:
        # print(response.text)
        raise Exception('status is not expected')


def _name_in_dict_list(dlist, name):
    for d in dlist:
        if d['name'] == name:
            return True
    return False


class Vearch(BaseANN):
    def __init__(self):
        self._db_name = 'annbench'
        self._table_name = 'annbench1'
        self._field = 'field1'
        # self._master_host = '172.16.0.251'
        # self._master_port = '443'
        self._master_host = 'localhost'
        self._master_port = '8817'  # docker
        # self._router_host = '172.16.0.251'
        # self._router_port = '80'
        self._router_host = 'localhost'
        self._router_port = '9001'  # docker
        self._master_prefix = 'http://' + self._master_host + ':' + self._master_port
        self._router_prefix = 'http://' + self._router_host + ':' + self._router_port

    def _drop_db(self):
        url = self._master_prefix + '/db/' + self._db_name
        response = requests.delete(url)
        print("delete: ", url, ", status: ", response.status_code)
        _check_response(response)

    def _drop_table(self):
        url = self._master_prefix + '/space/' + self._db_name + '/' + self._table_name
        response = requests.delete(url)
        print("delete: ", url, ", status: ", response.status_code)
        _check_response(response)

    def _db_exists(self):
        url = self._master_prefix + '/list/db'
        response = requests.get(url)
        _check_response(response)
        if not response.json()['data']:
            return False
        return _name_in_dict_list(response.json()['data'], self._db_name)

    def _table_exists(self):
        if not self._db_exists():
            return False
        url = self._master_prefix + '/list/space?db=' + self._db_name
        response = requests.get(url)
        _check_response(response)
        if not response.json()['data']:
            return False
        return _name_in_dict_list(response.json()['data'], self._table_name)

    def _create_db(self):
        if self._db_exists():
            if self._table_exists():
                self._drop_table()
            self._drop_db()
        url = self._master_prefix + '/db/_create'
        response = requests.put(url, json={"name": self._db_name})
        print("put: ", url, ", status: ", response.status_code)
        _check_response(response)

    def _create_table(self, payload):
        if self._table_exists():
            self._drop_table()
        url = self._master_prefix + '/space/' + self._db_name + '/_create'
        response = requests.put(url, json=payload)
        print("create table: ", url)
        retrieval_type = payload['engine']['retrieval_type']
        retrieval_param = payload['engine']['retrieval_param']
        print("retrieval_type: ", retrieval_type)
        for key, value in retrieval_param.items():
            print(key, ": ", value)
        print("status: ", response.status_code)
        _check_response(response)

    def _bulk_insert(self, X):
        url = self._router_prefix + '/' + self._db_name + '/' + self._table_name + '/_bulk'
        records_len = len(X)
        step = 2000
        for i in range(0, records_len, step):
            end = min(i + step, records_len)
            docs = ""
            for j in range(i, end):
                docs += json.dumps({
                    "index": {
                        "_id": j
                    }
                }) + "\n"
                docs += json.dumps({
                    self._field: {
                        "feature": X[j].tolist()
                    }
                }) + "\n"
            response = requests.request("POST", url, headers={"Content-Type": "application/json"}, data=docs)
            print("bulk insert docs: ", url, ", status: ", response.status_code)
            _check_response(response)

    def _single_insert(self, X):
        records_len = len(X)
        for i in range(records_len):
            url = self._router_prefix + '/' + self._db_name + '/' + self._table_name + '/' + str(i)
            doc = {
                self._field: {
                    "feature": X[i].tolist()
                }
            }
            print("post: ", url)
            response = requests.post(url, json=doc)
            _check_response(response)

    def _batch_query_with_payload(self, payload):
        self._res = []
        url = self._router_prefix + '/' + self._db_name + '/' + self._table_name + '/_msearch'
        response = requests.post(url, json=payload)
        print("query: ", url, ", status: ", response.status_code)
        _check_response(response)
        if response.json():
            # print(response.json())
            def _print_all_key(kv, indent=0):
                if not isinstance(kv, dict):
                    return
                for key, value in kv.items():
                    print('\t' * indent + key + ':')
                    if isinstance(value, dict):
                        _print_all_key(value, indent + 1)
                    if isinstance(value, list) and value:
                        _print_all_key(value[0], indent + 1)

            self._res = [[int(hit['_id']) for hit in results['hits']['hits']]
                         for results in response.json()['results']]

    def get_batch_results(self):
        return self._res

    def _create_index(self):
        url = self._router_prefix + '/' + self._db_name + '/' + self._table_name + '/_forcemerge'
        response = requests.post(url)
        print("create index : ", url, ", status: ", response.status_code)
        _check_response(response)
        if not response.json()['_shards']:
            return False
        return response.json()['_shards']["successful"] == 1

    def done(self):
        self._drop_table()
        self._drop_db()
        return

    def __str__(self):
        return "Vearch"


class VearchIVFPQ(Vearch):
    def __init__(self, metric_type, ncentroids, nsubvector=64, partition_num=1, replica_num=1, nbits_per_idx=8):
        Vearch.__init__(self)
        self._ncentroids = ncentroids
        self._nsubvector = 64
        self._partition_num = partition_num
        self._replica_num = replica_num
        if metric_type == 'L2':
            self._metric_type = 'L2'
        else:
            self._metric_type = 'InnerProduct'
        self._nbits_per_idx = nbits_per_idx

    def fit(self, X):
        self._create_db()
        max_size = X.shape[0]
        index_size = min(max_size, self._ncentroids * 128)
        dimension = X.shape[1]
        retrieval_type = "IVFPQ"
        retrieval_param = {
            "ncentroids": self._ncentroids,
            "nsubvector": self._nsubvector,
            "nbits_per_idx": self._nbits_per_idx,
            "metric_type": self._metric_type,
        }
        payload = {
            "name": self._table_name,
            "partition_num": self._partition_num,
            "replica_num": self._replica_num,
            "engine": {
                "name": "gamma",
                "index_size": index_size,
                "max_size": max_size,
                "retrieval_type": retrieval_type,
                "retrieval_param": retrieval_param
            },
            "properties": {
                self._field: {
                    "type": "vector",
                    "index": True,
                    "dimension": dimension,
                    "store_type": "MemoryOnly",
                }
            }
        }
        self._create_table(payload)
        self._bulk_insert(X)
        # self._single_insert(X)
        self._create_index()

    def set_query_arguments(self, nprobe):
        self._nprobe = min(nprobe, self._ncentroids)

    def batch_query(self, X, n):
        features = []
        for vector in X:
            features += vector.tolist()
            # features.append(vector.tolist())
        payload = {
            "query": {
                "sum": [{
                    "field": self._field,
                    "feature": features,
                }]
            },
            "size": n,
            "sort": [{
                "_score": {"order": "asc"}
            }],
            "retrieval_params": {
                "parallel_on_queries": 0,
                "recall_num": n,  # should equal to size here
                "nprobe": self._nprobe,
                "metric_type": self._metric_type
            }
        }
        self._batch_query_with_payload(payload)

    def __str__(self):
        return ("VearchIVFPQ" +
                ", ncentroids: " + str(self._ncentroids) +
                ", nprobe: " + str(self._nprobe) +
                ", nsubvector: " + str(self._nsubvector) +
                # ", master: " + self._master_prefix +
                # ", router: " + self._router_prefix +
                # ", partition_num: " + str(self._partition_num) +
                # ", replica_num: " + str(self._replica_num) +
                ", metric_type: " + str(self._metric_type) +
                ", nbits_per_idx: " + str(self._nbits_per_idx))


class VearchIVFFLAT(Vearch):
    def __init__(self, metric_type, ncentroids, partition_num=1, replica_num=1):
        Vearch.__init__(self)
        self._ncentroids = ncentroids
        self._partition_num = partition_num
        self._replica_num = replica_num
        if metric_type == 'L2':
            self._metric_type = 'L2'
        else:
            self._metric_type = 'InnerProduct'

    def fit(self, X):
        self._create_db()
        max_size = X.shape[0]
        index_size = min(max_size, self._ncentroids * 128)
        dimension = X.shape[1]
        retrieval_type = "IVFFLAT"
        retrieval_param = {
            "ncentroids": self._ncentroids,
            "metric_type": self._metric_type,
            # "nprobe": 80, # vearch default, what a shit here!
        }
        payload = {
            "name": self._table_name,
            "partition_num": self._partition_num,
            "replica_num": self._replica_num,
            "engine": {
                "name": "gamma",
                "index_size": index_size,
                "max_size": max_size,
                "retrieval_type": retrieval_type,
                "retrieval_param": retrieval_param
            },
            "properties": {
                self._field: {
                    "type": "vector",
                    "index": True,
                    "dimension": dimension,
                    "store_type": "RocksDB",
                }
            }
        }
        self._create_table(payload)
        self._bulk_insert(X)
        self._create_index()

    def set_query_arguments(self, nprobe):
        self._nprobe = min(nprobe, self._ncentroids)

    def batch_query(self, X, n):
        features = []
        for vector in X:
            # features.append(vector.tolist())
            features += vector.tolist()
        payload = {
            "query": {
                "sum": [{
                    "field": self._field,
                    "feature": features,
                }]
            },
            "size": n,
            "sort": [{
                "_score": {"order": "asc"}
            }],
            "retrieval_params": {
                "parallel_on_queries": 0,
                "nprobe": self._nprobe,
                "metric_type": self._metric_type
            }
        }
        self._batch_query_with_payload(payload)

    def __str__(self):
        return ("VearchIVFFLAT" +
                ", ncentroids: " + str(self._ncentroids) +
                ", nprobe: " + str(self._nprobe) +
                # ", master: " + self._master_prefix +
                # ", router: " + self._router_prefix +
                # ", partition_num: " + str(self._partition_num) +
                # ", replica_num: " + str(self._replica_num) +
                ", metric_type: " + str(self._metric_type))


class VearchHNSW(Vearch):
    def __init__(self, metric_type, nlinks, efConstruction, partition_num=1, replica_num=1):
        Vearch.__init__(self)
        self._partition_num = partition_num
        self._replica_num = replica_num
        if metric_type == 'L2':
            self._metric_type = 'L2'
        else:
            self._metric_type = 'InnerProduct'
        self._nlinks = nlinks
        self._efConstruction = efConstruction

    def fit(self, X):
        self._create_db()
        max_size = X.shape[0]
        index_size = 2
        dimension = X.shape[1]
        retrieval_type = "HNSW"
        retrieval_param = {
            "nlinks": self._nlinks,
            "efConstruction": self._efConstruction,
            "metric_type": self._metric_type,
            # "efSearch": 64, # vearch default, what a shit here!
        }
        payload = {
            "name": self._table_name,
            "partition_num": self._partition_num,
            "replica_num": self._replica_num,
            "engine": {
                "name": "gamma",
                "index_size": index_size,
                "max_size": max_size,
                "retrieval_type": retrieval_type,
                "retrieval_param": retrieval_param
            },
            "properties": {
                self._field: {
                    "type": "vector",
                    "index": True,
                    "dimension": dimension,
                    "store_type": "MemoryOnly",
                }
            }
        }
        self._create_table(payload)
        self._bulk_insert(X)
        # self._single_insert(X)
        self._create_index()

    def set_query_arguments(self, efSearch):
        self._efSearch = efSearch

    def batch_query(self, X, n):
        features = []
        for vector in X:
            features += vector.tolist()
        payload = {
            "query": {
                "sum": [{
                    "field": self._field,
                    "feature": features,
                }]
            },
            "size": n,
            "sort": [{
                "_score": {"order": "asc"}
            }],
            "retrieval_params": {
                "metric_type": self._metric_type,
                "efSearch": self._efSearch,
            }
        }
        self._batch_query_with_payload(payload)

    def __str__(self):
        return ("VearchHNSW" +
                ", nlinks: " + str(self._nlinks) +
                ", efConstruction: " + str(self._efConstruction) +
                ", efSearch: " + str(self._efSearch) +
                # ", master: " + self._master_prefix +
                # ", router: " + self._router_prefix +
                # ", partition_num: " + str(self._partition_num) +
                # ", replica_num: " + str(self._replica_num) +
                ", metric_type: " + str(self._metric_type))


class VearchGPU(Vearch):
    def __init__(self, metric_type, ncentroids, nsubvector=64, partition_num=1, replica_num=1, nbits_per_idx=8):
        Vearch.__init__(self)
        self._ncentroids = ncentroids
        self._nsubvector = 64
        self._partition_num = partition_num
        self._replica_num = replica_num
        if metric_type == 'L2':
            self._metric_type = 'L2'
        else:
            self._metric_type = 'InnerProduct'
        self._nbits_per_idx = nbits_per_idx

    def fit(self, X):
        self._create_db()
        max_size = X.shape[0]
        index_size = min(max_size, self._ncentroids * 128)
        dimension = X.shape[1]
        retrieval_type = "GPU"
        retrieval_param = {
            "ncentroids": self._ncentroids,
            "nsubvector": self._nsubvector,
            "nbits_per_idx": self._nbits_per_idx,
            "metric_type": self._metric_type,
        }
        payload = {
            "name": self._table_name,
            "partition_num": self._partition_num,
            "replica_num": self._replica_num,
            "engine": {
                "name": "gamma",
                "index_size": 0,
                "max_size": max_size,
                "retrieval_type": retrieval_type,
                "retrieval_param": retrieval_param
            },
            "properties": {
                self._field: {
                    "type": "vector",
                    "index": True,
                    "dimension": dimension,
                    "store_type": "MemoryOnly",
                }
            }
        }
        self._create_table(payload)
        self._bulk_insert(X)
        # self._single_insert(X)
        self._create_index()

    def set_query_arguments(self, nprobe):
        # nprobe <= 1024/2048
        self._nprobe = min(nprobe, self._ncentroids)

    def batch_query(self, X, n):
        features = []
        for vector in X:
            features += vector.tolist()
            # features.append(vector.tolist())
        payload = {
            "query": {
                "sum": [{
                    "field": self._field,
                    "feature": features,
                }]
            },
            "size": n,
            "sort": [{
                "_score": {"order": "asc"}
            }],
            "retrieval_params": {
                "parallel_on_queries": 0,
                "recall_num": n,  # should equal to size here
                "nprobe": self._nprobe,
                "metric_type": self._metric_type
            }
        }
        self._batch_query_with_payload(payload)

    def __str__(self):
        return ("VearchIVFPQ" +
                ", ncentroids: " + str(self._ncentroids) +
                ", nprobe: " + str(self._nprobe) +
                ", nsubvector: " + str(self._nsubvector) +
                # ", master: " + self._master_prefix +
                # ", router: " + self._router_prefix +
                # ", partition_num: " + str(self._partition_num) +
                # ", replica_num: " + str(self._replica_num) +
                ", metric_type: " + str(self._metric_type) +
                ", nbits_per_idx: " + str(self._nbits_per_idx))
