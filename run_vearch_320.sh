#!/bin/bash

# test, warm up, save time
# python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.hnsw.320
# python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.ivfpq.320
# python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.ivfflat.320

#python run.py --dataset sift-1000000-10000 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.hnsw.320
#python run.py --dataset sift-1000000-10000 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.ivfpq.320
# python run.py --dataset sift-1000000-10000 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.ivfflat.320


python3 run.py --force -k 50 --dataset sift-1m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.ivfflat.325 >sift.out&

#nohup docker run -d -p 8817:8817 -p 9001:9001 -v $PWD/config.toml:/vearch/config.toml -v $PWD/datas:/datas -v $PWD/datas1:/datas1 -v $PWD/logs:/logs --name vearch vearch/vearch:3.2.5 all