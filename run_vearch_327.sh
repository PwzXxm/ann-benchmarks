#!/bin/bash

set -x
set -e

# DEF=definitions/on-cloud/algos.yaml.vearch.hnsw.327
DEF=definitions/on-cloud/algos.yaml.vearch.ivfflat.327
DATASET=deep-96-angular
# DATASET=sift-10m-euclidean
# ALGO=hnsw-8-100
ALGO=ivfflat-8192
LOG_F=${DATASET}-${ALGO}.log
METRIC_F=${DATASET}-${ALGO}-metrics.log
COLS=6
if [[ $ALGO == *"ivf"* ]]; then
  COLS=5
fi
if [[ $DATASET == *"angular"* ]]; then
  COLS=$(($COLS-1))
fi

python3 run.py --force -k 50 --runs 5 --dataset $DATASET --batch --local --definitions $DEF >> $LOG_F
python3 create_website.py --outputdir web/ --recompute > $METRIC_F

cat $METRIC_F | grep -E 'Vearch|k-nn:|qps:' | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | xargs -n $COLS