#!/usr/bin/env bash

export BATCHSIZE=8
export NUMEPOCHS=${NUMEPOCHS:-8}
export DATASET_DIR="/dataset"
export EXTRA_PARAMS='--lr 0.001 --output-dir=/results'
