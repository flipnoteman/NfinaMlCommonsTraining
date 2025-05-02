#!/bin/bash

pip install fiftyone

cd /workspace/single_stage_detector/scripts

./download_openimages_mlperf.sh -d /dataset
