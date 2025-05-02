#!/bin/bash

./run_and_time.sh \
  --num-nodes 1 \
  --gpus-per-node 1 \
  --checkpoint /checkpoints/sd/512-base-ema.ckpt \
  --results-dir /results \
  --config configs/train_custom_raw_images.yaml
