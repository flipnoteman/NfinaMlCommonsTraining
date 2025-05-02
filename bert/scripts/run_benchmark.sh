#!/bin/bash

python ./bert/run_pretraining.py \
  --bert_config_file=./input_files/bert_config.json \
  --output_dir=/tmp/output/ \
  --input_file=./eval_10k \
  --do_eval \
  --nodo_train \
  --eval_batch_size=8 \
  --init_checkpoint=./input_files/tf2_ckpt/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --learning_rate=0.001 \
  --max_eval_steps=125 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_gpus=1 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=1562 \
  --start_warmup_step=0 \
  --train_batch_size=24 \
  --nouse_tpu
