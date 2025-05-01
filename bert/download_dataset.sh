#!/bin/bash

# Setup and download rclone and the wikipedia input files
curl https://rclone.org/install.sh | bash
rclone config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

# Download wikipedia input files
rclone copy mlc-training:mlcommons-training-wg-public/wikipedia_for_bert/input_files ./input_files -P

# Download Processed dataset files
rclone copy mlc-training:mlcommons-training-wg-public/wikipedia_for_bert/processed_dataset ./processed_dataset -P
tar -xvf ./processed_dataset/results_text.tar.gz -C ./processed_dataset/

# Create pretraining data for every part in the directory
./scripts/create_pd_script.sh 

# Create pretraining data for evaluation data
python3 ./bert/cleanup_scripts/create_pretraining_data.py \
  --input_file=./processed_dataset/results4/eval.txt \
  --output_file=./eval_intermediates \
  --vocab_file=input_files/vocab.txt
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

# Select samples from the evaluation data
python3 ./bert/cleanup_scripts/pick_eval_samples.py \
  --input_tfrecord=./eval_intermediates \
  --output_tfrecord=./eval_10k \
  --num_examples_to_pick=10000
