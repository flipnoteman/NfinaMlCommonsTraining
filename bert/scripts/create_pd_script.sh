#!/bin/bash

# Directory containing the files (change as needed)
dir="../bert/processed_dataset/results4"

# Iterate through files matching the pattern
for file in "$dir"/part-00???-of-00500; do
	# Check if the file exists to avoid wildcard expansion issues
	[[ -e "$file" ]] || continue
	filename=$(basename "$file")
	echo "Processing file: $file"
	python3 ../bert/cleanup_scripts/create_pretraining_data.py \
	   --input_file=./$file \
	   --output_file=$dir/tfrecords/$filename \
	   --vocab_file=../input_files/vocab.txt \
	   --do_lower_case=True \
	   --max_seq_length=512 \
	   --max_predictions_per_seq=76 \
	   --masked_lm_prob=0.15 \
	   --random_seed=12345 \
	   --dupe_factor=10
	 
done
