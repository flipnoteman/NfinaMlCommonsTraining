#!/bin/bash

# Directory containing the files
dir="./processed_dataset/results4"

echo $dir
ls "$dir/"

mkdir -p "$dir"/tfrecords

# Max number of parallel jobs
MAX_PARALLEL_JOBS=8   # <-- you can tune this based on your CPU/GPU resources

# Function to process a single file
process_file() {
    local file="$1"
    local filename
    filename=$(basename "$file")
    echo "Processing file: $file"
    python3 ./bert/cleanup_scripts/create_pretraining_data.py \
       --input_file=./$file \
       --output_file="$dir"/tfrecords/"$filename" \
       --vocab_file=./input_files/vocab.txt \
       --do_lower_case=True \
       --max_seq_length=512 \
       --max_predictions_per_seq=76 \
       --masked_lm_prob=0.15 \
       --random_seed=12345 \
       --dupe_factor=10
}

# Job counter
job_count=0

# Iterate through files
for file in "$dir"/part-00???-of-00500; do
    [[ -e "$file" ]] || continue

    process_file "$file" &  # Launch in background
    ((job_count++))

    # If we reach max jobs, wait for all to finish
    if (( job_count >= MAX_PARALLEL_JOBS )); then
        wait
        job_count=0
    fi
done

# Final wait to ensure all remaining background jobs finish
wait

