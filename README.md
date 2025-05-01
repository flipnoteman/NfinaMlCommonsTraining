# NfinaMlCommonsTraining


### Bert

#### Instructions to download dataset:

```bash
# Change directory to bert root
cd bert

# Build the dataset container
sudo docker build -f Dockerfile.download -t bert:dataset .

# Run the container in the background
sudo docker run --rm -d --gpus all \
  -v "$(pwd)/input_files:/workspace/input_files" \
  -v "$(pwd)/processed_dataset:/workspace/processed_dataset" \
  bert:dataset
```

This can take upwards of 3-4 hours even with the optimizations we included to the processing script, so ``-d`` is used to have it run in the background. This does use an immense amount of resources though, so it may not be optimal to do other things while this is running.

#### Run benchmark:

```bash
cd bert

sudo docker build -f Dockerfile.run -t bert:run .

sudo docker run --rm -d --gpus all \
  -v "$(pwd)/input_files:/workspace/input_files" \
  -v "$(pwd)/processed_dataset:/workspace/processed_dataset" \
  -v "$(pwd)/output:/tmp/output" \
  bert:run
```
