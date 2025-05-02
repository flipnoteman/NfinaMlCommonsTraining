# NfinaMlCommonsTraining
## Our process

We started implementing these models and quickly realized that broken dependencies and missing resources would be a common theme among all of the mlcommons benchmarks. Additionally, all of the models are meant for industry-grade AI-compute resources, which meant some deeper level of understanding how these incredibly complex machine learning models functioned would be required to make the models run on lower end industry hardware. With that said, we began attempting to work through the models by:
- Following the instructions for the benchmark. This often didn't go smoothly on first pass with the previously mentioned constraints.
- When we hit a version problem with python packges (which happened often), we attempted to modify the code/scripts causing the problems, OR install older versions of python modules when available that complied with the current implemenation. This sometimes led to further issues when the required version of a module only exists with an older python version. Since every one of these benchmark was tested or made with different versions of everything, this became a large, non-sustainable and non-scalable, task. These problems persisted through every step of the process, even when work was being done inside the premade docker containers that came with becnhmarks like stable diffusion and single stage detector.
- FInally, once every script was updated and made to work, we created separate dockerfiles for downloading the dataset and running the benchmark. This, along with updated requirements.txt files ensured that these benchmarks could be successfully executed in the future. It also improves the consistency of setup and run between benchmarks. 

The process to completely implement one model, given the dataset sizes and download times/run times, would net ~3 weeks of work. 

## Bert

#### Changes
- If you want to build the dataset from scratch as we did, you would need to edit the download scripts as the original dataset that was used no longer exists on wikipedias archive. They only support the enlish dataset from 2017 on and this benchmark was originally made for the 2015 dataset. We decided to use the premade dataset instead because of this.
- There was an issue with ```FixedLengthFeature``` in the ```preprocess_dataset.py``` script. We included default values for all of those parameters and that seems to have fixed the issue.
- Created both dockerfiles from scratch in order to make sure that things will install correclty and run as intended. This was the only benchmark we implemented that didnt already have some dockerfile to work inside of.
- When using python versions >=3.9, there doesn't exist a ```tensorflow-estimator``` package that works with the gpu accelerated versions of tensorflow, therefore the script was entirely CPU driven when we followed the instructions. To fix this we made sure our docker containers were using Python 3.7.
- Wrote a script called ```create_pd_script.sh``` that will process every part of the dataset. The given tools only allowed you to do one at a time. In addition, each process takes around 3 minutes to run, so we optimized this by using bash features to run 8 processes at once, since there aren't any conflicts between input and output for each file.
 
#### Download dataset:

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

This docker image is around 16 gbs in size. Once run, this can take upwards of 3-4 hours even with the optimizations we included to the processing script, so ``-d`` is used to have it run in the background. This does use an immense amount of resources though, so it may not be optimal to do other things while this is running.

#### Run benchmark:

```bash
cd bert

# Build docker container that will run the benchmark
sudo docker build -f Dockerfile.run -t bert:run .

# Run the benchmark and pass in dataset files
sudo docker run --rm -d --gpus all \
  -v "$(pwd)/input_files:/workspace/input_files" \
  -v "$(pwd)/processed_dataset:/workspace/processed_dataset" \
  -v "$(pwd)/output:/tmp/output" \
  bert:run
```

Output should be placed in the output directory in the bert root directory once it finishes. If you don't wish for the container to delete itself, remove the ```--rm``` from the run command. 

## Stable Diffusion

#### Changes
- This model uses ```pythorch lightning``` as its ML base, this presented problems early on as it had support for both the new version of ```pytorch_lightning``` which has kind of decoupled itself from ```pytorch``` and is now just ```lightning```, however the codebase was not written to support the newer version. This caused conflicts with types and functions that changed definitions such as the ```trainer.add_parse_args()``` function used in the training code. Our first approach to fixing these problems was to manually edit the code to match the newer version of lightning. This seemed successful
- The configuration files for this model expect at least one Node with 8 GPUs. Therefore all of the model hyperparameters (of which there are a lot) were made for much more profficient systems than we were working with. We were able to mostly fix this by creating a custom config file label ```training_custom_raw_images```.
- For this model, we chose to use the raw images for training as the preproccessed images took up >800gbs. The raw images only took ~230 gbs. Not a major change, though the preprocessed dataset is said to lead to quicker training times.
- ```mlperf_logging_utils``` does not exist so it is replaced by downloading the ```mlperf_logging``` library from git and installing it as a python module in the Dockerfile.

#### Download dataset:
```bash
# Change directory to bert root
cd stable_diffusion

# Build the dataset container
sudo docker build -f Dockerfile.download -t stable_diffusion:dataset .

# Make shared folders
mkdir checkpoints datasets

# Run the container in the background
sudo docker run --rm -d --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)/datasets:/datasets" \
  -v "$(pwd)/checkpoints:/checkpoints" \
  -v "$(pwd)/results:/results" \
  stable_diffusion:dataset
```

This docker container will be around 30 gb in size. Once run, it will likely take around 4 hours on gigabit. The final download size will be around 380 gbs.

#### Run benchmarks
```bash
# Change directory to bert root
cd stable_diffusion

# Build the dataset container
sudo docker build -f Dockerfile.run -t stable_diffusion:run .

# Run the container in the background
sudo docker run --rm -d --gpus all \
  -v "$(pwd)/datasets:/datasets" \
  -v "$(pwd)/checkpoints:/checkpoints" \
  -v "$(pwd)/results:/results" \
  stable_diffusion:run
```

This docker container will be around 

## Single Stage Detector
