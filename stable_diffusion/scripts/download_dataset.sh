#!/bin/bash

# Download raw images
/diffusion/scripts/datasets/laion400m-filtered-download-images.sh --output-dir /datasets/laion-400m/webdataset-filtered

# Download validation dataset
/diffusion/scripts/datasets/coco2014-validation-download-prompts.sh --output-dir /datasets/coco2014
/diffusion/scripts/datasets/coco2014-validation-download-stats.sh --output-dir /datasets/coco2014

# Download checkpoints
/diffusion/scripts/checkpoints/download_sd.sh --output-dir /checkpoints/sd
/diffusion/scripts/checkpoints/download_inception.sh --output-dir /checkpoints/inception
/diffusion/scripts/checkpoints/download_clip.sh --output-dir /checkpoints/clip
