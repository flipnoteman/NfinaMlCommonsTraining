# Start from TensorFlow 1.15.2 GPU base image
FROM tensorflow/tensorflow:1.15.2-gpu

# Fix the missing NVIDIA GPG key issue
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Update package lists and install Python 3.7 and basic utilities
RUN apt-get update && apt-get install -y \
    software-properties-common wget curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3.7-distutils \
    && ln -sf /usr/bin/python3.7 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7

# Make sure pip3 points correctly
RUN ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# Install NLTK version 3.4.5
RUN pip install nltk==3.4.5
RUN pip install tensorflow-gpu==1.15.2
RUN pip install absl-py
RUN pip install protobuf==3.20
# Create workspace directory
WORKDIR /workspace

# Copy everything from your current directory into /workspace
COPY download_dataset.sh /workspace/download_dataset.sh
COPY bert/ /workspace/bert/
COPY scripts/ /workspace/scripts/
RUN chmod +x download_dataset.sh

# Default command: execute the dataset download script
CMD ["./download_dataset.sh"]
