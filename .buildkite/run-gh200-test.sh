#!/bin/bash

# This script build the GH200 docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Download the python
PYTHON_VERSION=3.12
apt-get update -y \
  && apt-get update -y \
  && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
  && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
  && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
  && python3 --version && python3 -m pip --version

# Install the nightly version of torch and torchvision
python3 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu124 "torch==2.6.0.dev20241210+cu124"
python3 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu124 "torchvision==0.22.0.dev20241215"

# Skip the new torch installation during build since we are using the specified version
python3 use_existing_torch.py

# Try building the docker image
DOCKER_BUILDKIT=1 docker build . \
  --target vllm-openai \
  --platform "linux/arm64" \
  -t gh200-test \
  --build-arg max_jobs=66 \
  --build-arg nvcc_threads=2 \
  --build-arg torch_cuda_arch_list="9.0+PTX" \
  --build-arg vllm_fa_cmake_gpu_arches="90-real"

# Setup cleanup
remove_docker_container() { docker rm -f gh200-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and test offline inference
docker run --name gh200-test --gpus=all --entrypoint="" gh200-test bash -c '
    python3 examples/offline_inference.py
'
