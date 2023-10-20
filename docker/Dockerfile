FROM --platform=amd64 nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

ARG MAX_JOBS

WORKDIR /workspace

RUN apt update && \
    apt install -y python3-pip python3-packaging \
    git ninja-build && \
    pip3 install -U pip

# Tweak this list to reduce build time
# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST "7.0;7.2;7.5;8.0;8.6;8.9;9.0"

# We have to manually install Torch otherwise apex & xformers won't build
RUN pip3 install "torch==2.0.0"

# This build is slow but NVIDIA does not provide binaries. Increase MAX_JOBS as needed.
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82 && \
    sed -i '/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/d' setup.py && \
    python3 setup.py install --cpp_ext --cuda_ext

RUN pip3 install "xformers==0.0.22" "transformers==4.34.0" "vllm==0.2.0" "fschat[model_worker]==0.2.30"

COPY ./docker/entrypoint.sh .

COPY ./tgi/ tgi/

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]