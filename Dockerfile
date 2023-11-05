FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

# Install some basic utilities
RUN apt-get update && apt-get install python3 python3-pip -y

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    wget \
    unzip \
    nvidia-cuda-toolkit \
    tmux \
 && rm -rf /var/lib/apt/lists/*

### Mount Point ###
# When launching the container, mount the code directory to /app
ARG APP_MOUNT=/app
VOLUME [ ${APP_MOUNT} ]
WORKDIR ${APP_MOUNT}

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir fastapi ninja tokenizers

ENV LLVM_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
ENV PATH=$PATH:/opt/rocm/bin:/libtorch/bin:
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/:/libtorch/lib:
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/libtorch/include:/libtorch/include/torch/csrc/api/include/:/opt/rocm/include/:

# Install ROCm flash-attention
# The adaptation of a new flash attention interface seems imminent, so check out the last commit before such request is merged
RUN mkdir libs \
    && cd libs \
    && git clone https://github.com/ROCmSoftwarePlatform/flash-attention.git --recursive \
    && cd flash-attention \
    && patch /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/utils/hipify/hipify_python.py hipify_patch.patch \
    && python3 setup.py install \
    && cd ..

RUN cd /app \
    && git clone https://github.com/EmbeddedLLM/vllm-rocm.git \
    && cd vllm-rocm \
    && git checkout v0.2.x-rocm-dev \
    && python3 setup.py install \
    && cd ..

RUN cd /app \
    && mkdir dataset \
    && cd ..

COPY ./benchmark_throughput.sh /app/benchmark_throughput.sh

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir ray[all]

CMD ["/bin/bash"]