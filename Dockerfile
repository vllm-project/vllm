# syntax=docker/dockerfile:1
# the above line is important for the multiline format in dockerfile

# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

#################### BASE Python IMAGE ####################
FROM ubuntu:22.04 AS python

# install miniconda as python environment manager
# use /root/download for manual cache
RUN --mount=type=cache,target=/root/download <<EOF bash
apt-get update -y
apt-get install -y wget git vim
mkdir -p /opt/conda
if [ ! -f /root/download/miniconda.sh ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/download/miniconda.sh
else
    echo "/root/download/miniconda.sh already exists."
fi

bash /root/download/miniconda.sh -b -u -p /opt/conda

# this is useful for login shell
# inside dockerfile, however, RUN command is executed with non-login shell
# so we still need to manipulate PATH
/opt/conda/bin/conda init bash
EOF

ARG PYTHON='3.9'

# add conda and pythhon/pip to path
ENV PATH="/opt/conda/bin:${PATH}"

# install python
# cache location found from `conda info`
RUN --mount=type=cache,target=/root/.conda/pkgs conda install -y python=${PYTHON}

# now `python` is a new python
# and `pip` is the corresponding pip
#################### BASE Python IMAGE ####################

#################### BASE cuda IMAGE ####################
FROM python AS cudatoolkit

ARG CUDATOOLKIT='12.1'

# required for cuda toolkit install
RUN apt-get install -y libxml2 build-essential

# use /root/download for manual cache
RUN --mount=type=cache,target=/root/download <<EOF bash
# download url from https://developer.nvidia.com/cuda-toolkit-archive
mkdir -p /usr/local/cuda

if [[ "$CUDATOOLKIT" == "11.8" ]]; then
    if [ ! -f /root/download/${CUDATOOLKIT}-installer.run ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O /root/download/${CUDATOOLKIT}-installer.run
    else
        echo "/root/download/${CUDATOOLKIT}-installer.run already exists."
    fi
elif [[ "$CUDATOOLKIT" == "12.1" ]]; then
    if [ ! -f /root/download/${CUDATOOLKIT}-installer.run ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run -O /root/download/${CUDATOOLKIT}-installer.run
    else
        echo "/root/download/${CUDATOOLKIT}-installer.run already exists."
    fi
else
    echo "Unsupported CUDA version: $CUDATOOLKIT"
fi

chmod +x /root/download/${CUDATOOLKIT}-installer.run
/bin/bash /root/download/${CUDATOOLKIT}-installer.run --silent --toolkit --installpath=/usr/local/cuda
EOF

# now `/usr/local/cuda` is a new cudatoolkit
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin/:${PATH}"
#################### BASE cuda IMAGE ####################

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM cudatoolkit AS dev

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cuda.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BASE BUILD IMAGE ####################


#################### WHEEL BUILD IMAGE ####################
FROM dev AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# install compiler cache to speed up compilation leveraging local or remote caching
RUN apt-get update -y && apt-get install -y ccache

# files and directories related to build wheels
COPY csrc csrc
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
COPY pyproject.toml pyproject.toml
COPY vllm vllm

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads
# make sure punica kernels are built (for LoRA)
ENV VLLM_INSTALL_PUNICA_KERNELS=1

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=dist

# check the size of the wheel, we cannot upload wheels larger than 100MB
COPY .buildkite/check-wheel-size.py check-wheel-size.py
RUN python3 check-wheel-size.py dist

# the `vllm_nccl` package must be installed from source distribution
# pip is too smart to store a wheel in the cache, and other CI jobs
# will directly use the wheel from the cache, which is not what we want.
# we need to remove it manually
RUN --mount=type=cache,target=/root/.cache/pip \
    pip cache remove vllm_nccl*
#################### EXTENSION Build IMAGE ####################

#################### FLASH_ATTENTION Build IMAGE ####################
FROM dev as flash-attn-builder
# max jobs used for build
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# flash attention version
ARG flash_attn_version=v2.5.8
ENV FLASH_ATTN_VERSION=${flash_attn_version}

WORKDIR /usr/src/flash-attention-v2

# Download the wheel or build it if a pre-compiled release doesn't exist
RUN pip --verbose wheel flash-attn==${FLASH_ATTN_VERSION} \
    --no-build-isolation --no-deps --no-cache-dir

#################### FLASH_ATTENTION Build IMAGE ####################

#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM python AS vllm-base
WORKDIR /vllm-workspace

# install vllm wheel first, so that torch etc will be installed
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    pip install dist/*.whl --verbose

RUN --mount=type=bind,from=flash-attn-builder,src=/usr/src/flash-attention-v2,target=/usr/src/flash-attention-v2 \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /usr/src/flash-attention-v2/*.whl --no-cache-dir
#################### vLLM installation IMAGE ####################


#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# doc requires source code
# we hide them inside `test_docs/` , so that this source code
# will not be imported by other tests
RUN mkdir test_docs
RUN mv docs test_docs/
RUN mv vllm test_docs/

#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer modelscope

ENV VLLM_USAGE_SOURCE production-docker-image

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
