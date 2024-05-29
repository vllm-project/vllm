# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.4/compat/

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

#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM dev AS build

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

#################### EXTENSION Build IMAGE ####################

#################### FLASH_ATTENTION Build IMAGE ####################
FROM dev as flash-attn-builder
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
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS vllm-base
WORKDIR /vllm-workspace

RUN apt-get update -y \
    && apt-get install -y python3-pip git vim

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.4/compat/

# install nm-vllm wheel first, so that torch etc will be installed
ARG build_type="nightly"
ARG build_version="latest"
ENV INSTALL_TYPE=${build_type}
ENV INSTALL_VERSION=${build_version}
# UPSTREAM SYNC: Install nm-vllm with sparsity extras
# use nm pypi for now for testing
RUN --mount=type=bind,from=build \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_TYPE}" = "nightly" ]; then \
        if [ "${INSTALL_VERSION}" = "latest" ]; then \
            pip install nm-vllm-nightly[sparse] --extra-index-url https://pypi.neuralmagic.com/simple; \
        else \
            pip install nm-vllm-nightly[sparse]==${INSTALL_VERSION} --extra-index-url https://pypi.neuralmagic.com/simple; \
        fi; \
    else \
        if [ "${INSTALL_VERSION}" = "latest" ]; then \
            pip install nm-vllm[sparse] --extra-index-url https://pypi.neuralmagic.com/simple; \
        else \
            pip install nm-vllm[sparse]==${INSTALL_VERSION} --extra-index-url https://pypi.neuralmagic.com/simple; \
        fi; \
    fi

RUN --mount=type=bind,from=flash-attn-builder,src=/usr/src/flash-attention-v2,target=/usr/src/flash-attention-v2 \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /usr/src/flash-attention-v2/*.whl --no-cache-dir
#################### vLLM installation IMAGE ####################

#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

# check installed version
RUN pip freeze | grep -e nm-vllm -e nm-magic-wand

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
