# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends ccache software-properties-common git curl sudo \
       python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv cmake ninja-build uv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"

RUN <<EOF
gcc --version
EOF

WORKDIR /workspace

# install build and runtime dependencies

# arm64 (GH200) build follows the practice of "use existing pytorch" build,
# we need to install torch and torchvision from the nightly builds first,
# pytorch will not appear as a vLLM dependency in all of the following steps
# after this step
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --system --index-url https://download.pytorch.org/whl/nightly/cu128 "torch==2.8.0.dev20250318+cu128" "torchvision==0.22.0.dev20250319";  \
        uv pip install --system --index-url https://download.pytorch.org/whl/nightly/cu128 --pre pytorch_triton==3.3.0+gitab727c40; \
    fi

COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/cuda.txt \
    --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}
#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM
ARG CUDA_VERSION
ARG PYTHON_VERSION

# install build dependencies
COPY requirements/build.txt requirements/build.txt

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
    --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

COPY . .
ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

ARG USE_SCCACHE
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0
# if USE_SCCACHE is set, use sccache to speed up compilation
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && curl -L -o sccache.tar.gz https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp$(echo $PYTHON_VERSION | tr -d '.') \
        && sccache --show-stats; \
    fi

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git  \
    if [ "$USE_SCCACHE" != "1" ]; then \
        # Clean any existing CMake artifacts
        rm -rf .deps && \
        mkdir -p .deps && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp$(echo $PYTHON_VERSION | tr -d '.'); \
    fi

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# sync the default value with .buildkite/check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=400
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi
#################### EXTENSION Build IMAGE ####################

#################### DEV IMAGE ####################
FROM base as dev
ARG CUDA_VERSION
ARG PYTHON_VERSION

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"

# Workaround for #17068
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"

COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test.txt requirements/test.txt
COPY requirements/dev.txt requirements/dev.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
#################### DEV IMAGE ####################

#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

SHELL ["/bin/bash", "-c"]

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends ccache software-properties-common git curl wget sudo vim python3-pip \
       ffmpeg libsm6 libxext6 libgl1 libibverbs-dev \
       python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv uv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"

# arm64 (GH200) build follows the practice of "use existing pytorch" build,
# we need to install torch and torchvision from the nightly builds first,
# pytorch will not appear as a vLLM dependency in all of the following steps
# after this step
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --system --index-url https://download.pytorch.org/whl/nightly/cu128 "torch==2.8.0.dev20250318+cu128" "torchvision==0.22.0.dev20250319";  \
        uv pip install --system --index-url https://download.pytorch.org/whl/nightly/cu128 --pre pytorch_triton==3.3.0+gitab727c40; \
    fi

# Install vllm wheel first, so that torch etc will be installed.
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system dist/*.whl --verbose \
    --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')


RUN --mount=type=cache,target=/root/.cache/uv \
. /etc/environment && \
if [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
    # Build FlashInfer from source for Python 3.12
    export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX'; \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -lt 12 ]; then \
        export FLASHINFER_ENABLE_SM90=0; \
    fi; \
    uv pip install --system --no-build-isolation "git+https://github.com/flashinfer-ai/flashinfer@21ea1d2545f74782b91eb8c08fd503ac4c0743fc" ; \
fi
COPY examples examples
COPY benchmarks benchmarks
COPY ./vllm/collect_env.py .

RUN --mount=type=cache,target=/root/.cache/uv \
. /etc/environment && \
uv pip list

# Even when we build Flashinfer with AOT mode, there's still
# some issues w.r.t. JIT compilation. Therefore we need to
# install build dependencies for JIT compilation.
# TODO: Remove this once FlashInfer AOT wheel is fixed
COPY requirements/build.txt requirements/build.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
    --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

#################### vLLM installation IMAGE ####################

#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test
ARG CUDA_VERSION
ARG PYTHON_VERSION

ADD . /vllm-workspace/

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"

# Workaround for #17068
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/uv \   
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -ge 12 ]; then \
        uv pip install --system -r requirements/dev.txt; \
    fi

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e tests/vllm_test_utils

# enable fast downloads from hf (for testing)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER 1

# Copy in the v1 package for testing (it isn't distributed yet)
COPY vllm/v1 /usr/local/lib/python${PYTHON_VERSION}/dist-packages/vllm/v1

# doc requires source code
# we hide them inside `test_docs/` , so that this source code
# will not be imported by other tests
RUN mkdir test_docs
RUN mv docs test_docs/
RUN cp -r examples test_docs/
RUN mv vllm test_docs/
RUN mv mkdocs.yaml test_docs/
#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# base openai image with additional requirements, for any subsequent openai-style images
FROM vllm-base AS vllm-openai-base
ARG TARGETPLATFORM
ARG PYTHON_VERSION

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --system accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.42.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    else \
        # For x86_64, bitsandbytes version can be higher
        uv pip install --system accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.45.3' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    fi

ENV VLLM_USAGE_SOURCE production-docker-image

# define sagemaker first, so it is not default from `docker build`
FROM vllm-openai-base AS vllm-sagemaker

COPY examples/online_serving/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

FROM vllm-openai-base AS vllm-openai

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
