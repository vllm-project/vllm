# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/contributing/dockerfile/dockerfile.md and
# docs/assets/contributing/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12

# By parameterizing the base images, we allow third-party to use their own
# base images. One use case is hermetic builds with base images stored in
# private registries that use a different repository naming conventions.
#
# Example:
# docker build --build-arg BUILD_BASE_IMAGE=registry.acme.org/mirror/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
# TODO: Restore to base image after FlashInfer AOT wheel fixed
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# By parameterizing the Deadsnakes repository URL, we allow third-party to use
# their own mirror. When doing so, we don't benefit from the transparent
# installation of the GPG key of the PPA, as done by add-apt-repository, so we
# also need a URL for the GPG key.
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL

# The PyPA get-pip.py script is a self contained script+zip file, that provides
# both the installer script and the pip base85-encoded zip archive. This allows
# bootstrapping pip in environment where a dsitribution package does not exist.
#
# By parameterizing the URL for get-pip.py installation script, we allow
# third-party to use their own copy of the script stored in a private mirror.
# We set the default value to the PyPA owned get-pip.py script.
#
# Reference: https://pip.pypa.io/en/stable/installation/#get-pip-py
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

# PIP supports fetching the packages from custom indexes, allowing third-party
# to host the packages in private mirrors. The PIP_INDEX_URL and
# PIP_EXTRA_INDEX_URL are standard PIP environment variables to override the
# default indexes. By letting them empty by default, PIP will use its default
# indexes if the build process doesn't override the indexes.
#
# Uv uses different variables. We set them by default to the same values as
# PIP, but they can be overridden.
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

# PyTorch provides its own indexes for standard and nightly builds
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl
ARG PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL=https://download.pytorch.org/whl/nightly

# PIP supports multiple authentication schemes, including keyring
# By parameterizing the PIP_KEYRING_PROVIDER variable and setting it to
# disabled by default, we allow third-party to use keyring authentication for
# their private Python indexes, while not changing the default behavior which
# is no authentication.
#
# Reference: https://pip.pypa.io/en/stable/topics/authentication/#keyring-support
ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

# Flag enables built-in KV-connector dependency libs into docker images
ARG INSTALL_KV_CONNECTORS=false

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM ${BUILD_BASE_IMAGE} AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TARGETPLATFORM
ARG INSTALL_KV_CONNECTORS=false
ENV DEBIAN_FRONTEND=noninteractive

ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && if [ ! -z ${DEADSNAKES_MIRROR_URL} ] ; then \
        if [ ! -z "${DEADSNAKES_GPGKEY_URL}" ] ; then \
            mkdir -p -m 0755 /etc/apt/keyrings ; \
            curl -L ${DEADSNAKES_GPGKEY_URL} | gpg --dearmor > /etc/apt/keyrings/deadsnakes.gpg ; \
            sudo chmod 644 /etc/apt/keyrings/deadsnakes.gpg ; \
            echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] ${DEADSNAKES_MIRROR_URL} $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes.list ; \
        fi ; \
    else \
        for i in 1 2 3; do \
            add-apt-repository -y ppa:deadsnakes/ppa && break || \
            { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
        done ; \
    fi \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Upgrade to GCC 10 to avoid https://gcc.gnu.org/bugzilla/show_bug.cgi?id=92519
# as it was causing spam when compiling the CUTLASS kernels
RUN apt-get install -y gcc-10 g++-10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10
RUN <<EOF
gcc --version
EOF

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/cuda.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# install build dependencies
COPY requirements/build.txt requirements/build.txt

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

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
ARG SCCACHE_DOWNLOAD_URL=https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

# Flag to control whether to use pre-built vLLM wheels
ARG VLLM_USE_PRECOMPILED=""
ARG VLLM_MAIN_CUDA_VERSION=""

# if USE_SCCACHE is set, use sccache to speed up compilation
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && if [ ! -z ${SCCACHE_ENDPOINT} ] ; then export SCCACHE_ENDPOINT=${SCCACHE_ENDPOINT} ; fi \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
        && export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION}" \
        && export VLLM_DOCKER_BUILD_CONTEXT=1 \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git  \
    if [ "$USE_SCCACHE" != "1" ]; then \
        # Clean any existing CMake artifacts
        rm -rf .deps && \
        mkdir -p .deps && \
        export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" && \
        export VLLM_DOCKER_BUILD_CONTEXT=1 && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# sync the default value with .buildkite/check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=450
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi
#################### EXTENSION Build IMAGE ####################

#################### DEV IMAGE ####################
FROM base AS dev

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Install libnuma-dev, required by fastsafetensors (fixes #20384)
RUN apt-get update && apt-get install -y libnuma-dev && rm -rf /var/lib/apt/lists/*
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test.txt requirements/test.txt
COPY requirements/dev.txt requirements/dev.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/dev.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
#################### DEV IMAGE ####################

#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM ${FINAL_BASE_IMAGE} AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG INSTALL_KV_CONNECTORS=false
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

ARG GDRCOPY_CUDA_VERSION=12.8
# Keep in line with FINAL_BASE_IMAGE
ARG GDRCOPY_OS_VERSION=Ubuntu22_04

SHELL ["/bin/bash", "-c"]

ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl wget sudo vim python3-pip \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 \
    && if [ ! -z ${DEADSNAKES_MIRROR_URL} ] ; then \
        if [ ! -z "${DEADSNAKES_GPGKEY_URL}" ] ; then \
            mkdir -p -m 0755 /etc/apt/keyrings ; \
            curl -L ${DEADSNAKES_GPGKEY_URL} | gpg --dearmor > /etc/apt/keyrings/deadsnakes.gpg ; \
            sudo chmod 644 /etc/apt/keyrings/deadsnakes.gpg ; \
            echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] ${DEADSNAKES_MIRROR_URL} $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes.list ; \
        fi ; \
    else \
        for i in 1 2 3; do \
            add-apt-repository -y ppa:deadsnakes/ppa && break || \
            { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
        done ; \
    fi \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv libibverbs-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# arm64 (GH200) build follows the practice of "use existing pytorch" build,
# we need to install torch and torchvision from the nightly builds first,
# pytorch will not appear as a vLLM dependency in all of the following steps
# after this step
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --system \
            --index-url ${PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            "torch==2.8.0.dev20250318+cu128" "torchvision==0.22.0.dev20250319" ; \
        uv pip install --system \
            --index-url ${PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            --pre pytorch_triton==3.3.0+gitab727c40 ; \
    fi

# Install vllm wheel first, so that torch etc will be installed.
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# If we need to build FlashInfer wheel before its release:
# $ # Note we remove 7.0 from the arch list compared to the list below, since FlashInfer only supports sm75+
# $ export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0a 10.0a 12.0'
# $ git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
# $ cd flashinfer
# $ git checkout v0.2.6.post1
# $ python -m flashinfer.aot
# $ python -m build --no-isolation --wheel
# $ ls -la dist
# -rw-rw-r-- 1 mgoin mgoin 205M Jun  9 18:03 flashinfer_python-0.2.6.post1-cp39-abi3-linux_x86_64.whl
# $ # upload the wheel to a public location, e.g. https://wheels.vllm.ai/flashinfer/v0.2.6.post1/flashinfer_python-0.2.6.post1-cp39-abi3-linux_x86_64.whl

# Install FlashInfer from source
ARG FLASHINFER_GIT_REPO="https://github.com/flashinfer-ai/flashinfer.git"
# Keep this in sync with "flashinfer" extra in setup.py
ARG FLASHINFER_GIT_REF="v0.3.1"
# Flag to control whether to compile FlashInfer AOT kernels
# Set to "true" to enable AOT compilation:
# docker build --build-arg FLASHINFER_AOT_COMPILE=true ...
ARG FLASHINFER_AOT_COMPILE=false
RUN --mount=type=cache,target=/root/.cache/uv bash - <<'BASH'
  . /etc/environment
    git clone --depth 1 --recursive --shallow-submodules \
        --branch ${FLASHINFER_GIT_REF} \
        ${FLASHINFER_GIT_REPO} flashinfer
    pushd flashinfer
        if [ "${FLASHINFER_AOT_COMPILE}" = "true" ]; then
            # Exclude CUDA arches for older versions (11.x and 12.0-12.7)
            # TODO: Update this to allow setting TORCH_CUDA_ARCH_LIST as a build arg.
            if [[ "${CUDA_VERSION}" == 11.* ]]; then
                FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
            elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
                FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
            else
                # CUDA 12.8+ supports 10.0a and 12.0
                FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 12.0"
            fi
            echo "🏗️  Installing FlashInfer with AOT compilation for arches: ${FI_TORCH_CUDA_ARCH_LIST}"
            # Build AOT kernels
            TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
                python3 -m flashinfer.aot
            # Install with no-build-isolation since we already built AOT kernels
            TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
                uv pip install --system --no-build-isolation . \
                --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
            # Download pre-compiled cubins
            TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
                python3 -m flashinfer --download-cubin || echo "WARNING: Failed to download flashinfer cubins."
        else
            echo "🏗️  Installing FlashInfer without AOT compilation in JIT mode"
            uv pip install --system . \
                --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
        fi
    popd
    rm -rf flashinfer
BASH
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
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Install DeepGEMM from source
ARG DEEPGEMM_GIT_REF
COPY tools/install_deepgemm.sh /tmp/install_deepgemm.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_DOCKER_BUILD_CONTEXT=1 /tmp/install_deepgemm.sh --cuda-version "${CUDA_VERSION}" ${DEEPGEMM_GIT_REF:+--ref "$DEEPGEMM_GIT_REF"} 

COPY tools/install_gdrcopy.sh install_gdrcopy.sh
RUN set -eux; \
    case "${TARGETPLATFORM}" in \
      linux/arm64) UUARCH="aarch64" ;; \
      linux/amd64) UUARCH="x64" ;; \
      *) echo "Unsupported TARGETPLATFORM: ${TARGETPLATFORM}" >&2; exit 1 ;; \
    esac; \
    ./install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "${GDRCOPY_CUDA_VERSION}" "${UUARCH}"; \
    rm ./install_gdrcopy.sh

# Install EP kernels(pplx-kernels and DeepEP)
COPY tools/ep_kernels/install_python_libraries.sh install_python_libraries.sh
ENV CUDA_HOME=/usr/local/cuda
RUN export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a+PTX}" \
    && bash install_python_libraries.sh

#################### vLLM installation IMAGE ####################

#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

ARG PYTHON_VERSION

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

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

# Source code is used in the `python_only_compile.sh` test
# We hide it inside `src/` so that this source code
# will not be imported by other tests
RUN mkdir src
RUN mv vllm src/vllm
#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# base openai image with additional requirements, for any subsequent openai-style images
FROM vllm-base AS vllm-openai-base
ARG TARGETPLATFORM
ARG INSTALL_KV_CONNECTORS=false

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

COPY requirements/kv_connectors.txt requirements/kv_connectors.txt

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        uv pip install --system -r requirements/kv_connectors.txt; \
    fi; \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        BITSANDBYTES_VERSION="0.42.0"; \
    else \
        BITSANDBYTES_VERSION="0.46.1"; \
    fi; \
    uv pip install --system accelerate hf_transfer modelscope "bitsandbytes>=${BITSANDBYTES_VERSION}" 'timm>=1.0.17' boto3 runai-model-streamer runai-model-streamer[s3]

ENV VLLM_USAGE_SOURCE production-docker-image

# define sagemaker first, so it is not default from `docker build`
FROM vllm-openai-base AS vllm-sagemaker

COPY examples/online_serving/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

FROM vllm-openai-base AS vllm-openai

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
