# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/contributing/dockerfile/dockerfile.md and
# docs/assets/contributing/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.12

# By parameterizing the base images, we allow third-party to use their own
# base images. One use case is hermetic builds with base images stored in
# private registries that use a different repository naming conventions.
#
# Example:
# docker build --build-arg BUILD_BASE_IMAGE=registry.acme.org/mirror/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Important: We build with an old version of Ubuntu to maintain broad
# compatibility with other Linux OSes. The main reason for this is that the
# glibc version is baked into the distro, and binaries built with one glibc
# version are not backwards compatible with OSes that use an earlier version.
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
# Using cuda base image with minimal dependencies necessary for JIT compilation (FlashInfer, DeepGEMM, EP kernels)
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04

# By parameterizing the Deadsnakes repository URL, we allow third-party to use
# their own mirror. When doing so, we don't benefit from the transparent
# installation of the GPG key of the PPA, as done by add-apt-repository, so we
# also need a URL for the GPG key.
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL

# The PyPA get-pip.py script is a self contained script+zip file, that provides
# both the installer script and the pip base85-encoded zip archive. This allows
# bootstrapping pip in environment where a distribution package does not exist.
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

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including build tools
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache \
        software-properties-common \
        git \
        curl \
        sudo \
        python3-pip \
        libibverbs-dev \
        # Upgrade to GCC 10 to avoid https://gcc.gnu.org/bugzilla/show_bug.cgi?id=92519
        # as it was causing spam when compiling the CUTLASS kernels
        gcc-10 \
        g++-10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && $HOME/.local/bin/uv venv /opt/venv --python ${PYTHON_VERSION} \
    && rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip \
    && ln -s /opt/venv/bin/python3 /usr/bin/python3 \
    && ln -s /opt/venv/bin/python3-config /usr/bin/python3-config \
    && ln -s /opt/venv/bin/pip /usr/bin/pip \
    && python3 --version && python3 -m pip --version

# Activate virtual environment and add uv to PATH
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Environment for uv
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Verify GCC version
RUN gcc --version

# Workaround for triton/pytorch issues
# This temporarily adds the CUDA compatibility libraries to the ldconfig cache.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# ============================================================
# SLOW-CHANGING DEPENDENCIES BELOW
# These are the expensive layers that we want to cache
# ============================================================

# Install PyTorch and core CUDA dependencies
# This is ~2GB and rarely changes
ARG PYTORCH_CUDA_INDEX_BASE_URL

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# CUDA arch list used by torch
# Explicitly set the list to avoid issues with torch 2.2
# See https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BUILD BASE IMAGE ####################

#################### CSRC BUILD IMAGE ####################
FROM base AS csrc-build
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
    uv pip install --python /opt/venv/bin/python3 -r requirements/build.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

WORKDIR /workspace

COPY pyproject.toml setup.py CMakeLists.txt ./
COPY cmake cmake/
COPY csrc csrc/
COPY vllm/envs.py vllm/envs.py
COPY vllm/__init__.py vllm/__init__.py

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

ARG USE_SCCACHE
ARG SCCACHE_DOWNLOAD_URL
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

# Flag to control whether to use pre-built vLLM wheels
ARG VLLM_USE_PRECOMPILED=""
ARG VLLM_MERGE_BASE_COMMIT=""
ARG VLLM_MAIN_CUDA_VERSION=""

# Use dummy version for csrc-build wheel (only .so files are extracted, version doesn't matter)
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+csrc.build"

# if USE_SCCACHE is set, use sccache to speed up compilation
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && case "${TARGETPLATFORM}" in \
          linux/arm64) SCCACHE_ARCH="aarch64" ;; \
          linux/amd64) SCCACHE_ARCH="x86_64" ;; \
          *) echo "Unsupported TARGETPLATFORM for sccache: ${TARGETPLATFORM}" >&2; exit 1 ;; \
        esac \
        && export SCCACHE_DOWNLOAD_URL="${SCCACHE_DOWNLOAD_URL:-https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl.tar.gz}" \
        && curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl \
        && if [ ! -z ${SCCACHE_ENDPOINT} ] ; then export SCCACHE_ENDPOINT=${SCCACHE_ENDPOINT} ; fi \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
        && export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" \
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
    if [ "$USE_SCCACHE" != "1" ]; then \
        # Clean any existing CMake artifacts
        rm -rf .deps && \
        mkdir -p .deps && \
        export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" && \
        export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" && \
        export VLLM_DOCKER_BUILD_CONTEXT=1 && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi
#################### CSRC BUILD IMAGE ####################

#################### EXTENSIONS BUILD IMAGE ####################
# Build DeepGEMM, pplx-kernels, DeepEP - runs in PARALLEL with csrc-build
# This stage is independent and doesn't affect csrc cache
FROM base AS extensions-build
ARG CUDA_VERSION

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# Build DeepGEMM wheel
ARG DEEPGEMM_GIT_REF
COPY tools/install_deepgemm.sh /tmp/install_deepgemm.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /tmp/deepgemm/dist && \
    VLLM_DOCKER_BUILD_CONTEXT=1 TORCH_CUDA_ARCH_LIST="9.0a 10.0a" /tmp/install_deepgemm.sh \
        --cuda-version "${CUDA_VERSION}" \
        ${DEEPGEMM_GIT_REF:+--ref "$DEEPGEMM_GIT_REF"} \
        --wheel-dir /tmp/deepgemm/dist || \
    echo "DeepGEMM build skipped (CUDA version requirement not met)"

# Ensure the wheel dir exists so COPY won't fail when DeepGEMM is skipped
RUN mkdir -p /tmp/deepgemm/dist && touch /tmp/deepgemm/dist/.deepgemm_skipped

# Build pplx-kernels and DeepEP wheels
COPY tools/ep_kernels/install_python_libraries.sh /tmp/install_python_libraries.sh
ARG PPLX_COMMIT_HASH
ARG DEEPEP_COMMIT_HASH
ARG NVSHMEM_VER
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /tmp/ep_kernels_workspace/dist && \
    export TORCH_CUDA_ARCH_LIST='9.0a 10.0a' && \
    /tmp/install_python_libraries.sh \
        --workspace /tmp/ep_kernels_workspace \
        --mode wheel \
        ${PPLX_COMMIT_HASH:+--pplx-ref "$PPLX_COMMIT_HASH"} \
        ${DEEPEP_COMMIT_HASH:+--deepep-ref "$DEEPEP_COMMIT_HASH"} \
        ${NVSHMEM_VER:+--nvshmem-ver "$NVSHMEM_VER"} && \
    find /tmp/ep_kernels_workspace/nvshmem -name '*.a' -delete
#################### EXTENSIONS BUILD IMAGE ####################

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
    uv pip install --python /opt/venv/bin/python3 -r requirements/build.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

WORKDIR /workspace

# Copy pre-built csrc wheel directly
COPY --from=csrc-build /workspace/dist /precompiled-wheels

COPY . .

ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}

# Skip adding +precompiled suffix to version (preserves git-derived version)
ENV VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    if [ "${vllm_target_device}" = "cuda" ]; then \
        export VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /precompiled-wheels/*.whl); \
    fi && \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

# Copy extension wheels from extensions-build stage for later use
COPY --from=extensions-build /tmp/deepgemm/dist /tmp/deepgemm/dist
COPY --from=extensions-build /tmp/ep_kernels_workspace/dist /tmp/ep_kernels_workspace/dist

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# sync the default value with .buildkite/check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=500
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
RUN apt-get update && apt-get install -y --no-install-recommends libnuma-dev && rm -rf /var/lib/apt/lists/*
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test.txt requirements/test.txt
COPY requirements/dev.txt requirements/dev.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python3 -r requirements/dev.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
#################### DEV IMAGE ####################
#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM ${FINAL_BASE_IMAGE} AS vllm-base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /vllm-workspace


# Python version string for paths (e.g., "312" for 3.12)
RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and system dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        sudo \
        python3-pip \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
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
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        libibverbs-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install CUDA development tools for runtime JIT compilation
# (FlashInfer, DeepGEMM, EP kernels all require compilation at runtime)
RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-${CUDA_VERSION_DASH} \
        # Fixes nccl_allocator requiring nccl.h at runtime
        # https://github.com/vllm-project/vllm/blob/1336a1ea244fa8bfd7e72751cabbdb5b68a0c11a/vllm/distributed/device_communicators/pynccl_allocator.py#L22
        libnccl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
RUN python3 -m pip install uv

# Environment for uv
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Workaround for triton/pytorch issues
# This temporarily adds the CUDA compatibility libraries to the ldconfig cache.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# ============================================================
# SLOW-CHANGING DEPENDENCIES BELOW
# These are the expensive layers that we want to cache
# ============================================================

# Install PyTorch and core CUDA dependencies
# This is ~2GB and rarely changes
ARG PYTORCH_CUDA_INDEX_BASE_URL
COPY requirements/common.txt /tmp/common.txt
COPY requirements/cuda.txt /tmp/requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements-cuda.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') && \
    rm /tmp/requirements-cuda.txt /tmp/common.txt

# Install FlashInfer pre-compiled kernel cache and binaries
# This is ~1.1GB and only changes when FlashInfer version bumps
# https://docs.flashinfer.ai/installation.html
ARG FLASHINFER_VERSION=0.5.3
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system flashinfer-cubin==${FLASHINFER_VERSION} \
    && uv pip install --system flashinfer-jit-cache==${FLASHINFER_VERSION} \
        --extra-index-url https://flashinfer.ai/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
    && flashinfer show-config

# ============================================================
# OPENAI API SERVER DEPENDENCIES
# Pre-install these to avoid reinstalling on every vLLM wheel rebuild
# ============================================================

# Install gdrcopy (saves ~6s per build)
# TODO (huydhn): There is no prebuilt gdrcopy package on 12.9 at the moment
ARG GDRCOPY_CUDA_VERSION=12.8
ARG GDRCOPY_OS_VERSION=Ubuntu22_04
ARG TARGETPLATFORM
COPY tools/install_gdrcopy.sh /tmp/install_gdrcopy.sh
RUN set -eux; \
    case "${TARGETPLATFORM}" in \
      linux/arm64) UUARCH="aarch64" ;; \
      linux/amd64) UUARCH="x64" ;; \
      *) echo "Unsupported TARGETPLATFORM: ${TARGETPLATFORM}" >&2; exit 1 ;; \
    esac; \
    /tmp/install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "${GDRCOPY_CUDA_VERSION}" "${UUARCH}" && \
    rm /tmp/install_gdrcopy.sh

# Install vllm-openai dependencies (saves ~2.6s per build)
# These are stable packages that don't depend on vLLM itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        BITSANDBYTES_VERSION="0.42.0"; \
    else \
        BITSANDBYTES_VERSION="0.46.1"; \
    fi; \
    uv pip install --system accelerate hf_transfer modelscope \
        "bitsandbytes>=${BITSANDBYTES_VERSION}" 'timm>=1.0.17' 'runai-model-streamer[s3,gcs]>=0.15.3'

# ============================================================
# VLLM INSTALLATION (depends on build stage)
# ============================================================

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# Install vllm wheel first, so that torch etc will be installed.
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

RUN --mount=type=cache,target=/root/.cache/uv \
. /etc/environment && \
uv pip list

# Install deepgemm wheel that has been built in the `build` stage
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=build,source=/tmp/deepgemm/dist,target=/tmp/deepgemm/dist,ro \
    sh -c 'if ls /tmp/deepgemm/dist/*.whl >/dev/null 2>&1; then \
              uv pip install --system /tmp/deepgemm/dist/*.whl; \
           else \
              echo "No DeepGEMM wheels to install; skipping."; \
           fi'

# Pytorch now installs NVSHMEM, setting LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install EP kernels wheels (pplx-kernels and DeepEP) that have been built in the `build` stage
RUN --mount=type=bind,from=build,src=/tmp/ep_kernels_workspace/dist,target=/vllm-workspace/ep_kernels/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ep_kernels/dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# CUDA image changed from /usr/local/nvidia to /usr/local/cuda in 12.8 but will
# return to /usr/local/nvidia in 13.0 to allow container providers to mount drivers
# consistently from the host (see https://github.com/vllm-project/vllm/issues/18859).
# Until then, add /usr/local/nvidia/lib64 before the image cuda path to allow override.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Add CUDA compatibility libraries to the end of LD_LIBRARY_PATH.
# This ensures they are used as a fallback if the host driver is too old.
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/compat

# Copy examples and benchmarks at the end to minimize cache invalidation
COPY examples examples
COPY benchmarks benchmarks
COPY ./vllm/collect_env.py .
#################### vLLM installation IMAGE ####################
#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

ARG PYTHON_VERSION

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y git

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -ge 12 ]; then \
        uv pip install --system -r requirements/dev.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
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
ARG CUDA_VERSION

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

# install kv_connectors if requested
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=requirements/kv_connectors.txt,target=/tmp/kv_connectors.txt,ro \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-'); \
    CUDA_HOME=/usr/local/cuda; \
    # lmcache requires explicit specifying CUDA_HOME
    BUILD_PKGS="libcusparse-dev-${CUDA_VERSION_DASH} \
                libcublas-dev-${CUDA_VERSION_DASH} \
                libcusolver-dev-${CUDA_VERSION_DASH}"; \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        if [ "$CUDA_MAJOR" -ge 13 ]; then \
            uv pip install --system nixl-cu13; \
        fi; \
        uv pip install --system -r /tmp/kv_connectors.txt --no-build || ( \
            # if the above fails, install from source
            apt-get update -y && \
            apt-get install -y --no-install-recommends ${BUILD_PKGS} && \
            uv pip install --system -r /tmp/kv_connectors.txt --no-build-isolation && \
            apt-get purge -y ${BUILD_PKGS} && \
            # clean up -dev packages, keep runtime libraries
            rm -rf /var/lib/apt/lists/* \
        ); \
    fi

ENV VLLM_USAGE_SOURCE production-docker-image

# define sagemaker first, so it is not default from `docker build`
FROM vllm-openai-base AS vllm-sagemaker

COPY examples/online_serving/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

FROM vllm-openai-base AS vllm-openai

ENTRYPOINT ["vllm", "serve"]
#################### OPENAI API SERVER ####################
