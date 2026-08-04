# The LMCache Dockerfile is used to build a LMCache image that is integrated
# to run with vLLM OpenAI server.

# Please update any changes made here to
# docs/source/developer_guide/docker_file.rst
# docs/source/getting_started/installation.rst
# docs/source/production/docker_deployment.rst

ARG CUDA_VERSION=13.0
ARG UBUNTU_VERSION=24.04
ARG NGC_VERSION=25.09
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:${NGC_VERSION}-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

#################### BASE BUILD IMAGE ####################
# Prepare basic build environment

FROM ${BASE_IMAGE} AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache software-properties-common git curl sudo \
        python3 python3-dev python3-venv python3-pip tzdata libxcb1-dev \
    && ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/ \
    && mv ~/.local/bin/uvx /usr/local/bin/ \
    && uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && python3 --version

WORKDIR /workspace

# CUDA arch list used by torch
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

#################### vLLM IMAGE & LMCache (Build) ##########################
# Integrate vLLM nightly build and LMCache build, and expose vLLM OpenAI API

FROM base AS image-build

# install build dependencies
COPY ./requirements/build.txt build.txt

# Max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}

# Number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

ARG CUDA_VERSION
ARG VLLM_VERSION=nightly

RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/venv/bin/activate && \
    uv pip install -r build.txt

ARG LMCACHE_COMMIT_ID=1

COPY . /workspace/LMCache
WORKDIR /workspace/LMCache

 
RUN --mount=type=cache,target=/root/.cache/ccache,id=ccache \
    --mount=type=cache,target=/root/.cache/uv,id=uv-cache,sharing=locked \
    . /opt/venv/bin/activate && \
    CUDA_TAG=cu$(echo ${CUDA_VERSION} | tr -d '.') && \
    export LMCACHE_CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    if [ "$VLLM_VERSION" = "nightly" ]; then \
        VLLM_PRECOMPILED_WHEEL_VARIANT=${CUDA_TAG} uv pip install --prerelease=allow \
            'vllm[runai,tensorizer,flashinfer]' \
            --extra-index-url https://wheels.vllm.ai/nightly/${CUDA_TAG} \
            --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
            --index-strategy unsafe-first-match ; \
    else \
        VLLM_PRECOMPILED_WHEEL_VARIANT=${CUDA_TAG} uv pip install --prerelease=allow \
            "vllm[runai,tensorizer,flashinfer]==${VLLM_VERSION}" ; \
    fi && \
    python3 -c 'import torch; print("TORCH=", torch.__version__)' && \
    python3 setup.py bdist_wheel --dist-dir=dist_lmcache && \
    uv pip install ./dist_lmcache/*.whl --verbose && \
  
    uv pip install --reinstall-package "nixl-cu${LMCACHE_CUDA_MAJOR}" \
        "nixl-cu${LMCACHE_CUDA_MAJOR}"

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]

#################### vLLM IMAGE & LMCache (Release, cu13) #######################
# Integrate vLLM and LMCache stable releases, and expose vLLM OpenAI API.
# The default lmcache wheel on PyPI is built against cu13; the cu13 torch
# index is hinted explicitly so vLLM also resolves to its cu13 build.

FROM base AS image-release

ARG CUDA_VERSION

ARG LMCACHE_VERSION

RUN . /opt/venv/bin/activate && \
    CUDA_TAG=cu$(echo ${CUDA_VERSION} | tr -d '.') && \
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    VER=$(echo "${LMCACHE_VERSION}" | sed 's/^v//') && \
    VLLM_PRECOMPILED_WHEEL_VARIANT=${CUDA_TAG} uv pip install --prerelease=allow \
        vllm[runai,tensorizer,flashinfer] \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --index-strategy unsafe-best-match && \
    uv pip install "lmcache${VER:+==${VER}}" \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --index-strategy unsafe-best-match --verbose && \
  
    uv pip install --reinstall-package "nixl-cu${CUDA_MAJOR}" "nixl-cu${CUDA_MAJOR}"

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]

#################### vLLM IMAGE & LMCache (Release, cu129) ######################
# Installs the stable cu129 vLLM release wheel and
# lmcache from the v{tag}-cu129 GitHub Release.
# The cu129 torch index is hinted explicitly so vLLM resolves to its cu129 build.

FROM base AS image-release-cu129

ARG CUDA_VERSION=12.9
ARG LMCACHE_VERSION

RUN . /opt/venv/bin/activate && \
    CUDA_TAG=cu$(echo ${CUDA_VERSION} | tr -d '.') && \
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    VER=$(echo ${LMCACHE_VERSION} | sed 's/^v//') && \
    VLLM_PRECOMPILED_WHEEL_VARIANT=${CUDA_TAG} uv pip install --prerelease=allow \
        vllm[runai,tensorizer,flashinfer] \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --index-strategy unsafe-best-match && \
    python3 -c 'import torch; print("TORCH=", torch.__version__)' && \
    uv pip install lmcache==${VER} \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VER}-cu129 \
        --index-strategy unsafe-best-match --verbose && \
    
    uv pip install --reinstall-package "nixl-cu${CUDA_MAJOR}" "nixl-cu${CUDA_MAJOR}"

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]
