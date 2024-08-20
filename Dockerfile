# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
#################### BASE IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS base
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_USE_DEPRECATED=legacy-resolver

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && python3 --version

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && python3 -m pip --version
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# Install build and runtime dependencies
COPY setup_files/* /workspace/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-cuda.txt

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BASE IMAGE ####################


#################### WHEEL BUILD IMAGE ####################
FROM base AS wheel-build
ARG PYTHON_VERSION=3.10
ARG USE_SCCACHE
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
ARG NVCC_THREADS=8
ENV NVCC_THREADS=${NVCC_THREADS}
ARG buildkite_commit
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ENV BUILDKITE_COMMIT=${buildkite_commit}
ENV CCACHE_DIR=/root/.cache/ccache

# Install build dependencies
COPY setup_files/* .
COPY csrc csrc
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY pyproject.toml pyproject.toml
COPY vllm vllm
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt

# install compiler cache to speed up compilation leveraging local or remote caching
RUN apt-get update -y && apt-get install -y ccache

# If USE_SCCACHE is set to 1, install sccache and build wheel with it
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && curl -L -o sccache.tar.gz https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=us-west-2 \
        && export CMAKE_BUILD_TYPE=Release \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

# If USE_SCCACHE is not set to 1, build wheel with ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_SCCACHE" != "1" ]; then \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

# check the size of the wheel, we cannot upload wheels larger than 100MB
COPY .buildkite/check-wheel-size.py check-wheel-size.py
RUN python3 check-wheel-size.py dist

#################### WHEEL BUILD IMAGE ####################

#################### DEV IMAGE ####################
FROM base as dev

COPY setup_files/* /workspace/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-dev.txt

#################### DEV IMAGE ####################

#################### MAMBA Build IMAGE ####################
FROM dev as mamba-builder
# max jobs used for build
ARG MAX_JOBS=2
ENV MAX_JOBS=${MAX_JOBS}

WORKDIR /usr/src/mamba
COPY setup_files/requirements-mamba.txt requirements-mamba.txt
# Download the wheel or build it if a pre-compiled release doesn't exist
RUN pip --verbose wheel -r requirements-mamba.txt \
    --no-build-isolation --no-deps --no-cache-dir

#################### MAMBA Build IMAGE ####################

#################### vLLM BASE IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04 AS vllm-base
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo vim \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && python3 --version
    
# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && python3 -m pip --version
RUN apt-get update -y \
    && apt-get install -y libibverbs-dev
RUN ldconfig /usr/local/cuda-$(echo ${CUDA_VERSION} | cut -d. -f1,2)/compat/

# install vllm wheel first, so that torch etc will be installed
RUN --mount=type=bind,from=wheel-build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install dist/*.whl --verbose

RUN --mount=type=bind,from=mamba-builder,src=/usr/src/mamba,target=/usr/src/mamba \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install /usr/src/mamba/*.whl --no-cache-dir

# Install flashinfer. Only support 3.8-3.11
RUN --mount=type=cache,target=/root/.cache/pip \
    . /etc/environment && \
    python3 -m pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.3/flashinfer-0.1.3+cu121torch2.4-cp${PYTHON_VERSION_STR}-cp${PYTHON_VERSION_STR}-linux_x86_64.whl

#################### vLLM BASE IMAGE ####################

#################### TEST IMAGE ####################
FROM vllm-base AS test
ADD . /vllm-workspace/

ARG PYTHON_VERSION_TEST="${PYTHON_VERSION}"
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION_TEST} python${PYTHON_VERSION_TEST}-dev python${PYTHON_VERSION_TEST}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION_TEST} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION_TEST} \
    && ln -sf /usr/bin/python${PYTHON_VERSION_TEST}-config /usr/bin/python3-config \
    && python3 --version

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION_TEST} && python3 -m pip --version
COPY setup_files/* .
# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-test.txt

RUN mkdir test_docs && mv docs test_docs/ && mv vllm test_docs/
#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

ARG PYTHON_VERSION_OPENAI="${PYTHON_VERSION}"
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION_OPENAI} python${PYTHON_VERSION_OPENAI}-dev python${PYTHON_VERSION_OPENAI}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION_OPENAI} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION_OPENAI} \
    && ln -sf /usr/bin/python${PYTHON_VERSION_OPENAI}-config /usr/bin/python3-config \
    && python3 --version
# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION_OPENAI} && python3 -m pip --version

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install accelerate hf_transfer 'modelscope!=1.15.0'

ENV VLLM_USAGE_SOURCE production-docker-image

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
