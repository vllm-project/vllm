# This vLLM Dockerfile is used to construct image that can build and run vLLM on x86 CPU platform.

FROM ubuntu:22.04 AS cpu-test-1

ENV CCACHE_DIR=/root/.cache/ccache

ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl ccache git wget vim numactl gcc-12 g++-12 python3 python3-pip libtcmalloc-minimal4 libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html
# intel-openmp provides additional performance improvement vs. openmp
# tcmalloc provides better memory allocation efficiency, e.g, holding memory in caches to speed up access of commonly-used objects.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install intel-openmp==2025.0.1

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so"

RUN echo 'ulimit -c 0' >> ~/.bashrc

RUN pip install intel_extension_for_pytorch==2.5.0

WORKDIR /workspace

ARG PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,src=requirements-build.txt,target=requirements-build.txt \
    pip install --upgrade pip && \
    pip install -r requirements-build.txt

FROM cpu-test-1 AS build

WORKDIR /workspace/vllm

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,src=requirements-common.txt,target=requirements-common.txt \
    --mount=type=bind,src=requirements-cpu.txt,target=requirements-cpu.txt \
    pip install -v -r requirements-cpu.txt

COPY . .
ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != 0 ]; then bash tools/check_repo.sh ; fi

# Support for building with non-AVX512 vLLM: docker build --build-arg VLLM_CPU_DISABLE_AVX512="true" ...
ARG VLLM_CPU_DISABLE_AVX512
ENV VLLM_CPU_DISABLE_AVX512=${VLLM_CPU_DISABLE_AVX512}

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=bind,source=.git,target=.git \
    VLLM_TARGET_DEVICE=cpu python3 setup.py bdist_wheel && \
    pip install dist/*.whl && \
    rm -rf dist

WORKDIR /workspace/

RUN ln -s /workspace/vllm/tests && ln -s /workspace/vllm/examples && ln -s /workspace/vllm/benchmarks

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e tests/vllm_test_utils

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
