FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# image to build pytorch extensions
FROM dev AS build

# copy input files
COPY csrc csrc
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY vllm/__init__.py vllm/__init__.py

# max jobs used by Ninja to build extensions
ENV MAX_JOBS=$max_jobs
RUN python3 setup.py build_ext --inplace

# image to run unit testing suite
FROM dev AS test

# copy pytorch extensions separately to avoid having to rebuild
# when python code changes
COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY tests tests
COPY vllm vllm

ENTRYPOINT ["python3", "-m", "pytest", "tests"]

# use CUDA base as CUDA runtime dependencies are already installed via pip
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS vllm-base

# libnccl required for ray
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

FROM vllm-base AS vllm
COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY vllm vllm

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.api_server"]

# openai api server alternative
FROM vllm-base AS vllm-openai
# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate fschat

COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY vllm vllm

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

