FROM python:3.10 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements-cpu.txt requirements-cpu.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cpu.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# image to build pytorch extensions
FROM dev AS build

# install build dependencies
COPY requirements-build-cpu.txt requirements-build-cpu.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build-cpu.txt

# copy input files
COPY csrc csrc
COPY setup.py setup.py
COPY requirements-cpu.txt requirements-cpu.txt
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
FROM python:3.10 AS dev

# libnccl required for ray
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace
COPY requirements-cpu.txt requirements-cpu.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cpu.txt

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

