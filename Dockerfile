FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip python3-venv

WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
    
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

FROM dev AS build

COPY csrc csrc
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY vllm/__init__.py vllm/__init__.py

ENV MAX_JOBS=$max_jobs 
RUN python3 setup.py build_ext --inplace

FROM dev AS test

COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY tests tests
COPY vllm vllm

ENTRYPOINT ["python3", "-m", "pytest", "tests"]

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS api_server

RUN apt-get update -y \
    && apt-get install -y python3-pip libnccl2
WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY vllm vllm

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.api_server"]

