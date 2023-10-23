FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip python3-venv

WORKDIR /vllm
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
    
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

FROM dev AS build

ARG max_jobs=4

COPY csrc csrc
COPY vllm vllm
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY MANIFEST.in MANIFEST.in
COPY setup.py setup.py

RUN --mount=type=cache,target=/root/.cache/pip \
    MAX_JOBS=$max_jobs python3 -m build

FROM dev AS test

COPY --from=build /vllm/dist/*.whl .
RUN pip install *.whl
COPY tests tests
 
ENTRYPOINT ["python3", "-m", "pytest", "tests"]

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS prod

RUN apt-get update -y \
    && apt-get install -y python3-pip libnccl2

WORKDIR /vllm
COPY --from=build /vllm/dist/*.whl .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install *.whl

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.api_server"]

