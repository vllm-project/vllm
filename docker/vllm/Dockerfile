FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip libnccl2

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install vllm

EXPOSE 8000

