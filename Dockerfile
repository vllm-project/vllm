ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

RUN apt update && apt install git -y && apt install curl -y
 
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install vllm.
WORKDIR /workspace/vllm
RUN uv venv .vllm --python 3.12
RUN . .vllm/bin/activate && VLLM_USE_PRECOMPILED=1 uv pip install -e .

# Checkout a specific commit.
ENV VLLM_SHA=550f8a052cae03c7e14a46767f689ab09c1cc28d
RUN git fetch && git checkout ${VLLM_SHA}

ENTRYPOINT ["/bin/bash"]
