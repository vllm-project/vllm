ARG CUDA_VERSION=12.8.1
from nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

RUN wget -qO- https://astral.sh/uv/install.sh | sh
 
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git && \
    VLLM_USE_PRECOMPILED=1 uv pip install -e .

WORKDIR /workspace/vllm
ENV VLLM_SHA=270d05d9fdf9fc68767056204a1fee078358b122
RUN git fetch && git checkout VLLM_SHA
