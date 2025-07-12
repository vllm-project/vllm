ARG CUDA_VERSION=12.8.1
from nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

RUN wget -qO- https://astral.sh/uv/install.sh | sh
 
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git && \
    VLLM_USE_PRECOMPILED=1 uv pip install -e .

WORKDIR /workspace/vllm
ENV VLLM_SHA=8ce3cad72fbd0dc6524e495ecddbbc58fd8fd09e
RUN git fetch && git checkout ${VLLM_SHA}
