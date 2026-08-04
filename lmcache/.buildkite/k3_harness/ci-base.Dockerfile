# CI base image: CUDA + Python + uv + build deps.
# No vLLM or LMCache — those are installed per-job by setup-env.sh.
#
# Built automatically by setup-cluster.sh and imported into K3s containerd.
# Rebuild when requirements/*.txt changes.

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache software-properties-common git curl sudo jq lsof \
        python3 python3-dev python3-venv python3-pip tzdata libxcb1-dev \
        libcudart12 \
    && ldconfig \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/ \
    && mv ~/.local/bin/uvx /usr/local/bin/ \
    && uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pre-install requirements that rarely change
COPY requirements/common.txt requirements/build.txt requirements/cuda.txt \
     requirements/cuda13_core.txt requirements/nixl.txt /tmp/reqs/
RUN . /opt/venv/bin/activate && \
    uv pip install -r /tmp/reqs/cuda.txt && \
    uv pip install -r /tmp/reqs/build.txt && \
    rm -rf /tmp/reqs

# Set at build time to match the CI machine's GPU.
# Query with: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
