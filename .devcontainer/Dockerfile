FROM rocm/dev-ubuntu-22.04:6.2-complete

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    htop \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python python3.11 1 \
    && python3 -m pip install --upgrade pip

# Install PyTorch with ROCm 6.2 support (~2GB, cached as a Docker layer)
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

# Install common ML and dev tools (vLLM-specific deps installed via postCreateCommand)
RUN pip install \
    ninja \
    packaging \
    wheel \
    setuptools \
    transformers \
    tokenizers \
    accelerate \
    numpy \
    scipy \
    ipython \
    pytest \
    ruff

# ROCm environment for AMD RX 6900 XT (gfx1030 / NAVI 21 / RDNA2)
# HSA_OVERRIDE_GFX_VERSION forces ROCm to treat the card as gfx1030
# PYTORCH_ROCM_ARCH ensures compiled kernels target gfx1030
ENV ROCM_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH=gfx1030
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV ROCR_VISIBLE_DEVICES=0
ENV HIP_FORCE_DEV_KERNARG=1
ENV SAFETENSORS_FAST_GPU=1
ENV TOKENIZERS_PARALLELISM=false
# Limit parallel kernel compilation to avoid OOM during build
ENV MAX_JOBS=8

WORKDIR /workspace
