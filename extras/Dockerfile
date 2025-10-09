# vLLM Development Container with GPU Support
# Uses vLLM's own requirements for automatic dependency management

# Build-time args to control CUDA/OS base and PyTorch nightly index
ARG CUDA_VERSION=13.0.1
ARG UBI_VERSION=10
ARG TORCH_CUDA_INDEX=cu130
# Base flavor for CUDA image: e.g. 'rockylinux9' (default) or 'ubi9'
ARG BASE_FLAVOR=rockylinux10
ARG INSTALL_CUDA_OPTIONAL_DEVEL=1
ARG CUDNN_FLAVOR=9

# Switchable base: defaults to Rocky Linux to avoid subscription-gated repos
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-${BASE_FLAVOR}

# Set CUDA environment variables for build tools
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
ENV CUDNN_LIBRARY_PATH=/usr/lib64
ENV CUDNN_INCLUDE_PATH=/usr/include

# Install system packages with additional CUDA development libraries
RUN dnf update -y && dnf install --allowerasing -y \
    python3 python3-pip python3-devel \
    git gcc gcc-c++ cmake \
    make patch which findutils tar rsync \
    wget curl vim nano pkgconfig \
    zlib-devel bzip2 bzip2-devel xz xz-devel libffi-devel \
    openssl-devel sqlite-devel \
    && (dnf install -y readline-devel || true) \
    && dnf clean all

# Prefer Python 3.12 from packages if available (fallback to system python3)
## Ensure /usr/bin/python exists for compatibility
RUN ln -sf $(command -v python3) /usr/bin/python || true

# Create a non-root user for development
RUN useradd -m -s /bin/bash vllmuser && \
    echo "vllmuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install essential system tools
RUN dnf install -y hostname iproute iputils

ARG REQUIRE_FFMPEG=1
# Multimedia and image libs with optional ffmpeg-devel enforcement
# Install EPEL and RPM Fusion repos for EL (9/10) and pull ffmpeg/ffmpeg-devel from there.
# When REQUIRE_FFMPEG=1, fail the build if ffmpeg is still unavailable.
RUN set -euxo pipefail \
        && (dnf install -y dnf-plugins-core || true) \
        && (dnf config-manager --set-enabled crb || true) \
        && (dnf makecache -y || true) \
    && . /etc/os-release \
    && ELVER="${VERSION_ID%%.*}" \
    && echo "[Dockerfile] Detected Enterprise Linux major version: ${ELVER}" \
        && dnf install -y \
                libjpeg-turbo-devel libpng-devel zlib-devel freetype-devel \
                libsndfile libsndfile-devel sox sox-devel || true \
        && if [ "${REQUIRE_FFMPEG}" = "1" ]; then \
         echo "[Dockerfile] Enabling EPEL and RPM Fusion for ffmpeg (EL${ELVER})"; \
         dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-${ELVER}.noarch.rpm; \
         dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-${ELVER}.noarch.rpm; \
         dnf install -y https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-${ELVER}.noarch.rpm; \
                 dnf makecache -y; \
                 dnf install -y ffmpeg ffmpeg-devel; \
                 command -v ffmpeg >/dev/null 2>&1; \
             else \
                 # Best-effort install when not enforced
         (dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-${ELVER}.noarch.rpm || true); \
         (dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-${ELVER}.noarch.rpm || true); \
         (dnf install -y https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-${ELVER}.noarch.rpm || true); \
                 (dnf makecache -y || true); \
                 (dnf install -y ffmpeg ffmpeg-devel || true); \
             fi \
        && (dnf install -y --enablerepo=crb ninja-build || \
            dnf install -y --enablerepo=crb ninja || \
            dnf install -y ninja-build || \
            dnf install -y ninja || true) \
    && dnf clean all || true

# Add NVIDIA Machine Learning repo for RHEL9/UBI9 and install NCCL runtime/devel
# Needed for PyTorch nightly cu129 to avoid ncclCommWindowRegister symbol errors
# Install NCCL runtime/devel from the CUDA repository available in the base image
RUN set -euxo pipefail \
    && dnf makecache -y \
    && (dnf install -y libnccl libnccl-devel || dnf install -y libnccl-2 libnccl-devel-2) \
    && dnf clean all

# Optional CUDA developer libraries (cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND, cuSPARSElt, cuDNN, NVRTC, CUPTI, NVTX, NVJITLINK)
RUN if [ "${INSTALL_CUDA_OPTIONAL_DEVEL}" = "1" ]; then \
        set -euxo pipefail; \
        CUDA_MAJOR="$(printf '%s' "${CUDA_VERSION}" | cut -d. -f1)"; \
        CUDA_MINOR="$(printf '%s' "${CUDA_VERSION}" | cut -d. -f2)"; \
    CUDA_SLOT="${CUDA_MAJOR}-${CUDA_MINOR}"; \
        CUDNN_PKGS=""; \
        if [ "${CUDNN_FLAVOR}" = "8" ]; then \
            CUDNN_PKGS="libcudnn8 libcudnn8-devel"; \
        else \
            CUDNN_PKGS="libcudnn${CUDNN_FLAVOR}-cuda-${CUDA_MAJOR} libcudnn${CUDNN_FLAVOR}-devel-cuda-${CUDA_MAJOR}"; \
            if dnf list --available "libcudnn${CUDNN_FLAVOR}-headers-cuda-${CUDA_MAJOR}" >/dev/null 2>&1; then \
                CUDNN_PKGS="${CUDNN_PKGS} libcudnn${CUDNN_FLAVOR}-headers-cuda-${CUDA_MAJOR}"; \
            fi; \
        fi; \
        dnf install -y \
            "libcublas-${CUDA_SLOT}" "libcublas-devel-${CUDA_SLOT}" \
            "libcurand-${CUDA_SLOT}" "libcurand-devel-${CUDA_SLOT}" \
            "libcusolver-${CUDA_SLOT}" "libcusolver-devel-${CUDA_SLOT}" \
            "libcufft-${CUDA_SLOT}" "libcufft-devel-${CUDA_SLOT}" \
            "libcusparse-${CUDA_SLOT}" "libcusparse-devel-${CUDA_SLOT}" \
            "libcufile-${CUDA_SLOT}" "libcufile-devel-${CUDA_SLOT}" \
            "libcusparselt0-cuda-${CUDA_MAJOR}" "libcusparselt0-devel-cuda-${CUDA_MAJOR}" \
            "cuda-nvrtc-${CUDA_SLOT}" "cuda-nvrtc-devel-${CUDA_SLOT}" \
            "cuda-cupti-${CUDA_SLOT}" \
            "cuda-nvtx-${CUDA_SLOT}" \
            "libnvjitlink-${CUDA_SLOT}" "libnvjitlink-devel-${CUDA_SLOT}" \
            ${CUDNN_PKGS}; \
        if dnf list --available "libcurand-static-${CUDA_SLOT}" >/dev/null 2>&1; then \
            dnf install -y "libcurand-static-${CUDA_SLOT}" || true; \
        fi; \
        dnf clean all; \
    fi

# Set working directory and adjust ownership
WORKDIR /workspace
RUN chown -R vllmuser:vllmuser /workspace

# Create build directories with proper permissions
RUN mkdir -p /workspace/.deps && chown -R vllmuser:vllmuser /workspace/.deps && \
    mkdir -p /tmp/vllm-build && chmod 777 /tmp/vllm-build && \
    mkdir -p /opt/work && chmod 777 /opt/work && \
    mkdir -p /home/vllmuser/.cache && chown -R vllmuser:vllmuser /home/vllmuser/.cache && \
    mkdir -p /home/vllmuser/.ccache && chown -R vllmuser:vllmuser /home/vllmuser/.ccache && \
    mkdir -p /home/vllmuser/.cmake && chown -R vllmuser:vllmuser /home/vllmuser/.cmake && \
    chmod -R 755 /workspace && \
    chmod -R 777 /tmp

# Switch to the non-root user
USER vllmuser

# Create and activate virtual environment using the best available Python (3.12 preferred)
ENV VIRTUAL_ENV=/home/vllmuser/venv
RUN PY_BIN="$(command -v python3.12 || command -v python3)" && "$PY_BIN" -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set pip configuration
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=5
ENV PIP_PREFER_BINARY=1

# CUDA arch list: CUDA 13+ drops SM70/SM75; default to supported archs only.
# Override at build time with: --build-arg TORCH_CUDA_ARCH_LIST="..."
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0 13.0"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Also set CUDAARCHS (semicolon separated) for CMake/NVCC generators.
# Override at build time with: --build-arg CUDA_ARCHS="80;86;89;90;120"
ARG CUDA_ARCHS="80;86;89;90;120"
ENV CUDAARCHS="${CUDA_ARCHS}"

# Upgrade pip and setuptools to latest versions
RUN pip install --upgrade pip setuptools>=61 wheel

COPY requirements/ /tmp/requirements/

# Install PyTorch nightly first (includes latest GPU arch support such as Blackwell sm_120 when present)
ARG TORCH_CUDA_INDEX
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/${TORCH_CUDA_INDEX}
RUN pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/${TORCH_CUDA_INDEX}
RUN pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/${TORCH_CUDA_INDEX}

# Install PyAV for torchvision video I/O (read_video) compatibility
RUN pip install --upgrade av

# Install TorchCodec to support torchaudio.load on recent nightlies
RUN set -euxo pipefail \
    && (pip install --pre torchcodec \
        || pip install torchcodec \
        || pip install --no-deps 'git+https://github.com/pytorch/torchcodec@main')

# Install modern build tools and vLLM's build dependencies and CUDA deps early,
# but sanitize requirements to avoid downgrading torch-family or forcing xformers pins.
COPY pyproject.toml /tmp/pyproject.toml
RUN set -euxo pipefail \
        && cd /tmp \
        && pip install "setuptools>=61" "setuptools-scm>=8" build wheel ninja cmake \
        && mkdir -p /tmp/requirements_sanitized \
        && for f in build.txt cuda.txt common.txt; do \
                 if [ -f "/tmp/requirements/$f" ]; then \
                     sed -E '/^(torch|torchvision|torchaudio|xformers)\b/Id' "/tmp/requirements/$f" > "/tmp/requirements_sanitized/$f"; \
                 fi; \
             done \
    && pip install --pre \
        -r /tmp/requirements_sanitized/build.txt \
        -r /tmp/requirements_sanitized/cuda.txt \
        -r /tmp/requirements_sanitized/common.txt \
    && pip install --pre --upgrade \
        torch --index-url https://download.pytorch.org/whl/nightly/${TORCH_CUDA_INDEX}

# Install minimal development extras
RUN pip install pytest pytest-asyncio ipython

# Note: vLLM will be installed from source in development mode via dev-setup.sh
# This ensures compatibility with the PyTorch nightly build

# Create activation script for easy virtual environment access
RUN echo '#!/bin/bash' > /home/vllmuser/activate_venv.sh && \
    echo 'source /home/vllmuser/venv/bin/activate' >> /home/vllmuser/activate_venv.sh && \
    echo 'echo "Virtual environment activated: $VIRTUAL_ENV"' >> /home/vllmuser/activate_venv.sh && \
    echo 'echo "Python version: $(python --version)"' >> /home/vllmuser/activate_venv.sh && \
    echo 'echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"' >> /home/vllmuser/activate_venv.sh && \
    echo 'echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"' >> /home/vllmuser/activate_venv.sh && \
    chmod +x /home/vllmuser/activate_venv.sh

# Ensure virtual environment is activated in .bashrc
RUN echo 'source /home/vllmuser/venv/bin/activate' >> /home/vllmuser/.bashrc && \
    echo 'echo "🐍 Python virtual environment activated"' >> /home/vllmuser/.bashrc && \
    echo 'echo "🚀 Ready for vLLM development!"' >> /home/vllmuser/.bashrc

# Create development helper script that uses current workspace requirements
RUN echo '#!/bin/bash' > /home/vllmuser/setup_vllm_dev.sh && \
    echo 'echo "🔧 Setting up vLLM for development..."' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'cd /workspace' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo '# Use temporary build directory to avoid permission issues' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'export TMPDIR=/tmp/vllm-build' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'mkdir -p "$TMPDIR" && chmod 777 "$TMPDIR"' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'export CMAKE_BUILD_PARALLEL_LEVEL=4' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'export VLLM_INSTALL_PUNICA_KERNELS=0' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'export MAX_JOBS=4' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo '# Install current workspace requirements first' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'if [ -f requirements/common.txt ]; then pip install -r requirements/common.txt; fi' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo '# Use temporary directory for CMake build files' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'FETCHCONTENT_BASE_DIR="$TMPDIR/deps" pip install -e . --no-deps --no-build-isolation --verbose' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'echo "✅ vLLM installed in editable mode!"' >> /home/vllmuser/setup_vllm_dev.sh && \
    echo 'python -c "import vllm; print(\"vLLM version:\", vllm.__version__)"' >> /home/vllmuser/setup_vllm_dev.sh && \
    chmod +x /home/vllmuser/setup_vllm_dev.sh

# Provide a helper to apply repo patches against the mounted /workspace
# Create under /usr/local/bin as root, then switch back to non-root user
USER root
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd /workspace 2>/dev/null || exit 0' \
    'SCRIPT=./extras/patches/apply_patches.sh' \
    'if [ -f "$SCRIPT" ]; then' \
    '  echo "[apply-patches] Running $SCRIPT"' \
    '  # Copy to temp and normalize EOL to avoid permission errors on mounted FS' \
    '  TMP_SCRIPT=$(mktemp /tmp/apply_patches.XXXXXX.sh)' \
    '  tr -d '\''\r'\'' < "$SCRIPT" > "$TMP_SCRIPT" || cp "$SCRIPT" "$TMP_SCRIPT"' \
    '  chmod +x "$TMP_SCRIPT"' \
    '  bash "$TMP_SCRIPT" || {' \
    '    echo "[apply-patches] Warning: patch apply failed (continuing)" >&2; exit 0; }' \
    'fi' \
    > /usr/local/bin/apply-vllm-patches && \
    chmod +x /usr/local/bin/apply-vllm-patches
USER vllmuser

# Add environment variables for better CUDA memory management and build optimization
# Use the new variable name to avoid deprecation warnings.
# (Not working with vllm)
# ENV PYTORCH_ALLOC_CONF=expandable_segments:True
#
# Do not pin a single GPU here; let runtime inject device selection
# ENV CUDA_VISIBLE_DEVICES=0
ENV CMAKE_BUILD_PARALLEL_LEVEL=4
ENV VLLM_INSTALL_PUNICA_KERNELS=0
ENV MAX_JOBS=4

# Enable ccache for faster rebuilds
ENV CCACHE_DIR=/home/vllmuser/.ccache
ENV CCACHE_MAXSIZE=10G
ENV PATH=/usr/lib64/ccache:$PATH

# (TORCH_CUDA_ARCH_LIST defined earlier)
# Do not force-disable Machete; allow upstream defaults. User may still pass -e CMAKE_ARGS for custom CMake settings.
ENV CMAKE_ARGS=""

# WSL2-specific CUDA environment configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/lib/wsl/drivers:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH

# Add runtime library detection script
RUN echo '#!/bin/bash' > /home/vllmuser/check_cuda_libs.sh && \
    echo 'echo "=== CUDA Library Check ==="' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'echo "Searching for CUDA libraries..."' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'find /usr/lib/wsl -name "libcuda.so*" 2>/dev/null | head -3 || echo "No WSL CUDA libs"' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'ldconfig -p | grep cuda | head -3 || echo "No CUDA in ldconfig"' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'echo "--- Optional CUDA Libraries ---"' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'libs=(libcublas libcufft libcusolver libcusparse libcurand libcudnn libcusparselt libnvrtc libcupti libnvtx libnvjitlink)' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'for entry in "${libs[@]}"; do' >> /home/vllmuser/check_cuda_libs.sh && \
    echo '  if ldconfig -p | grep -q "$entry"; then' >> /home/vllmuser/check_cuda_libs.sh && \
    echo '    echo "  ✓ $entry"' >> /home/vllmuser/check_cuda_libs.sh && \
    echo '  else' >> /home/vllmuser/check_cuda_libs.sh && \
    echo '    echo "  ✗ $entry (not found in ldconfig cache)"' >> /home/vllmuser/check_cuda_libs.sh && \
    echo '  fi' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'done' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'echo "PyTorch CUDA status:"' >> /home/vllmuser/check_cuda_libs.sh && \
    echo 'python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"Device count: {torch.cuda.device_count()}\")" 2>/dev/null || echo "PyTorch not available"' >> /home/vllmuser/check_cuda_libs.sh && \
    chmod +x /home/vllmuser/check_cuda_libs.sh
