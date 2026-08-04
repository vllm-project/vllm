#!/bin/bash
# Build the LMCache ROCm/HIP wheel for AMD Instinct gfx942 + gfx950.
#
# Runs inside rocm/dev-ubuntu-22.04:<rocm>-complete. No GPU required to
# compile. Produces a manylinux-tagged wheel in /work/LMCache/dist that
# excludes torch + ROCm runtime libs (bound at runtime by the host image).
#
# Env (set by the workflow, with sensible defaults for local runs):
#   PYTORCH_ROCM_ARCH          gfx target list      (default gfx942;gfx950)
#   TORCH_ROCM_SPEC            pip torch spec        (default torch==2.11.0+rocm7.2)
#   TORCH_ROCM_INDEX          torch wheel index     (default rocm7.2 index)
#   SETUPTOOLS_SCM_PRETEND_VERSION  wheel version   (default 0.0.0.dev0)
set -euxo pipefail

PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"
TORCH_ROCM_SPEC="${TORCH_ROCM_SPEC:-torch==2.11.0+rocm7.2}"
TORCH_ROCM_INDEX="${TORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends \
    software-properties-common curl ca-certificates patchelf git
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -q
# Match the upstream vllm/vllm-openai-rocm interpreter (cp312).
apt-get install -y -q --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# The repo is bind-mounted from the CI runner (owned by a non-root UID) while
# this container runs as root, so git refuses to operate on it. setup.py's
# version/git introspection runs during the wheel build, so mark it safe.
git config --global --add safe.directory /work/LMCache

PY=python3.12
$PY --version

# Build against the public ROCm torch whose ABI matches the vLLM image.
$PY -m pip install --no-cache-dir "${TORCH_ROCM_SPEC}" --index-url "${TORCH_ROCM_INDEX}"
$PY -m pip install --no-cache-dir \
    ninja "setuptools>=77.0.3,<81.0.0" setuptools_scm wheel pybind11 auditwheel
$PY -c 'import torch; print("BUILD TORCH:", torch.__version__, "hip:", torch.version.hip, "cxx11abi:", torch._C._GLIBCXX_USE_CXX11_ABI)'

cd /work/LMCache
rm -rf build dist dist_rocm csrc_hip
# Clean stale hipify artifacts so the build is reproducible.
find csrc -name '*_hip.*' -delete 2>/dev/null || true
find csrc -name '*.hip' -delete 2>/dev/null || true

export BUILD_WITH_HIP=1
export CXX=hipcc
export PYTORCH_ROCM_ARCH
$PY setup.py bdist_wheel --dist-dir=dist_rocm

# Repair: bundle generic userspace libs, but exclude torch and every
# ROCm/driver-coupled runtime lib so they bind to the host vLLM image at
# runtime (same policy as the cu129 wheel excluding libcudart). Globs match
# versioned SONAMEs (e.g. libamdhip64.so.7) -- an unversioned "libamdhip64.so"
# exclude silently misses them and the HIP runtime gets vendored into the
# wheel, which then couples to the build host's KFD driver.
$PY -m auditwheel repair \
    --plat manylinux_2_35_x86_64 \
    --exclude 'libtorch*.so*' \
    --exclude 'libc10*.so*' \
    --exclude 'libamdhip64.so*' \
    --exclude 'libhsa-runtime64.so*' \
    --exclude 'librocprofiler-register.so*' \
    --exclude 'libamd_comgr.so*' \
    --exclude 'librocm-core.so*' \
    --exclude 'librocblas.so*' \
    --exclude 'libhipblas.so*' \
    --exclude 'libMIOpen.so*' \
    --exclude 'libdrm.so*' \
    --exclude 'libdrm_amdgpu.so*' \
    -w dist dist_rocm/*.whl

echo "=== ROCm code-object targets in the built extension ==="
python3 -c "import zipfile,glob,sys; w=glob.glob('dist/*.whl')[0]; zipfile.ZipFile(w).extractall('/tmp/whcheck')"
SO=$(find /tmp/whcheck -name 'c_ops*.so')
/opt/rocm/llvm/bin/llvm-objdump --offloading "$SO" 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | sort -u

echo "=== final ROCm wheel ==="
ls -la /work/LMCache/dist/
