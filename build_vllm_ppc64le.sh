#!/bin/bash
set -eoux pipefail

########################################
# Resolve repo root (IMPORTANT)
########################################
REPO_ROOT="$(pwd)"

cd "$REPO_ROOT"

########################################
# DevPI configuration
########################################

IBM_DEVPI_URL=${IBM_DEVPI_URL:-"https://wheels.developerfirst.ibm.com/ppc64le/linux/+simple/"}
RHOAI_INDEX_URL=${RHOAI_INDEX_URL:-"https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple/"}

########################################
# wheel dir
########################################

WHEEL_DIR=${WHEEL_DIR:-"/tmp/wheels"}
mkdir -p "$WHEEL_DIR"

########################################
# Helpers
########################################
try_install_from_devpi() {
    local pkg=$1
    uv pip install \
        --extra-index-url "${IBM_DEVPI_URL}" \
        --index-strategy unsafe-best-match \
        --no-build-isolation \
        "${pkg}"
}

########################################
# Package Versions
########################################
cd "$REPO_ROOT"
TORCH_VERSION=${TORCH_VERSION:-$(grep -E '^torch==.+==\s*"ppc64le"' requirements/cpu.txt | grep -Eo '\b[0-9\.]+\b' || true)}
TORCH_VERSION=${TORCH_VERSION:-2.11.0}

TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.26.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-${TORCH_VERSION}}

export TORCH_VERSION
export TORCHVISION_VERSION
export TORCHAUDIO_VERSION
export OPENCV_VERSION=${OPENCV_VERSION:-4.13.0.92}
export XGRAMMAR_VERSION=${XGRAMMAR_VERSION:-0.2.1}

########################################
# install system dependencies
########################################

rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm || true

microdnf install -y \
    python3.12 python3.12-devel python3.12-pip gcc \
    git jq gcc-toolset-14 gcc-toolset-14-libatomic-devel \
    automake libtool clang-devel openssl-devel \
    harfbuzz-devel kmod lcms2-devel libimagequant-devel libjpeg-turbo-devel \
    llvm15-devel libraqm-devel libtiff-devel libwebp-devel libxcb-devel \
    ninja-build openjpeg2-devel pkgconfig \
    tcl-devel tk-devel xsimd-devel zeromq-devel zlib-devel patchelf file openblas openblas-devel protobuf numactl numactl-devel openmpi openmpi-devel
    
rpm -ivh --nodeps \
    https://mirror.stream.centos.org/9-stream/CRB/ppc64le/os/Packages/protobuf-lite-devel-3.14.0-17.el9.ppc64le.rpm

rpm -ivh --nodeps \
    https://mirror.stream.centos.org/9-stream/CRB/ppc64le/os/Packages/protobuf-devel-3.14.0-17.el9.ppc64le.rpm

rpm -ivh --nodeps \
    https://mirror.stream.centos.org/9-stream/CRB/ppc64le/os/Packages/protobuf-compiler-3.14.0-17.el9.ppc64le.rpm

########################################
# Python 3.12 virtual environment
########################################

python3.12 -m venv /opt/vllm
source /opt/vllm/bin/activate

export PATH=/opt/vllm/bin:$PATH

python --version

########################################
# install build tools (stable uv)
########################################

pip install -U pip setuptools-rust
pip install uv
pip install "setuptools<70" build wheel cmake auditwheel
uv pip install "setuptools<70" cython meson-python pybind11 "sympy>=1.13.3" --no-build-isolation

########################################
# Rust
########################################

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source /root/.cargo/env

########################################
# Compiler env
########################################

source /opt/rh/gcc-toolset-14/enable

export PATH=/usr/lib64/llvm15/bin:$PATH
export LLVM_CONFIG=/usr/lib64/llvm15/bin/llvm-config
export CMAKE_ARGS="-DPython3_EXECUTABLE=python"

export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1

########################################
# Install Packages From Devpi
########################################
uv pip install numpy==2.3.5 pillow==12.2.0 --extra-index-url "$IBM_DEVPI_URL"
try_install_from_devpi "opencv-python-headless==${OPENCV_VERSION}"
try_install_from_devpi "torch==${TORCH_VERSION}"
try_install_from_devpi "torchvision==${TORCHVISION_VERSION}"

########################################
# torch audio
########################################

TEMP_BUILD_DIR=$(mktemp -d)
cd "${TEMP_BUILD_DIR}"
export BUILD_SOX=1 BUILD_KALDI=1 BUILD_RNNT=1 USE_FFMPEG=0 USE_ROCM=0 USE_CUDA=0
export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_FFMPEG=1
git clone --recursive https://github.com/pytorch/audio.git -b v${TORCHAUDIO_VERSION}
cd audio
#patching 
sed -i '
s|_CSRC_DIR / "_torchaudio.cpp"|str(_CSRC_DIR / "_torchaudio.cpp")|;
s|_CSRC_DIR / "utils.cpp"|str(_CSRC_DIR / "utils.cpp")|;
s|sources=\[_CSRC_DIR / s for s in sources\]|sources=[str(_CSRC_DIR / s) for s in sources]|;
' tools/setup_helpers/extension.py
MAX_JOBS=${MAX_JOBS:-$(nproc)} \
BUILD_VERSION=${TORCHAUDIO_VERSION} \
uv build --wheel --out-dir "${WHEEL_DIR}" --no-build-isolation
uv pip install "${WHEEL_DIR}"/torchaudio*.whl
cd "${REPO_ROOT}"
rm -rf "${TEMP_BUILD_DIR}"

########################################
# Xgrammar
########################################
uv pip install \
   "scikit-build-core==0.11.6" \
   "pyproject-metadata<0.8" \
    pathspec \
    packaging \
    distro \
   "setuptools<70" \
    setuptools_scm \
    cmake \
    ninja \
    pybind11 \
    nanobind
uv pip install apache-tvm-ffi==0.1.12 \
  --no-build-isolation \
  --no-cache

TEMP_BUILD_DIR=$(mktemp -d)

pushd "${TEMP_BUILD_DIR}"

export CFLAGS="-fno-lto -mcpu=power9"
export CXXFLAGS="-fno-lto -mcpu=power9"
export LDFLAGS="-fno-lto"
export PATH=/opt/vllm/bin:$PATH

export Python_EXECUTABLE=/opt/vllm/bin/python3
export Python3_EXECUTABLE=/opt/vllm/bin/python3
export PYTHON_EXECUTABLE=/opt/vllm/bin/python3

export Python_ROOT_DIR=/opt/vllm
export Python3_ROOT_DIR=/opt/vllm

git clone \
    --recursive \
    https://github.com/mlc-ai/xgrammar \
    -b "v${XGRAMMAR_VERSION}"

cd xgrammar

cp cmake/config.cmake .
export PYTHONPATH=/opt/vllm/lib64/python3.12/site-packages:/opt/vllm/lib/python3.12/site-packages:${PYTHONPATH:-}

uv build \
    --wheel \
    --out-dir "${WHEEL_DIR}" \
    --no-build-isolation

uv pip install "${WHEEL_DIR}"/xgrammar*.whl -v

popd

rm -rf "${TEMP_BUILD_DIR}"
cd "${REPO_ROOT}"

########################################
# RHOAI Binary Downloads
########################################
pip download \
    --index-url "${RHOAI_INDEX_URL}" \
    --only-binary=:all: \
    --no-deps \
    llvmlite==0.47.0 \
    -d "${WHEEL_DIR}"

pip download \
    --index-url "${RHOAI_INDEX_URL}" \
    --only-binary=:all: \
    --no-deps \
    Numba==0.65.0 \
    -d "${WHEEL_DIR}"

########################################
# install built wheels
########################################
uv pip install setuptools_scm maturin setuptools-rust ninja scikit-build-core pybind11 nanobind \
    --no-build-isolation
uv pip install "${WHEEL_DIR}"/*.whl

########################################
# install remaining deps
########################################

sed -i.bak -e 's/.*torch.*//g' pyproject.toml requirements/*.txt

uv pip install "setuptools>=78.1.1" --no-build-isolation

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:/usr/lib64/pkgconfig

uv pip install -r requirements/common.txt \
               -r requirements/cpu.txt \
               -r requirements/build/cpu.txt --index-strategy unsafe-best-match
