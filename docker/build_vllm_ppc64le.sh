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
TORCH_VERSION=${TORCH_VERSION:-2.13.0}

# Parse version parts
TORCH_MAJOR=$(echo "${TORCH_VERSION}" | cut -d. -f1)
TORCH_MINOR=$(echo "${TORCH_VERSION}" | cut -d. -f2)
TORCH_PATCH=$(echo "${TORCH_VERSION}" | cut -d. -f3 | cut -d+ -f1)
TORCH_PATCH=${TORCH_PATCH:-0}

# 1. Calculate torchvision (always minor + 15)
VISION_MINOR=$((TORCH_MINOR + 15))
TORCHVISION_VERSION="0.${VISION_MINOR}.${TORCH_PATCH}"

# 2. Calculate torchaudio (minor - 2 for >= 2.13, otherwise same as torch)
if [ "${TORCH_MAJOR}" -eq 2 ] && [ "${TORCH_MINOR}" -ge 13 ]; then
    AUDIO_MINOR=$((TORCH_MINOR - 2))
    TORCHAUDIO_VERSION="${TORCH_MAJOR}.${AUDIO_MINOR}.${TORCH_PATCH}"
else
    TORCHAUDIO_VERSION="${TORCH_VERSION}"
fi

export TORCH_VERSION
export TORCHVISION_VERSION
export TORCHAUDIO_VERSION

echo "Resolved PyTorch versions:"
echo "  torch:       ${TORCH_VERSION}"
echo "  torchvision: ${TORCHVISION_VERSION}"
echo "  torchaudio:  ${TORCHAUDIO_VERSION}"

OPENCV_REQUIREMENT=$(grep -E '^opencv-python-headless[[:space:]]*(==|>=|<=|~=)' \
    requirements/common.txt |
    sed -E 's/^opencv-python-headless[[:space:]]*//' |
    sed -E 's/[[:space:]]*#.*$//' |
    tr -d ' ' |
    head -n1)

export OPENCV_REQUIREMENT
echo "OpenCV requirement: ${OPENCV_REQUIREMENT}"


XGRAMMAR_VERSION=$(
    sed -nE 's/^xgrammar[[:space:]]*==[[:space:]]*([^,;[:space:]]+).*/\1/p' requirements/common.txt |
    head -n1
)

if [ -z "${XGRAMMAR_VERSION}" ]; then
    XGRAMMAR_VERSION=$(
        sed -nE 's/^xgrammar[[:space:]]*>=[[:space:]]*([^,;[:space:]]+).*/\1/p' requirements/common.txt |
        head -n1
    )
fi

export XGRAMMAR_VERSION

if [ -z "${XGRAMMAR_VERSION}" ]; then
    echo "xgrammar version not found in requirements/common.txt"
    exit 1
fi
echo "XGRAMMAR_VERSION=${XGRAMMAR_VERSION}"

########################################
# install system dependencies
########################################

rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm || true

# Dynamically match openssl-devel version with installed openssl-libs
OPENSSL_DEVEL="openssl-devel-$(rpm -q --queryformat '%{VERSION}-%{RELEASE}' openssl-libs 2>/dev/null || echo 'openssl-devel')"

microdnf install -y \
    "${OPENSSL_DEVEL}" \
    python3.12 python3.12-devel python3.12-pip gcc \
    git jq gcc-toolset-14 gcc-toolset-14-libatomic-devel \
    automake libtool clang-devel \
    harfbuzz-devel kmod lcms2-devel libimagequant-devel libjpeg-turbo-devel \
    llvm15-devel libraqm-devel libtiff-devel libwebp-devel libxcb-devel \
    ninja-build openjpeg2-devel pkgconfig \
    tcl-devel tk-devel xsimd-devel zeromq-devel zlib-devel patchelf file \
    openblas openblas-devel protobuf numactl numactl-devel openmpi openmpi-devel

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
pip install uv
pip install build cmake
uv pip install setuptools==78.1.1 cython meson-python pybind11 "sympy>=1.13.3" --no-build-isolation

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
# Install Torch packages from DevPI or build from source
########################################

is_available_on_devpi() {
    local pkg=$1
    local version=$2

    echo "Checking DevPI for ${pkg}==${version}..."

    if pip index versions "${pkg}" \
        --index-url "${IBM_DEVPI_URL}" 2>/dev/null |
        grep -F "${version}" >/dev/null; then

        echo "${pkg}==${version} is available on DevPI"
        return 0
    fi

    echo "${pkg}==${version} is NOT available on DevPI"
    return 1
}

########################################
# Common packages from DevPI
########################################
uv pip install numpy==2.3.5 pillow --extra-index-url "$IBM_DEVPI_URL"

########################################
# OpenCV
########################################

if [[ "${OPENCV_REQUIREMENT}" == ">=4.13.0" ]]; then
    echo "OpenCV requirement is ${OPENCV_REQUIREMENT}"
    echo "Installing opencv-python-headless from DevPI"

    OPENCV_VERSION="==4.13.0.92+ppc64le1"
else
    echo "Using OpenCV requirement: ${OPENCV_REQUIREMENT}"

    OPENCV_VERSION="${OPENCV_REQUIREMENT}"
fi

try_install_from_devpi "opencv-python-headless${OPENCV_VERSION}"

########################################
# Check Torch / Torchvision / Torchaudio
########################################

TORCH_FROM_DEVPI=false
TORCHVISION_FROM_DEVPI=false
TORCHAUDIO_FROM_DEVPI=false

if is_available_on_devpi "torch" "${TORCH_VERSION}"; then
    TORCH_FROM_DEVPI=true
fi

if is_available_on_devpi "torchvision" "${TORCHVISION_VERSION}"; then
    TORCHVISION_FROM_DEVPI=true
fi

if is_available_on_devpi "torchaudio" "${TORCHAUDIO_VERSION}"; then
    TORCHAUDIO_FROM_DEVPI=true
fi

########################################
# Torch
########################################

if [[ "${TORCH_FROM_DEVPI}" == "true" ]]; then
    echo "Installing torch==${TORCH_VERSION} from DevPI"

    uv pip install \
        --extra-index-url "${IBM_DEVPI_URL}" \
        --index-strategy unsafe-best-match \
        --only-binary=:all: \
        --no-build-isolation \
        "torch==${TORCH_VERSION}"
else
    echo "Torch wheel not available on DevPI. Building torch==${TORCH_VERSION} from source."

    TEMP_BUILD_DIR=$(mktemp -d)
    cd "${TEMP_BUILD_DIR}"

    git clone \
        --recursive \
        https://github.com/pytorch/pytorch.git \
        -b "v${TORCH_VERSION}"

    cd pytorch

    export BLAS=OpenBLAS
    export USE_OPENMP=1
    export USE_MKLDNN=OFF
    export USE_MKLDNN_CBLAS=OFF
    export _GLIBCXX_USE_CXX11_ABI=1
    uv pip install maturin
    pip install -r requirements.txt --extra-index-url "$IBM_DEVPI_URL"
    python setup.py develop
    rm -f dist/torch*+git*whl
    MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    PYTORCH_BUILD_VERSION=${TORCH_VERSION} PYTORCH_BUILD_NUMBER=1 uv build --wheel --out-dir "${WHEEL_DIR}"
    uv pip install "${WHEEL_DIR}"/torch*.whl
    cd "${REPO_ROOT}"
    rm -rf "${TEMP_BUILD_DIR}"
fi

########################################
# Torchvision
########################################

if [[ "${TORCHVISION_FROM_DEVPI}" == "true" ]]; then
    echo "Installing torchvision==${TORCHVISION_VERSION} from DevPI"

    uv pip install \
        --extra-index-url "${IBM_DEVPI_URL}" \
        --index-strategy unsafe-best-match \
        --only-binary=:all: \
        --no-build-isolation \
        "torchvision==${TORCHVISION_VERSION}"
else
    echo "Torchvision wheel not available on DevPI. Building torchvision==${TORCHVISION_VERSION} from source."

    TEMP_BUILD_DIR=$(mktemp -d)
    cd "${TEMP_BUILD_DIR}"

    export TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_FFMPEG=0
    git clone \
        --recursive \
        https://github.com/pytorch/vision.git \
        -b "v${TORCHVISION_VERSION}"

    cd vision
    uv pip install standard-pkg-resources --no-build-isolation
    MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    BUILD_VERSION=${TORCHVISION_VERSION} \
    uv build --wheel --out-dir "${WHEEL_DIR}" --no-build-isolation

    export BUILD_VERSION="${TORCHVISION_VERSION}"

    uv build \
        --wheel \
        --out-dir "${WHEEL_DIR}" \
        --no-build-isolation

    uv pip install "${WHEEL_DIR}"/torchvision*.whl

    cd "${REPO_ROOT}"
    rm -rf "${TEMP_BUILD_DIR}"
fi

########################################
# Torchaudio
########################################

if [[ "${TORCHAUDIO_FROM_DEVPI}" == "true" ]]; then
    echo "Installing torchaudio==${TORCHAUDIO_VERSION} from DevPI"

    uv pip install \
        --extra-index-url "${IBM_DEVPI_URL}" \
        --index-strategy unsafe-best-match \
        --only-binary=:all: \
        --no-build-isolation \
        "torchaudio==${TORCHAUDIO_VERSION}"
else
    echo "Torchaudio wheel not available on DevPI. Building torchaudio==${TORCHAUDIO_VERSION} from source."

    TEMP_BUILD_DIR=$(mktemp -d)
    cd "${TEMP_BUILD_DIR}"

    export BUILD_SOX=1
    export BUILD_KALDI=1
    export BUILD_RNNT=1
    export USE_FFMPEG=0
    export USE_ROCM=0
    export USE_CUDA=0
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_FFMPEG=1

    git clone \
        --recursive \
        https://github.com/pytorch/audio.git \
        -b "v${TORCHAUDIO_VERSION}"

    cd audio

    # Patches required for newer Python versions
    sed -i '
    s|_CSRC_DIR / "_torchaudio.cpp"|str(_CSRC_DIR / "_torchaudio.cpp")|;
    s|_CSRC_DIR / "utils.cpp"|str(_CSRC_DIR / "utils.cpp")|;
    s|sources=\[_CSRC_DIR / s for s in sources\]|sources=[str(_CSRC_DIR / s) for s in sources]|;
    ' tools/setup_helpers/extension.py

    MAX_JOBS="${MAX_JOBS}" \
    BUILD_VERSION="${TORCHAUDIO_VERSION}" \
    uv build \
        --wheel \
        --out-dir "${WHEEL_DIR}" \
        --no-build-isolation

    uv pip install "${WHEEL_DIR}"/torchaudio*.whl

    cd "${REPO_ROOT}"
    rm -rf "${TEMP_BUILD_DIR}"
fi

########################################
# Xgrammar
########################################
XGRAMMAR_FROM_DEVPI=false

if is_available_on_devpi "xgrammar" "${XGRAMMAR_VERSION}"; then
    XGRAMMAR_FROM_DEVPI=true
fi

if [[ "${XGRAMMAR_FROM_DEVPI}" == "true" ]]; then
    echo "Installing xgrammar==${XGRAMMAR_VERSION} from DevPI"

    uv pip install \
        --extra-index-url "${IBM_DEVPI_URL}" \
        --index-strategy unsafe-best-match \
        --only-binary=:all: \
        --no-build-isolation \
        "xgrammar==${XGRAMMAR_VERSION}"
else
    echo "Xgrammar wheel not available on DevPI. Building xgrammar==${XGRAMMAR_VERSION} from source."
    # Install xgrammar build dependencies only when building from source.
    uv pip install \
        "scikit-build-core==0.11.6" \
        "pyproject-metadata<0.8" \
        pathspec \
        packaging \
        distro \
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
fi

NUMBA_REQUIREMENT=$(grep -E '^numba[[:space:]]*(==|>=|<=|~=)' \
    requirements/cpu.txt |
    sed -E 's/^numba[[:space:]]*//' |
    sed -E 's/[[:space:]]*#.*$//' |
    sed -E 's/;.*$//' |
    tr -d ' ' |
    head -n1)

echo "Numba requirement: ${NUMBA_REQUIREMENT}"
try_install_from_devpi "numba${NUMBA_REQUIREMENT}"

########################################
# install built wheels
########################################
uv pip install setuptools_scm maturin setuptools-rust ninja scikit-build-core pybind11 nanobind \
    --no-build-isolation

########################################
# install remaining deps
########################################

sed -i.bak -e 's/.*torch.*//g' pyproject.toml requirements/*.txt
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:/usr/lib64/pkgconfig

uv pip install -r requirements/common.txt \
               -r requirements/cpu.txt \
               -r requirements/build/cpu.txt --index-strategy unsafe-best-match


