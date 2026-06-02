#!/bin/bash
set -eoux pipefail

CURDIR=$(pwd)

########################################
# DevPI configuration
########################################

DEVPI_URL=${DEVPI_URL:-"https://wheels.developerfirst.ibm.com/ppc64le/linux/+simple/"}

if [[ -n "$DEVPI_URL" ]]; then
    echo "Using DevPI index: $DEVPI_URL"
fi

########################################
# install development packages
########################################

rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

microdnf install -y \
    git \
    jq \
    gcc \
    gcc-c++ \
    gcc-toolset-14 \
    gcc-toolset-14-libatomic-devel \
    automake \
    libtool \
    clang-devel \
    openssl-devel \
    freetype-devel \
    fribidi-devel \
    harfbuzz-devel \
    kmod \
    lcms2-devel \
    libimagequant-devel \
    libjpeg-turbo-devel \
    llvm15-devel \
    libraqm-devel \
    libtiff-devel \
    libwebp-devel \
    libxcb-devel \
    ninja-build \
    openjpeg2-devel \
    pkgconfig \
    protobuf* \
    tcl-devel \
    tk-devel \
    xsimd-devel \
    zeromq-devel \
    zlib-devel \
    patchelf \
    file \
    cmake \
    make \
    python3.12-devel

rpm -ivh --nodeps \
https://mirror.stream.centos.org/9-stream/CRB/ppc64le/os/Packages/protobuf-lite-devel-3.14.0-17.el9.ppc64le.rpm

rpm -ivh --nodeps \
https://mirror.stream.centos.org/9-stream/CRB/ppc64le/os/Packages/protobuf-devel-3.14.0-17.el9.ppc64le.rpm

rpm -ivh --nodeps \
https://mirror.stream.centos.org/9-stream/CRB/ppc64le/os/Packages/protobuf-compiler-3.14.0-17.el9.ppc64le.rpm
########################################
# install rust
########################################

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

source /opt/rh/gcc-toolset-14/enable
export PATH=/opt/rh/gcc-toolset-14/root/usr/bin:$PATH

export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc

export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
source /root/.cargo/env

########################################
# compiler environment
########################################

export PATH=/usr/lib64/llvm15/bin:$PATH
export LLVM_CONFIG=/usr/lib64/llvm15/bin/llvm-config

export CMAKE_ARGS="-DPython3_EXECUTABLE=python"

export MAX_JOBS=${MAX_JOBS:-$(nproc)}

export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1

export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++

########################################
# python build tools
########################################

python3.12 -m pip install -U \
    pip \
    uv \
    setuptools \
    wheel \
    build \
    auditwheel \
    cmake \
    meson-python \
    maturin \
    setuptools-rust \
    scikit-build-core \
    pybind11 \
    nanobind

########################################
# wheel dir
########################################

export WHEEL_DIR=/wheelhouse

mkdir -p ${WHEEL_DIR}

########################################
# helper function: install from devpi
########################################

try_install_from_devpi() {

    pkg=$1

    if [[ -n "$DEVPI_URL" ]]; then

        if uv pip install \
            --extra-index-url "$DEVPI_URL" \
            --index-strategy unsafe-best-match \
            "$pkg"; then

            echo "Installed $pkg from DevPI"

            return 0
        fi
    fi

    return 1
}

########################################
# LAPACK
########################################
install_lapack() {

    if try_install_from_devpi lapack==3.12.1; then

        echo "Installed LAPACK from DevPI"

        return
    fi

    cd /root

    export LAPACK_VERSION=3.12.1

    git clone --recursive \
        https://github.com/Reference-LAPACK/lapack.git \
        -b v${LAPACK_VERSION}

    cd lapack

    cmake -B build -S .

    cmake --build build -j ${MAX_JOBS}

    cmake --install build
}

########################################
# NUMACTL
########################################
install_numactl() {

    cd /root

    export NUMACTL_VERSION=2.0.19

    git clone --recursive \
        https://github.com/numactl/numactl.git \
        -b v${NUMACTL_VERSION}

    cd numactl

    autoreconf -i

    ./configure

    make -j ${MAX_JOBS}

    make install
}

########################################
# OPENBLAS
########################################

install_openblas() {

    if try_install_from_devpi openblas; then

        echo "Installed OpenBLAS from DevPI"

        return
    fi

    echo "Falling back to source build"

    cd /root

    export OPENBLAS_VERSION=0.3.33

    curl -L \
    https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz \
    | tar xz

    mv OpenBLAS-${OPENBLAS_VERSION}/ OpenBLAS/

    cd OpenBLAS/

    make -j${MAX_JOBS} \
        TARGET=POWER9 \
        BINARY=64 \
        USE_OPENMP=1 \
        USE_THREAD=1 \
        NUM_THREADS=120 \
        DYNAMIC_ARCH=1 \
        INTERFACE64=0

    make install

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib
}
########################################
# PYTORCH FAMILY
########################################

install_torch_family() {

cd ${CURDIR}

TORCH_VERSION=2.11.0

if try_install_from_devpi torch==$TORCH_VERSION; then

    try_install_from_devpi torchvision

    try_install_from_devpi torchaudio

    return
fi

TEMP_BUILD_DIR=$(mktemp -d)

cd $TEMP_BUILD_DIR

export _GLIBCXX_USE_CXX11_ABI=1

git clone --recursive \
    https://github.com/pytorch/pytorch.git \
    -b v${TORCH_VERSION}

cd pytorch

pip install uv==0.5.9

uv pip install -r requirements.txt

python setup.py develop

pip install -U uv

PYTORCH_BUILD_VERSION=${TORCH_VERSION} \
PYTORCH_BUILD_NUMBER=1 \
uv build --wheel --out-dir ${WHEEL_DIR}

cd ${CURDIR}

rm -rf $TEMP_BUILD_DIR
}

########################################
# LLVMLITE
########################################

install_llvmlite() {

if try_install_from_devpi llvmlite==0.47.0; then
    return
fi

cd ${CURDIR}

LLVMLITE_VERSION=0.47.0

TEMP_BUILD_DIR=$(mktemp -d)

cd $TEMP_BUILD_DIR

git clone --recursive \
    https://github.com/numba/llvmlite.git \
    -b v${LLVMLITE_VERSION}

cd llvmlite

echo "setuptools<70.0.0" > build_constraints.txt

uv build \
    --wheel \
    --out-dir /llvmlitewheel \
    --build-constraint build_constraints.txt

cd /llvmlitewheel

auditwheel repair llvmlite*.whl

mv wheelhouse/llvmlite*.whl ${WHEEL_DIR}

cd ${CURDIR}

rm -rf $TEMP_BUILD_DIR
}

########################################
# PYARROW
########################################

install_pyarrow() {

if try_install_from_devpi pyarrow==23.0.1; then
    return
fi

TEMP_BUILD_DIR=$(mktemp -d)
cd $TEMP_BUILD_DIR

#PYARROW_VERSION=$(curl -s https://api.github.com/repos/apache/arrow/releases/latest | jq -r '.tag_name' | grep -Eo "[0-9.]+")
PYARROW_VERSION=24.0.0
git clone --depth 1 https://github.com/apache/arrow.git -b apache-arrow-${PYARROW_VERSION}

cd arrow/cpp
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=release \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DARROW_PYTHON=ON \
-DARROW_PARQUET=ON \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make install -j ${MAX_JOBS}

cd ../../python
export PYARROW_BUNDLE_ARROW_CPP=1
export PATH=/opt/vllm/bin:$PATH
export ARROW_HOME=/usr/local
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig
export CMAKE_ARGS="-DPython3_EXECUTABLE=/opt/vllm/bin/python"
pip wheel libcst==1.8.6 -w /tmp/libcst_wheel
uv pip install /tmp/libcst_wheel/libcst-*.whl

uv pip install -r requirements-wheel-build.txt \
    --extra-index-url "$DEVPI_URL" \
    --index-strategy unsafe-best-match \
    --no-build-isolation

#python setup.py build_ext \
#--build-type=release --bundle-arrow-cpp \
#bdist_wheel --dist-dir ${WHEEL_DIR}

pip install "cython>=3.1"

python -m build --wheel --outdir ${WHEEL_DIR} --no-isolation

cd "$REPO_ROOT"
rm -rf $TEMP_BUILD_DIR
}


########################################
# NUMBA
########################################

install_numba() {

if try_install_from_devpi numba==0.65.0; then
    return
fi

cd ${CURDIR}

NUMBA_VERSION=$(grep -Eo '^numba.+;' requirements/cpu.txt | grep -Eo '[0-9\.]+' | tail -1)

TEMP_BUILD_DIR=$(mktemp -d)

cd $TEMP_BUILD_DIR

git clone --recursive \
    https://github.com/numba/numba.git \
    -b ${NUMBA_VERSION}

cd numba

sed -i \
'/#include "internal\/pycore_atomic.h"/i\#include "dynamic_annotations.h"' \
numba/_dispatcher.cpp || true

uv build --wheel --out-dir ${WHEEL_DIR}

cd ${CURDIR}

rm -rf $TEMP_BUILD_DIR
}

########################################
# XGRAMMAR
########################################

install_xgrammar() {

if try_install_from_devpi xgrammar==0.2.0; then
    return
fi

cd ${CURDIR}

XGRAMMAR_VERSION=0.2.0

TEMP_BUILD_DIR=$(mktemp -d)

cd $TEMP_BUILD_DIR

export CFLAGS="-fno-lto -mcpu=power9"
export CXXFLAGS="-fno-lto -mcpu=power9"
export LDFLAGS="-fno-lto"

git clone --recursive \
    https://github.com/mlc-ai/xgrammar \
    -b v${XGRAMMAR_VERSION}

cd xgrammar

cp cmake/config.cmake .

uv build --wheel --out-dir ${WHEEL_DIR}

uv pip install ${WHEEL_DIR}/xgrammar*.whl

cd ${CURDIR}

rm -rf $TEMP_BUILD_DIR
}

install_opencv() {

if try_install_from_devpi opencv-python; then
    return
fi

cd ${CURDIR}

TEMP_BUILD_DIR=$(mktemp -d)

cd $TEMP_BUILD_DIR

export OPENCV_VERSION=92
export ENABLE_HEADLESS=1

git clone --recursive \
    https://github.com/opencv/opencv-python.git \
    -b ${OPENCV_VERSION}

cd opencv-python

if [[ ${OPENCV_VERSION} == "92" ]]; then
    sed -i \
    's/__ARCH_PWR10__/__ARCH_PWR10__)/' \
    opencv/modules/core/include/opencv2/core/vsx_utils.hpp
fi

uv build --wheel --out-dir ${WHEEL_DIR}

cd ${CURDIR}

rm -rf $TEMP_BUILD_DIR
}

uv pip install \
    opencv-python-headless==4.13.0.92 \
    --extra-index-url "$DEVPI_URL" \
    --index-strategy unsafe-best-match \
    --no-deps || true
uv pip install sentencepiece==0.2.1 --no-build-isolation



########################################
# RUN BUILDS
########################################
install_lapack

install_numactl

install_opencv

install_torch_family

install_llvmlite

install_pyarrow

install_numba

install_xgrammar

########################################
# install built wheels
########################################

if ls ${WHEEL_DIR}/*.whl >/dev/null 2>&1; then
    find ${WHEEL_DIR} -name "*.whl" -print0 | \
    xargs -0 uv pip install
else
    echo "No local wheels found in ${WHEEL_DIR}"
fi
########################################
# install remaining deps
########################################

cd ${CURDIR}

sed -i.bak -e 's/.*torch.*//g' \
    pyproject.toml requirements/*.txt
sed -i '/opencv-python/d' requirements/*.txt
sed -i '/opencv-python-headless/d' requirements/*.txt
sed -i '/numba/d' requirements/*.txt
sed -i '/llvmlite/d' requirements/*.txt
export PKG_CONFIG_PATH=$(find / -type d -name "pkgconfig" 2>/dev/null | tr '\n' ':')

uv pip install \
    -r requirements/common.txt \
    -r requirements/cpu.txt \
    -r requirements/build/cpu.txt \
    --index-strategy unsafe-best-match


 
