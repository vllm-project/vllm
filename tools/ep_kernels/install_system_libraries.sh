set -ex

# prepare workspace directory
WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)/ep_kernels_workspace
fi

if [ ! -d "$WORKSPACE" ]; then
    mkdir -p $WORKSPACE
fi

# build and install gdrcopy system packages
pushd $WORKSPACE
cd gdrcopy_src/packages
apt install devscripts -y
CUDA=${CUDA_HOME:-/usr/local/cuda} ./build-deb-packages.sh
dpkg -i *.deb
