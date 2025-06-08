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
if ! ls *.deb 1> /dev/null 2>&1; then
    apt install devscripts -y
    CUDA=${CUDA_HOME:-/usr/local/cuda} ./build-deb-packages.sh
fi
sudo dpkg -i *.deb
