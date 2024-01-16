#!/bin/bash
set -e

XFORMERS_VERSION="0.0.23"

export XFORMERS_INSTALLED_VERSION=$(python -c 'import xformers; print(xformers.__version__)')

if [ "$XFORMERS_INSTALLED_VERSION" != "$XFORMERS_VERSION" ]; then
    echo "ERROR: xformers version must be ${XFORMERS_VERSION}. ${XFORMERS_INSTALLED_VERSION} is installed"
    exit 1
fi

export XFORMERS_FMHA_FLASH_PATH=$(python -c 'from xformers import ops as xops; print(xops.fmha.flash.__file__)')
export XFORMERS_FMHA_COMMON_PATH=$(python -c 'from xformers import ops as xops; print(xops.fmha.common.__file__)')

echo "XFORMERS_FMHA_FLASH_PATH = ${XFORMERS_FMHA_FLASH_PATH}"
echo "XFORMERS_FMHA_COMMON_PATH = ${XFORMERS_FMHA_COMMON_PATH}"

if ! patch -R -p0 -s -f --dry-run $XFORMERS_FMHA_FLASH_PATH "./rocm_patch/flashpy_xformers-${XFORMERS_VERSION}.rocm.patch"; then
    echo "Applying patch to ${XFORMERS_FMHA_FLASH_PATH}"
    patch -p0 $XFORMERS_FMHA_FLASH_PATH "./rocm_patch/flashpy_xformers-${XFORMERS_VERSION}.rocm.patch"
    echo "Successfully patch ${XFORMERS_FMHA_FLASH_PATH}"
else
    echo "${XFORMERS_FMHA_FLASH_PATH} was patched before"
fi

if ! patch -R -p0 -s -f --dry-run $XFORMERS_FMHA_COMMON_PATH "./rocm_patch/commonpy_xformers-${XFORMERS_VERSION}.rocm.patch"; then
    echo "Applying patch to ${XFORMERS_FMHA_COMMON_PATH}"
    patch -p0 $XFORMERS_FMHA_COMMON_PATH "./rocm_patch/commonpy_xformers-${XFORMERS_VERSION}.rocm.patch"
    echo "Successfully patch ${XFORMERS_FMHA_COMMON_PATH}"
else
    echo "${XFORMERS_FMHA_COMMON_PATH} was patched before"
fi
