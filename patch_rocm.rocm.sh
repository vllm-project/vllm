#!/bin/bash
set -e

if [ -z "$ROCM_PATH" ]; then
    echo "Could not determine ROCm installation path by ROCM_PATH. Abort HIP patching"
    exit 1
fi

export __HIP_FILE_TO_PATCH="$ROCM_PATH/include/hip/amd_detail/amd_hip_bf16.h"
export __HIP_PATCH_FILE="./rocm_patch/rocm__amd_bf16.patch"

if [ ! -f "$__HIP_FILE_TO_PATCH" ]; then
    echo "Could not find the file to be patched in $__HIP_FILE_TO_PATCH. Abort HIP patching"
    exit 2
fi

echo "File to be patched: $__HIP_FILE_TO_PATCH"

if ! patch -R -p0 -s -f --dry-run $__HIP_FILE_TO_PATCH $__HIP_PATCH_FILE; then
    echo "Applying patch to ${__HIP_FILE_TO_PATCH}"
    patch -p0 $__HIP_FILE_TO_PATCH $__HIP_PATCH_FILE
    echo "Successfully patched ${__HIP_FILE_TO_PATCH}"
else
    echo "${__HIP_FILE_TO_PATCH} has been patched before"
fi

exit 0