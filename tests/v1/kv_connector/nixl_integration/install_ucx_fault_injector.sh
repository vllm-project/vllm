#!/bin/bash
set -e

# script to install ucx-fault-injector
# finds git repository root and installs to .ucx-fault-injector directory

GIT_ROOT=$(git rev-parse --show-toplevel)
UCX_INJECTOR_DIR="${GIT_ROOT}/.ucx-fault-injector"

echo "Installing ucx-fault-injector to ${UCX_INJECTOR_DIR}..."
mkdir -p "${UCX_INJECTOR_DIR}"
cd "${UCX_INJECTOR_DIR}"

# download and extract the latest release
TARBALL="ucx-fault-injector-linux-amd64.tar.gz"
curl -LO "https://github.com/wseaton/ucx-fault-injector/releases/latest/download/${TARBALL}"
tar xzf "${TARBALL}" --strip-components=1
rm "${TARBALL}"

# verify installation
if [[ ! -f "${UCX_INJECTOR_DIR}/libucx_fault_injector.so" ]]; then
  echo "ERROR: ucx-fault-injector library not found at ${UCX_INJECTOR_DIR}/libucx_fault_injector.so"
  exit 1
fi

echo "ucx-fault-injector installed successfully"
echo "Library: ${UCX_INJECTOR_DIR}/libucx_fault_injector.so"
echo "Client: ${UCX_INJECTOR_DIR}/ucx-fault-client"
