#!/usr/bin/env bash
set -euo pipefail

GDRCOPY_VERSION="${GDRCOPY_VERSION:-2.5.1}"
GDRCOPY_PKG_VER="${GDRCOPY_VERSION}-1"

# This code path is used for CUDA 12.9 because there is no prebuilt package
# for that CUDA version available yet
build_from_source() {
  local cuda_home=/usr/local/cuda
  local gdrcopy_home=/usr/local/gdrcopy

  if [ ! -e "/dev/gdrdrv" ]; then
    echo "Installing gdrcopy"
    wget "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz"
    tar xzf "v${GDRCOPY_VERSION}.tar.gz"; rm "v${GDRCOPY_VERSION}.tar.gz"
    pushd "gdrcopy-${GDRCOPY_VERSION}"
    make prefix="${gdrcopy_home}" CUDA="${cuda_home}" all install

    sudo ./insmod.sh
    popd
  else
    echo "Found /dev/gdrdrv. Skipping gdrcopy installation"
  fi
}

# Usage: install_gdrcopy.sh <GDRCOPY_OS_VERSION> <GDRCOPY_CUDA_VERSION> <uuarch>
# uuarch must be "x64" or "aarch64"
# Optional: set GDRCOPY_VERSION to override the libgdrapi package version (default: 2.5.1-1)
# Requires: curl, apt-get, root privileges
if [[ $(id -u) -ne 0 ]]; then
  echo "Must be run as root" >&2
  exit 1
fi
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <GDRCOPY_OS_VERSION> <GDRCOPY_CUDA_VERSION> <uuarch(x64|aarch64)>" >&2
  exit 1
fi

OS_VER="$1"
CUDA_VER="$2"
UUARCH_RAW="$3"

# Normalize/validate arch
case "${UUARCH_RAW,,}" in
  aarch64|arm64)
    URL_ARCH="aarch64"
    DEB_ARCH="arm64"
    ;;
  x64|x86_64|amd64)
    URL_ARCH="x64"
    DEB_ARCH="amd64"
    ;;
  *)
    echo "Unsupported uuarch: ${UUARCH_RAW}. Use 'x64' or 'aarch64'." >&2
    exit 1
    ;;
esac

# TODO (huydhn): Clean this up once there is prebuilt gdrcopy package for 12.9
# or when we move to CUDA 13.0
if [[ "${CUDA_VER}" == *12.9* ]]; then
  build_from_source
else
  OS_VER_LOWER="$(tr '[:upper:]' '[:lower:]' <<<"$OS_VER")"
  DEB_NAME="libgdrapi_${GDRCOPY_PKG_VER}_${DEB_ARCH}.${OS_VER}.deb"
  BASE_URL="https://developer.download.nvidia.com/compute/redist/gdrcopy"
  URL="${BASE_URL}/CUDA%20${CUDA_VER}/${OS_VER_LOWER}/${URL_ARCH}/${DEB_NAME}"

  echo "Downloading: ${URL}"
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "${TMPDIR}"' EXIT

  curl -fSL "${URL}" -o "${TMPDIR}/${DEB_NAME}"

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y "${TMPDIR}/${DEB_NAME}"
  apt-get clean
  rm -rf /var/lib/apt/lists/*

  echo "Installed ${DEB_NAME}"
fi
