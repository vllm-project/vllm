#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euxo pipefail

# Surface the isolated snapshot-runtime install cost in the CI build log. The
# trap captures the status first: the date substitution below would reset it.
_snapshot_install_started=$(date +%s)
code=0
trap 'code=$?; echo "[TIMING] snapshot runtime install: $(( $(date +%s) - _snapshot_install_started )) s (exit $code)"' EXIT

readonly CRIU_VERSION="4.2.1"
readonly CRIU_SOURCE_SHA256="feffdf4638125ebb12d2434754f80a1d7bbba85a3e6bee98c216f88fb99a5d96"
readonly DYNAMO_CHECKPOINT_HELPER_COMMIT="fdf25efad60f696a73393caf341c527a93d84190"
readonly DYNAMO_CHECKPOINT_HELPER_SHA256="7b8ea21baecb1da729202c32e0fecbbbafb789d65e5c1c0649ea253ad09b8f82"
readonly DYNAMO_LICENSE_SHA256="aeeca3d74a13b91d17ab9211c407b8d94a77bf8d9e5750c467818d3066b8adcf"

# Keep the compiler and NL headers required by runtime JIT and RDMA.
snapshot_shared_build_deps=(
    build-essential
    libnl-3-dev
    libnl-route-3-dev
)
snapshot_build_deps=(
    pkg-config
    libbsd-dev
    libcap-dev
    libnet1-dev
    libprotobuf-dev
    libprotobuf-c-dev
    protobuf-c-compiler
    protobuf-compiler
    python3-protobuf
    libgnutls28-dev
    libnftables-dev
    uuid-dev
)
snapshot_runtime_deps=(
    libbsd0
    libcap2
    libnet1
    libnl-3-200
    libnl-route-3-200
    libprotobuf-c1
    libgnutls30t64
    libnftables1
    iproute2
    iptables
    procps
    uuid-runtime
    util-linux
)

apt-get update -y
apt-get install -y --no-install-recommends \
    "${snapshot_shared_build_deps[@]}" \
    "${snapshot_build_deps[@]}" \
    "${snapshot_runtime_deps[@]}"

snapshot_build_dir="$(mktemp -d)"
curl -fsSL \
    "https://github.com/checkpoint-restore/criu/archive/refs/tags/v${CRIU_VERSION}.tar.gz" \
    -o "${snapshot_build_dir}/criu.tar.gz"
echo "${CRIU_SOURCE_SHA256}  ${snapshot_build_dir}/criu.tar.gz" | sha256sum -c -
mkdir "${snapshot_build_dir}/criu"
tar -xzf "${snapshot_build_dir}/criu.tar.gz" \
    --strip-components=1 -C "${snapshot_build_dir}/criu"
make -C "${snapshot_build_dir}/criu" -j"${MAX_JOBS:-$(nproc)}" criu cuda_plugin
install -D -m 0755 "${snapshot_build_dir}/criu/criu/criu" \
    /usr/local/sbin/criu
install -D -m 0755 \
    "${snapshot_build_dir}/criu/plugins/cuda/cuda_plugin.so" \
    /usr/local/lib/criu/cuda_plugin.so

helper_url="https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_CHECKPOINT_HELPER_COMMIT}/deploy/snapshot/cmd/cuda-checkpoint-helper/main.c"
curl -fsSL "${helper_url}" -o "${snapshot_build_dir}/cuda-checkpoint.c"
echo "${DYNAMO_CHECKPOINT_HELPER_SHA256}  ${snapshot_build_dir}/cuda-checkpoint.c" \
    | sha256sum -c -
gcc -O2 -Wall -Wextra \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64/stubs \
    "${snapshot_build_dir}/cuda-checkpoint.c" -lcuda \
    -o /usr/local/sbin/cuda-checkpoint

license_url="https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_CHECKPOINT_HELPER_COMMIT}/LICENSE"
curl -fsSL "${license_url}" -o "${snapshot_build_dir}/DYNAMO-LICENSE"
echo "${DYNAMO_LICENSE_SHA256}  ${snapshot_build_dir}/DYNAMO-LICENSE" \
    | sha256sum -c -

install -D -m 0644 "${snapshot_build_dir}/criu.tar.gz" \
    /usr/share/doc/criu/source/criu.tar.gz
install -D -m 0644 "${snapshot_build_dir}/criu/COPYING" \
    /usr/share/doc/criu/COPYING
install -D -m 0644 "${snapshot_build_dir}/cuda-checkpoint.c" \
    /usr/share/doc/cuda-checkpoint-helper/main.c
install -D -m 0644 "${snapshot_build_dir}/DYNAMO-LICENSE" \
    /usr/share/doc/cuda-checkpoint-helper/LICENSE

criu --version
ln -s /usr/local/cuda/lib64/stubs/libcuda.so \
    "${snapshot_build_dir}/libcuda.so.1"
LD_LIBRARY_PATH="${snapshot_build_dir}" \
    cuda-checkpoint --help | grep -q -- "--action"
test -x /usr/local/lib/criu/cuda_plugin.so

apt-get purge -y --auto-remove "${snapshot_build_deps[@]}"
rm -rf "${snapshot_build_dir}"
rm -rf /var/lib/apt/lists/*
