#!/usr/bin/env bash
# Install the AWS EFA userspace (libfabric, aws-ofi-nccl, hwloc) into the
# release image and check that NIXL and NCCL resolve to it. Run by the
# vllm-openai-base stage of docker/Dockerfile when INSTALL_KV_CONNECTORS=true.
#
# Env: EFA_INSTALLER_VERSION, EFA_INSTALLER_SHA256
set -eux

EFA_INSTALLER_ARCHIVE="/tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
curl -fsSL --retry 3 --retry-all-errors \
    "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
    -o "${EFA_INSTALLER_ARCHIVE}"
echo "${EFA_INSTALLER_SHA256}  ${EFA_INSTALLER_ARCHIVE}" | sha256sum -c -
tar -xzf "${EFA_INSTALLER_ARCHIVE}" -C /tmp
# The installer apt-installs dependencies; earlier layers purged the lists.
apt-get update -y
# Minimal install: libfabric, aws-ofi-nccl and hwloc only. --build-ngc
# implies --skip-kmod --skip-limit-conf --no-verify.
(cd /tmp/aws-efa-installer &&
    ./efa_installer.sh --skip-mpi --skip-rdma-core --build-ngc)
test -x /opt/amazon/efa/bin/fi_info
/opt/amazon/efa/bin/fi_info --version
# `-l` lists providers without EFA hardware, so this works in a build.
/opt/amazon/efa/bin/fi_info -l | grep -q '^efa:'
EFA_NIXL_PLUGIN=$(find /usr/local/lib -name libplugin_LIBFABRIC.so -print -quit)
test -n "${EFA_NIXL_PLUGIN}"
ldd "${EFA_NIXL_PLUGIN}" | grep -q 'libfabric\.so\.1 => /opt/amazon/efa/lib/'
# Plugins that NCCL_NET_PLUGIN=ofi and NCCL_TUNER_PLUGIN=ofi resolve to.
ldconfig -p | grep -q '/opt/amazon/ofi-nccl/lib/libnccl-net-ofi\.so$'
ldconfig -p | grep -q '/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi\.so$'
apt-get clean
rm -rf "${EFA_INSTALLER_ARCHIVE}" /tmp/aws-efa-installer /var/lib/apt/lists/*
