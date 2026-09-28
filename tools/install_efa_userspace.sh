#!/usr/bin/env bash
# Install the AWS EFA userspace (libfabric, rdma-core libraries/providers and
# aws-ofi-nccl) under /opt/amazon without touching the system rdma-core.
#
# aws-efa-installer replaces the distro rdma-core with its own build, whose
# providers use a different ABI suffix (rdmav64amzn0 instead of rdmav34). That
# breaks IB/RoCE users of the system libibverbs, so instead the packages are
# only unpacked here and enabled per process by efa-entrypoint.sh through
# LD_LIBRARY_PATH. The EFA kernel driver and gdrdrv come from the host.
set -euo pipefail

EFA_INSTALLER_VERSION="${1:-latest}"
arch="$(uname -m)"
. /etc/os-release
debs="aws-efa-installer/DEBS/UBUNTU${VERSION_ID//./}/${arch}"

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT
cd "${workdir}"
curl --retry 3 --retry-delay 2 -fsSL \
    "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" | tar xz

for p in "${debs}"/rdma-core/libibverbs1_*.deb \
         "${debs}"/rdma-core/ibverbs-providers_*.deb \
         "${debs}"/rdma-core/librdmacm1_*.deb \
         "${debs}"/libfabric1-aws_*.deb \
         "${debs}"/libfabric-aws-bin_*.deb \
         "${debs}"/libnccl-ofi_*.deb; do
    dpkg-deb -x "$p" stage
done

mkdir -p /opt/amazon
cp -a stage/opt/amazon/. /opt/amazon/

# Every rdma-core library must come from the same build as libibverbs: e.g.
# the system libmlx5 needs IBVERBS_PRIVATE_34 and fails to load against the
# AWS libibverbs.
libdir="stage/usr/lib/${arch}-linux-gnu"
cp -a "${libdir}"/lib*.so.1* /opt/amazon/efa/lib/

# Providers are either real files or symlinks to ../lib<name>.so.1; re-point
# the symlinks at the copies that now sit next to them. libibverbs dlopens
# providers by bare name first, so LD_LIBRARY_PATH finds them here.
for f in "${libdir}"/libibverbs/*-rdmav*.so; do
    if [ -L "$f" ]; then
        ln -sf "$(basename "$(readlink "$f")")" "/opt/amazon/efa/lib/$(basename "$f")"
    else
        cp -a "$f" /opt/amazon/efa/lib/
    fi
done

# Driver config files are shared with the system rdma-core; only add missing ones.
mkdir -p /etc/libibverbs.d
for f in stage/etc/libibverbs.d/*.driver; do
    [ -e "/etc/libibverbs.d/$(basename "$f")" ] || cp "$f" /etc/libibverbs.d/
done
