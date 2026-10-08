#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Install MoRI proxy dependencies and the userspace NIC libraries MoRI dlopens.
# MoRI auto-detects the NIC at runtime (MORI_DEVICE_NIC env var override).
#
# Required env: NIC_BACKEND (none, ainic, bnxt or all), AINIC_VERSION,
# UBUNTU_CODENAME.

set -euo pipefail

install_ainic() {
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg apt-transport-https
    rm -rf /var/lib/apt/lists/*
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/amdainic.list
    apt-get update && apt-get install -y --no-install-recommends \
        libionic-dev \
        ionic-common
    rm -rf /var/lib/apt/lists/*
}

# NOTE: requires FW 235.2.86.0 and kernel drivers on the host:
#   bnxt-en-dkms=1.10.3.235.2.86.0 bnxt-re-dkms=235.2.86.0 (from packages.broadcom.com PPA)
install_bnxt() {
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public \
        -o /etc/apt/keyrings/broadcom-nic.asc
    chmod a+r /etc/apt/keyrings/broadcom-nic.asc
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/broadcom-nic.asc] https://packages.broadcom.com/artifactory/ethernet-nic-debian-public jammy main" \
        > /etc/apt/sources.list.d/broadcom-nic.list
    apt-get update && apt-get install -y --no-install-recommends \
        bnxt-rocelib=235.2.86.0
    cp -a /usr/local/lib/x86_64-linux-gnu/libbnxt_re* /usr/local/lib/
    ldconfig
    rm -rf /var/lib/apt/lists/*
}

echo "[MORI] Install MoRI proxy deps"
pip install --quiet --ignore-installed blinker
pip install --quiet quart msgpack aiohttp pyzmq
echo "[MORI] NIC_BACKEND=${NIC_BACKEND}"

# Only vendor packages are installed here for dlopen; no compile-time flags needed.
case "${NIC_BACKEND}" in
    none) ;;
    all) install_ainic; install_bnxt ;;
    ainic) install_ainic ;;
    bnxt) install_bnxt ;;
    *) echo "ERROR: unknown NIC_BACKEND=${NIC_BACKEND}. Use one of: none, ainic, bnxt, all"; exit 2 ;;
esac
