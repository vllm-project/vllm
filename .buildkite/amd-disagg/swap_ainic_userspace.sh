#!/usr/bin/env bash
# Swap the AINIC/ionic *userspace* RoCE provider inside a vllm-openai-rocm
# container to a different repo.radeon.com channel.
#
# Only libionic1 / libionic-dev / ionic-common exist in this image. The kernel
# modules (ionic-dkms and required libraries) and the NIC firmware live on the host and are
# NOT touched here.
#
#   ./swap_ainic_userspace.sh --list
#   ./swap_ainic_userspace.sh 1.117.5-a-56
#   ./swap_ainic_userspace.sh 1.117.5-a-56 --dry-run
#
set -euo pipefail

REPO=https://repo.radeon.com/amdainic/pensando/ubuntu
PKGS=(libionic-dev libionic1 ionic-common)
DRY=0

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "${1:-}" == "--list" ]] && {
    curl -fsS "$REPO/" | grep -oP '(?<=href=")[0-9][^"]*(?=/")' | sort -V
    exit 0
}

CHANNEL="${1:-}"
[[ -n "$CHANNEL" ]] || die "usage: $0 <channel|--list> [--dry-run]"
[[ "${2:-}" == "--dry-run" ]] && DRY=1

[[ $EUID -eq 0 ]] || die "must run as root inside the container"

. /etc/os-release
CODENAME="${VERSION_CODENAME:?cannot determine ubuntu codename}"

echo "==> current state"
dpkg-query -W -f='    ${Package} ${Version}\n' "${PKGS[@]}" 2>/dev/null || echo "    (none installed)"

# Resolve exact versions from the target channel rather than hardcoding them:
# the channel name and the package versions do not track each other.
echo "==> resolving ${CHANNEL} (${CODENAME})"
INDEX=$(curl -fsS "$REPO/$CHANNEL/dists/$CODENAME/main/binary-amd64/Packages") \
    || die "no such channel/codename: $CHANNEL/$CODENAME"

declare -a PINS=()
for p in "${PKGS[@]}"; do
    v=$(awk -v P="$p" '$1=="Package:"&&$2==P{f=1;next} f&&$1=="Version:"{print $2;exit}' <<<"$INDEX")
    [[ -n "$v" ]] || die "$p not present in channel $CHANNEL"
    echo "    $p=$v"
    PINS+=("$p=$v")
done

if [[ $DRY -eq 1 ]]; then echo "==> dry run, stopping"; exit 0; fi

# Purging libionic1 removes /etc/libibverbs.d/ionic.driver, so there is no
# usable RoCE provider between here and the install below. Do not run this
# while a job is holding QPs on the device.
echo "==> purging existing"
apt-mark unhold "${PKGS[@]}" 2>/dev/null || true
apt-get purge -y "${PKGS[@]}" 2>/dev/null || true

echo "==> pointing apt at ${CHANNEL}"
install -d -m 0755 /etc/apt/keyrings
[[ -f /etc/apt/keyrings/amdainic.gpg ]] || \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] $REPO/$CHANNEL $CODENAME main" \
    > /etc/apt/sources.list.d/amdainic.list

# Update only this list. libc6/libibverbs1 are already installed, so the
# other lists (purged at image build time) are not needed.
APT_ONLY=(-o Dir::Etc::sourcelist=/etc/apt/sources.list.d/amdainic.list
          -o Dir::Etc::sourceparts=/dev/null)
apt-get "${APT_ONLY[@]}" update

echo "==> installing"
apt-get install -y --no-install-recommends --allow-downgrades "${PINS[@]}"

# Without the hold, any later apt upgrade silently walks this back forward.
apt-mark hold "${PKGS[@]}"
ldconfig

echo "==> verify"
dpkg-query -W -f='    ${Package} ${Version}\n' "${PKGS[@]}"
ls -l /usr/lib/x86_64-linux-gnu/libionic.so.1 /etc/libibverbs.d/ionic.driver
command -v ibv_devinfo >/dev/null && ibv_devinfo 2>&1 | head -20 || true

