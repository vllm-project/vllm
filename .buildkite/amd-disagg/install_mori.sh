#!/usr/bin/env bash
# Clone MoRI into /app/mori and install it before the servers start.
#
# WideEP at DP16 (xP=2/yD=2) needs a newer MoRI than the image carries.
# vllm/vllm-openai-rocm:nightly installs amd-mori from a wheel built in the
# throwaway build_mori stage of docker/Dockerfile.rocm_base, so the released
# image has the library but no source tree -- hence the clone.
#
# This runs entirely unprivileged and entirely inside the container. It needs
# three things from run_xPyD_disagg.slurm, which sets them only for the runs
# that rebuild:
#   /app/mori          a writable tmpfs, since /app is root-owned in the image
#   CCACHE_DIR         writable; the image pins ccache to /root/.cache/ccache
#   PYTHONUSERBASE     writable prefix for the install, since dist-packages is
#                      root-owned. It precedes dist-packages on sys.path, so
#                      this build wins over the image's amd-mori.
# Running the container as root instead is not an option: /data is NFS and
# squashes container root to an anonymous uid that cannot write LOG_PATH.
#
# Nothing is written to the shared mount, so there is no cross-node state and
# nothing to serialise -- each node just builds its own copy in parallel.
set -euo pipefail

log() { echo "[install-mori] $*"; }
die() { echo "[install-mori] ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# MORI_GPU_ARCHS lives in cluster.sh (pinned to gfx950 for this cluster) and is
# not forwarded as a docker -e, so pick it up the same way vllm_disagg.sh does.
CLUSTER_ENV="${CLUSTER_ENV:-${SCRIPT_DIR}/cluster.sh}"
# shellcheck disable=SC1090
[[ -f "${CLUSTER_ENV}" ]] && source "${CLUSTER_ENV}"

WIDE_EP_MODE="${WIDE_EP_MODE:-0}"
xP="${xP:-1}"
yD="${yD:-1}"

MORI_SRC="${MORI_SRC:-/app/mori}"
MORI_REPO="${MORI_REPO:-https://github.com/ROCm/mori.git}"
MORI_COMMIT="${MORI_COMMIT:-c22c33a7200c766e52aa7c285fd9e4e65901e3bd}"
MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx950}"
MORI_REINSTALL_FORCE="${MORI_REINSTALL_FORCE:-0}"
MORI_SKIP_REINSTALL="${MORI_SKIP_REINSTALL:-0}"

# ------------------------------------------------------------------------ gate
# Narrow by design: only the WideEP DP16 fanout needs the override, and
# silently swapping MoRI under the TP jobs would make their results
# incomparable to the nightly baseline. MORI_REINSTALL_FORCE=1 opts a different
# topology in; MORI_SKIP_REINSTALL=1 opts DP16 back out.
if [[ "${MORI_SKIP_REINSTALL}" == "1" ]]; then
    log "skip: MORI_SKIP_REINSTALL=1; keeping the image's MoRI"
    exit 0
fi

if [[ "${MORI_REINSTALL_FORCE}" != "1" ]]; then
    if [[ "${WIDE_EP_MODE}" != "1" || "${xP}" != "2" || "${yD}" != "2" ]]; then
        log "skip (WIDE_EP_MODE=${WIDE_EP_MODE} xP=${xP} yD=${yD}); keeping the image's MoRI"
        exit 0
    fi
fi

# --------------------------------------------------------------- preconditions
mkdir -p "${MORI_SRC}" 2>/dev/null || true
[[ -w "${MORI_SRC}" ]] || die "${MORI_SRC} is not writable by uid=$(id -u). run_xPyD_disagg.slurm must mount a tmpfs there (--tmpfs ${MORI_SRC}:rw,mode=1777); set MORI_SKIP_REINSTALL=1 to keep the image's MoRI."

[[ -n "${PYTHONUSERBASE:-}" ]] || die "PYTHONUSERBASE is unset; the install would target root-owned dist-packages. run_xPyD_disagg.slurm must export it (e.g. /tmp/mori_pyuser)."

# The image points ccache at /root/.cache/ccache, which an unprivileged build
# cannot create -- every compile then fails on "Failed to create directory".
export CCACHE_DIR="${CCACHE_DIR:-/tmp/ccache}"
mkdir -p "${CCACHE_DIR}" "${PYTHONUSERBASE}"

# The tmpfs mount point belongs to root even though its mode lets us write, so
# git refuses the checkout with "detected dubious ownership". Waive the check
# via the environment rather than a global config so it also reaches the git
# processes spawned for the submodules.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="safe.directory"
export GIT_CONFIG_VALUE_0="*"

log "installing MoRI ${MORI_COMMIT:0:9} at ${MORI_SRC} (WIDE_EP_MODE=1 xP=${xP} yD=${yD}, uid=$(id -u))"

# ----------------------------------------------------------------------- clone
# git clone insists on an empty target, and the tmpfs may carry leftovers if
# this ever runs twice in one container.
if [[ -n "$(ls -A "${MORI_SRC}" 2>/dev/null)" ]]; then
    log "clearing existing contents of ${MORI_SRC}"
    rm -rf -- "${MORI_SRC:?}"/* "${MORI_SRC:?}"/.[!.]* 2>/dev/null || true
fi

log "cloning ${MORI_REPO}"
git clone --quiet "${MORI_REPO}" "${MORI_SRC}" || die "failed to clone ${MORI_REPO} into ${MORI_SRC}"
cd "${MORI_SRC}"
git checkout --quiet --force "${MORI_COMMIT}" \
    || die "could not check out ${MORI_COMMIT} (not in ${MORI_REPO}?)"
git submodule update --quiet --init --recursive || die "submodule init failed"
log "checked out $(git rev-parse --short HEAD)"

# ----------------------------------------------------------------------- build
log "installing build requirements"
python3 -m pip install --quiet --user -r requirements-build.txt \
    || die "pip install -r requirements-build.txt failed"

log "building MoRI for MORI_GPU_ARCHS=${MORI_GPU_ARCHS} (a few minutes)"
# Install via pip rather than `setup.py install`: the latter produces an egg
# that leaves libmori_pybinds.so out of the installed tree, so `import mori`
# succeeds (it resolves lazily) but `mori.ops` dies on a missing .so at model
# load. pip builds a wheel, which is also how Dockerfile.rocm_base does it.
# --no-build-isolation reuses the requirements-build.txt install above.
MORI_GPU_ARCHS="${MORI_GPU_ARCHS}" \
    python3 -m pip install --user --no-build-isolation . \
    || die "pip install of ${MORI_SRC} failed"

# mori reloads whatever kernels it finds in its cache without checking which
# build produced them, so kernels compiled by the image's MoRI would be picked
# up by this one and fail at EP init with "device kernel image is invalid".
# Purge before precompiling, or the warm cache below gets thrown away.
_jit="${MORI_JIT_CACHE_DIR:-${HOME:-/workspace}/.mori/jit}"
if [[ -d "${_jit}" ]]; then
    log "purging mori jit cache ${_jit} (library changed; forcing recompile)"
    rm -rf "${_jit}" 2>/dev/null || true
fi

# ---------------------------------------------------------------------- verify
# mori's top-level __getattr__ imports submodules lazily, so a bare
# `import mori` proves almost nothing. Import mori.ops to force the native
# extension to load, with the two env vars this cluster needs -- MORI_PRECOMPILE
# also warms MORI_JIT_CACHE_DIR here instead of inside every server rank.
log "precompiling and verifying (MORI_ENABLE_HOST_PROXY=1 MORI_PRECOMPILE=1)"
_got="$(MORI_ENABLE_HOST_PROXY=1 MORI_PRECOMPILE=1 python3 -c '
import mori, mori.ops
mori.ops.EpDispatchCombineOp
print(getattr(mori, "__version__", "unknown"), mori.__file__)
' 2>&1)" || die "MoRI failed to import/precompile after install:
${_got}"

# The image's amd-mori is still in root-owned dist-packages and cannot be
# uninstalled from here, so confirm the user prefix actually shadows it rather
# than trusting sys.path ordering.
case "${_got}" in
    *"${PYTHONUSERBASE}"*) log "active MoRI: ${_got##*$'\n'}" ;;
    *) die "installed MoRI is being shadowed by the image's copy (import resolved to: ${_got})" ;;
esac

# ------------------------------------------------- paired vLLM overlay check
# This MoRI pin requires vLLM to hand combine() the rank's own routing rather
# than dispatch()'s out_idx, so run_xPyD_disagg.slurm bind-mounts a patched
# prepare_finalize/mori.py over the image's copy. That mount is path-specific:
# if it misses, docker silently creates the path and MoRI's guard only fires
# once the model is loaded, minutes into the run. Check it here instead.
_pf=""
for _d in /usr/local/lib/python3*/dist-packages /usr/local/lib/python3*/site-packages \
          /usr/lib/python3*/dist-packages /usr/lib/python3*/site-packages; do
    _cand="${_d}/vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py"
    if [[ -f "${_cand}" ]]; then _pf="${_cand}"; break; fi
done
if [[ -z "${_pf}" ]]; then
    log "WARNING: could not find vLLM's prepare_finalize/mori.py; skipping the combine() fix check"
elif grep -q '_dispatch_topk_ids' "${_pf}"; then
    log "vLLM combine() routing fix active at ${_pf}"
else
    die "vLLM at ${_pf} still passes dispatch()'s out_idx to combine(), which this MoRI pin rejects at model load (ROCm/mori#475). The overlay mount did not take effect -- check VLLM_DIST_PACKAGES in run_xPyD_disagg.slurm."
fi
