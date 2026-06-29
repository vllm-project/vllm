#!/bin/bash
# apply_moriio_2pd_patches.sh — apply vLLM multi-node DP MoRIIO fixes at startup
# =============================================================================
# Thin wrapper around the REBASED, anchor-based Python patcher
# (apply_39276_rebased.py, vendored alongside this script). That patcher ports
# vLLM PR #39276 plus the follow-on multi-node DP fixes (notify-direction,
# LL split, DP-rank failsafe, hang-proofing) using string-anchor replacements
# instead of `patch -p1` hunks, so it survives base-image drift:
#   - idempotent: already-applied hunks are skipped
#   - anchors that don't match THIS image WARN + skip (don't abort)
#
# Why this replaced the old live-PR download: PR #39276 evolved to a 3-commit
# patch whose hunks no longer applied onto vLLM 0.23.0 (2/7 hunks failed, the
# moriio_engine timeout hunk never landed -> verification aborted). The vendored
# rebased patcher applies cleanly to 0.23.0 and needs no network at runtime.
#
# Override knobs:
#   REBASED_PATCHER        path to the python patcher (default: next to this file)
#   REBASED_PATCHER_ARGS   extra args, e.g. "--core-only" for nightlies whose
#                          MoRIIO connector already has the notify/remote_dp_rank
#                          logic natively (layering connector hunks there breaks it)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="${REBASED_PATCHER:-${SCRIPT_DIR}/apply_39276_rebased.py}"
PATCHER_ARGS="${REBASED_PATCHER_ARGS:-}"

if [ ! -f "${PATCHER}" ]; then
    echo "[moriio-patch] ERROR: rebased patcher not found: ${PATCHER}"
    exit 1
fi

# Locate the vLLM install dir (package dir, e.g. .../dist-packages/vllm).
VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
if [ -z "${VLLM_DIR}" ] || [ ! -d "${VLLM_DIR}" ]; then
    for _c in /usr/local/lib/python3.*/dist-packages/vllm; do
        [ -d "${_c}" ] && VLLM_DIR="${_c}" && break
    done
fi
if [ -z "${VLLM_DIR}" ] || [ ! -d "${VLLM_DIR}" ]; then
    echo "[moriio-patch] ERROR: cannot find vLLM install directory"
    exit 1
fi
echo "[moriio-patch] vLLM dir: ${VLLM_DIR}"

# Apply (word-splitting on PATCHER_ARGS is intentional for optional flags).
echo "[moriio-patch] applying rebased patcher: $(basename "${PATCHER}") ${PATCHER_ARGS}"
# shellcheck disable=SC2086
python3 "${PATCHER}" ${PATCHER_ARGS} "${VLLM_DIR}"

# Verify the multi-node DP fixes actually landed (markers the rebased patcher
# produces on a stock upstream vLLM such as 0.23.0).
echo "[moriio-patch] verifying patched markers..."
_ok=0
_total=0
_check() {
    _total=$((_total + 1))
    if [ -f "${VLLM_DIR}/$1" ] && grep -q "$2" "${VLLM_DIR}/$1" 2>/dev/null; then
        echo "  ok  $3"
        _ok=$((_ok + 1))
    else
        echo "  XX  $3 — marker '$2' not found in $1"
    fi
}
_check "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"    "data_parallel_size_local" "multi-node DP sizing"
_check "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py" "_req_kv_params"            "kv_transfer_params caching"
_check "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py" "_is_kv_master"            "child-node guard"

# Sanity: vLLM (and the MoRIIO connector) must still import after patching.
if ! python3 -c 'import vllm' >/dev/null 2>&1; then
    echo "[moriio-patch] ERROR: vLLM fails to import after patching"
    exit 1
fi

if [ "${_ok}" -ne "${_total}" ]; then
    echo "[moriio-patch] ERROR: verification ${_ok}/${_total} — required multi-node markers missing"
    exit 1
fi
echo "[moriio-patch] OK: ${_ok}/${_total} markers present; vLLM imports cleanly"
