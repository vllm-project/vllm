#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared helper for rewriting a wheel's platform tag from the generic
# ``linux_<arch>`` to the correct ``manylinux_<major>_<minor>_<arch>``.
# After sourcing, call ``apply_manylinux_tag <wheel>`` on each wheel
# that still carries the generic tag; the renamed path is printed on
# stdout (logs go to stderr).
#
# Why a pinned Docker container instead of using whatever Python
# happens to be on the agent:
#   - vLLM's release agents are heterogeneous -- they don't agree on
#     a Python minor version, and we can't rely on a particular
#     ``auditwheel`` being installed.
#   - ``detect-manylinux-tag.py`` reads ``auditwheel.wheel_abi`` and
#     ``Policy.sym_policy``, which are *internal* APIs without a
#     stability promise. Pinning both Python and auditwheel makes the
#     detected tag a function of the inputs alone, and shifts version
#     bumps from "implicit drift" to "deliberate, retested change".
#   - Other release scripts (``generate-and-upload-nightly-index.sh``,
#     ``upload-rocm-wheels.sh``) already use the python:3-slim image
#     when the agent's interpreter is too old; this is the same idea
#     made stricter.
#
# To keep the per-wheel cost down (the ROCm upload retags ~10 wheels
# each run), we install auditwheel into a long-lived helper container
# once on source, then ``docker exec`` into it for each call.
#
# Trap behaviour:
# - Sourcing installs an EXIT trap that calls ``manylinux_cleanup`` to
#   tear down the helper container. Any EXIT trap that was already in
#   place when this file was sourced is captured and run AFTER our
#   cleanup, so we don't silently clobber it.
# - If a caller sets a new EXIT trap *after* sourcing, that trap will
#   replace ours; in that case the caller should call
#   ``manylinux_cleanup`` from their own handler.

if [[ -n "${_MANYLINUX_LIB_SOURCED:-}" ]]; then
    return 0
fi
_MANYLINUX_LIB_SOURCED=1

# Pin both sides. Bump these deliberately and re-run a representative
# wheel from each build target through the detection.
_MANYLINUX_PYTHON_IMAGE="python:3.12-slim"
_MANYLINUX_AUDITWHEEL_VERSION="6.6.0"

# Resolve our own directory (and the sibling detect script) using the
# canonical, symlink-resolved path. The container mounts cwd at the
# same absolute path on both sides, so all paths we hand to it -- the
# script, the wheel -- must canonicalise to a location under cwd.
_MANYLINUX_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_MANYLINUX_DETECT_SCRIPT="$(cd "${_MANYLINUX_LIB_DIR}/.." && pwd -P)/detect-manylinux-tag.py"
_MANYLINUX_CWD="$(pwd -P)"

docker pull --quiet "$_MANYLINUX_PYTHON_IMAGE" >/dev/null

# Spin up a long-lived helper container so we install auditwheel once
# and then ``docker exec`` into it for each wheel.
#
# The container runs as root so ``pip install`` can write into the
# system site-packages; individual ``docker exec`` calls below pin
# themselves to the host UID so any file rename happens with host
# ownership, not root.
_MANYLINUX_CONTAINER="$(docker run -d --rm \
    -v "$_MANYLINUX_CWD:$_MANYLINUX_CWD" \
    -w "$_MANYLINUX_CWD" \
    "$_MANYLINUX_PYTHON_IMAGE" \
    sleep infinity)"
docker exec "$_MANYLINUX_CONTAINER" \
    pip install --quiet --disable-pip-version-check \
    --root-user-action=ignore \
    "auditwheel==${_MANYLINUX_AUDITWHEEL_VERSION}"

# Public cleanup -- safe to call multiple times.
manylinux_cleanup() {
    if [[ -n "${_MANYLINUX_CONTAINER:-}" ]]; then
        docker rm -f "$_MANYLINUX_CONTAINER" >/dev/null 2>&1 || true
        _MANYLINUX_CONTAINER=""
    fi
}

# Capture any EXIT trap that was already in place so we can chain to
# it rather than overwrite it. ``trap -p EXIT`` prints the handler in
# eval-able form (``trap -- 'CMD' EXIT``) or nothing if unset; we
# strip the wrapper to recover ``CMD``. Handles the common case --
# CMDs without embedded single quotes -- and degrades gracefully (we
# still run our own cleanup) for the pathological case.
_manylinux_prev_exit_trap_cmd=""
_manylinux_existing_exit_trap="$(trap -p EXIT)"
if [[ -n "$_manylinux_existing_exit_trap" ]]; then
    _tmp="${_manylinux_existing_exit_trap#trap -- \'}"
    _manylinux_prev_exit_trap_cmd="${_tmp%\' EXIT}"
    unset _tmp
fi
unset _manylinux_existing_exit_trap

_manylinux_run_exit_chain() {
    manylinux_cleanup
    if [[ -n "$_manylinux_prev_exit_trap_cmd" ]]; then
        eval "$_manylinux_prev_exit_trap_cmd"
    fi
}
trap _manylinux_run_exit_chain EXIT

# Detect the manylinux platform tag for a single wheel and rename it
# in place, printing the renamed wheel path on stdout. Returns
# non-zero on failure (which under ``set -e`` propagates to caller).
#
# The wheel must be reachable via a path under the host cwd so it's
# visible inside the helper container; in CI the wheels always live
# under ``artifacts/`` so this is fine.
apply_manylinux_tag() {
    local wheel="$1"
    local abs_wheel
    abs_wheel="$(realpath "$wheel")"
    local new_wheel
    new_wheel="$(docker exec -u "$(id -u):$(id -g)" \
        "$_MANYLINUX_CONTAINER" \
        python "$_MANYLINUX_DETECT_SCRIPT" "$abs_wheel")"
    if [[ -z "$new_wheel" || ! -f "$new_wheel" ]]; then
        echo "apply_manylinux_tag: detect-manylinux-tag.py did not produce a valid wheel path for $wheel" >&2
        return 1
    fi
    printf '%s\n' "$new_wheel"
}
