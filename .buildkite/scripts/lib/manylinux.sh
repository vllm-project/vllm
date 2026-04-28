#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared helper for rewriting a wheel's platform tag from the generic
# ``linux_<arch>`` to the correct ``manylinux_<major>_<minor>_<arch>``.
#
# Sourcing this file eagerly creates an isolated venv with auditwheel and
# registers an EXIT trap to clean it up. After sourcing, call
# ``apply_manylinux_tag <wheel>`` on each ``linux_<arch>`` wheel; the
# renamed path is printed on stdout.
#
# The venv is created eagerly (rather than lazily on first call) because
# callers use ``apply_manylinux_tag`` inside command substitution, which
# runs in a subshell -- any state or trap set up there would be lost the
# moment the substitution returns.
#
# Configuration:
# - ``MANYLINUX_PYTHON`` (env var, default ``python3``): the Python
#   interpreter used to build the venv. Must point to a directly-callable
#   ``python3`` >= 3.10 (auditwheel's floor); a Docker-wrapped Python
#   won't work because ``python3 -m venv`` would create the venv inside
#   the container.
#
# Trap behaviour:
# - Sourcing installs an EXIT trap that calls ``manylinux_cleanup``. Any
#   EXIT trap that was already in place when this file was sourced is
#   captured and run AFTER our cleanup, so we don't silently clobber it.
# - If a caller sets a new EXIT trap *after* sourcing, that trap will
#   replace ours. In that case the caller should call
#   ``manylinux_cleanup`` from their own handler.

if [[ -n "${_MANYLINUX_LIB_SOURCED:-}" ]]; then
    return 0
fi
_MANYLINUX_LIB_SOURCED=1

# Resolve our own directory regardless of caller cwd, so we can locate
# ``detect-manylinux-tag.py`` next to us.
_MANYLINUX_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MANYLINUX_DETECT_SCRIPT="${_MANYLINUX_LIB_DIR}/../detect-manylinux-tag.py"

# Pin auditwheel: ``detect-manylinux-tag.py`` reads the internal
# ``analyze_wheel_abi`` / ``sym_policy`` API, which is not part of any
# stability promise, so an upstream release could silently change the
# resulting platform tag. Bump this deliberately and re-test the
# detection on a representative wheel from each build target.
_MANYLINUX_AUDITWHEEL_VERSION="6.6.0"

_MANYLINUX_PYTHON="${MANYLINUX_PYTHON:-python3}"
_MANYLINUX_VENV_PARENT="$(mktemp -d)"
_MANYLINUX_VENV="${_MANYLINUX_VENV_PARENT}/auditwheel-venv"
"$_MANYLINUX_PYTHON" -m venv "$_MANYLINUX_VENV"
"$_MANYLINUX_VENV/bin/pip" install --quiet --disable-pip-version-check \
    "auditwheel==${_MANYLINUX_AUDITWHEEL_VERSION}"

# Public cleanup function -- safe to call multiple times.
manylinux_cleanup() {
    rm -rf "$_MANYLINUX_VENV_PARENT"
}

# Capture any EXIT trap that was already in place so we can chain to it
# rather than overwrite it. ``trap -p EXIT`` prints the existing handler
# in eval-able form (``trap -- 'CMD' EXIT``) or nothing if unset; we
# strip the wrapper to recover ``CMD``. This handles the common case --
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

# Detect the manylinux platform tag for a single wheel and rename it in
# place, printing the renamed wheel path on stdout. Returns non-zero on
# failure (which under ``set -e`` propagates to the caller).
apply_manylinux_tag() {
    local wheel="$1"
    local new_wheel
    new_wheel="$("$_MANYLINUX_VENV/bin/python" "$_MANYLINUX_DETECT_SCRIPT" "$wheel")"
    if [[ -z "$new_wheel" || ! -f "$new_wheel" ]]; then
        echo "apply_manylinux_tag: detect-manylinux-tag.py did not produce a valid wheel path for $wheel" >&2
        return 1
    fi
    printf '%s\n' "$new_wheel"
}
