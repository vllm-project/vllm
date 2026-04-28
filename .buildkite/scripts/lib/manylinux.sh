#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared helper for rewriting a wheel's platform tag from the generic
# ``linux_<arch>`` to the correct ``manylinux_<major>_<minor>_<arch>``.
#
# Sourcing this file eagerly creates an isolated venv with auditwheel and
# registers an EXIT trap to clean it up. After sourcing, call
# ``apply_manylinux_tag <wheel>`` on each ``linux_<arch>`` wheel; the renamed
# path is printed on stdout.
#
# The venv is created eagerly (rather than lazily on first call) because
# callers use ``apply_manylinux_tag`` inside command substitution, which
# runs in a subshell -- any state or trap set up there would be lost the
# moment the substitution returns.
#
# Caveats for callers:
# - The library installs an EXIT trap. Callers must not set their own EXIT
#   trap, or they will clobber the cleanup. (None of the current callers
#   do.)
# - The library expects ``python3`` (>= 3.10, auditwheel's floor) on PATH.

if [[ -n "${_MANYLINUX_LIB_SOURCED:-}" ]]; then
    return 0
fi
_MANYLINUX_LIB_SOURCED=1

# Resolve our own directory regardless of caller cwd, so we can locate
# ``detect-manylinux-tag.py`` next to us.
_MANYLINUX_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MANYLINUX_DETECT_SCRIPT="${_MANYLINUX_LIB_DIR}/../detect-manylinux-tag.py"

_MANYLINUX_VENV_PARENT="$(mktemp -d)"
_MANYLINUX_VENV="${_MANYLINUX_VENV_PARENT}/auditwheel-venv"
python3 -m venv "$_MANYLINUX_VENV"
"$_MANYLINUX_VENV/bin/pip" install --quiet --disable-pip-version-check auditwheel
# shellcheck disable=SC2064  # Expand the path now, not when the trap fires.
trap "rm -rf '$_MANYLINUX_VENV_PARENT'" EXIT

# Detect the manylinux platform tag for a single wheel and rename it in
# place, printing the renamed wheel path on stdout. Returns non-zero on
# failure (which under ``set -e`` propagates to caller).
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
