#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Pick a Python interpreter for buildkite scripts: prefer a local
# ``python3`` if it is recent enough (>= 3.12), otherwise fall back to
# a one-shot Docker container running ``python:3-slim``. After
# ``select_python`` returns, ``$PYTHON`` is set in the caller's shell
# and is safe to use as a command (e.g. ``$PYTHON some_script.py``).
#
# The 3.12 threshold matches what the existing nightly-index work
# expects -- typing features used by ``generate-nightly-index.py``.
# This helper does not pin the *minor* version; if you need stricter
# reproducibility (e.g. relying on auditwheel internals), invoke
# Docker yourself with a pinned tag rather than calling this.

if [[ -n "${_SELECT_PYTHON_LIB_SOURCED:-}" ]]; then
    return 0
fi
_SELECT_PYTHON_LIB_SOURCED=1

# Sets ``PYTHON`` in the caller's shell and exports it. Idempotent --
# calling twice is safe and the second call simply re-runs the probe.
select_python() {
    local py="${PYTHON_PROG:-python3}"
    local has_new_python
    has_new_python=$("$py" -c \
        "print(1 if __import__('sys').version_info >= (3,12) else 0)" \
        2>/dev/null || echo 0)
    if [[ "$has_new_python" -eq 0 ]]; then
        # ``-u $(id -u):$(id -g)`` so files created via the container
        # end up owned by the host user, not root.
        docker pull python:3-slim
        PYTHON="docker run --rm -u $(id -u):$(id -g) -v $(pwd):/app -w /app python:3-slim python3"
    else
        PYTHON="$py"
    fi
    export PYTHON
    echo "Using python interpreter: $PYTHON"
    echo "Python version: $($PYTHON --version)"
}
