# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal FlyDSL compilation and dispatch helper."""

import flydsl.compiler as flyc
from flydsl._mlir import ir


def run_compiled(executable, *args):
    """Compile on first use and dispatch the cached function thereafter."""
    compiled = getattr(executable, "_vllm_compiled", None)
    if compiled is not None:
        compiled(*args)
        return
    try:
        executable._vllm_compiled = flyc.compile(executable, *args)
    except Exception:
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:
            pass
        raise
