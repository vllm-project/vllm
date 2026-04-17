# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monitor unexpected kernel JIT compilation during inference.

After server warmup completes, any Triton JIT compilation or autotuning
event indicates a cache miss or unexpected input shape that causes a
latency spike. This module registers hooks in the Triton runtime to
detect and log such events so they can be investigated.

Currently monitors:
- Triton ``@triton.autotune`` cache misses (via ``knobs.autotuning.print``)
- Triton ``@triton.jit`` first-time compilations
  (via ``knobs.runtime.jit_post_compile_hook``)

Designed to be extended in the future with monitors for DeepGEMM,
FlashInfer, and other JIT-compiled kernel backends.
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_active: bool = False


def is_active() -> bool:
    """Return whether the JIT compilation monitor is currently active."""
    return _active


def activate() -> None:
    """Enable JIT compilation monitoring after warmup.

    Call once per worker process at the end of
    :func:`compile_or_warm_up_model`.  After activation every Triton
    kernel compilation or autotuning benchmark that happens during
    inference will be logged as a warning.

    Safe to call multiple times — subsequent calls are no-ops.

    If the user has explicitly set ``TRITON_PRINT_AUTOTUNING=0`` in
    their environment, autotuning printing is left disabled; the JIT
    compilation hook is still registered regardless.
    """
    global _active
    if _active:
        return
    _active = True

    _setup_triton_autotuning_print()
    _setup_triton_jit_hook()

    logger.debug(
        "Kernel JIT monitor activated — Triton JIT compilations "
        "during inference will be logged as warnings."
    )


# ------------------------------------------------------------------
# Triton autotuning print
# ------------------------------------------------------------------


def _setup_triton_autotuning_print() -> None:
    """Enable ``TRITON_PRINT_AUTOTUNING`` unless the user opted out."""
    try:
        from triton import knobs  # type: ignore[import-untyped]
    except ImportError:
        return

    user_val = os.environ.get("TRITON_PRINT_AUTOTUNING")
    if user_val == "0":
        logger.debug(
            "TRITON_PRINT_AUTOTUNING=0 set by user — "
            "autotuning messages will stay suppressed."
        )
        return

    knobs.autotuning.print = True


# ------------------------------------------------------------------
# Triton JIT compilation hook
# ------------------------------------------------------------------


def _setup_triton_jit_hook() -> None:
    """Register a ``jit_post_compile_hook`` that warns on compilation."""
    try:
        from triton import knobs  # type: ignore[import-untyped]
    except ImportError:
        return

    existing_hook = knobs.runtime.jit_post_compile_hook

    def _on_jit_compile(
        *,
        key,
        repr,  # noqa: A002  (shadows builtin, but matches Triton's API)
        fn,
        compile,  # noqa: A002
        is_manual_warmup,
        already_compiled,
    ):
        logger.warning(
            "Triton kernel JIT compilation during inference: %s. "
            "This causes a latency spike; consider extending warmup "
            "to cover this shape/config.",
            fn.name,
        )
        if existing_hook is not None:
            return existing_hook(
                key=key,
                repr=repr,
                fn=fn,
                compile=compile,
                is_manual_warmup=is_manual_warmup,
                already_compiled=already_compiled,
            )
        return None

    knobs.runtime.jit_post_compile_hook = _on_jit_compile
