# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monitor unexpected kernel JIT compilation during inference.

After server warmup completes, any kernel JIT compilation or autotuning event
indicates a cache miss or unexpected input shape that causes a latency spike.
This module registers hooks in supported runtimes to detect such events so
they can be investigated.

Set ``--jit-monitor-mode=error`` to fail fast on unexpected runtime
compilation. Set ``--jit-monitor-verbose`` to log every JIT compile with
additional runtime details. Verbose logging is intentionally opt-in because it
can emit many logs and add overhead.

Currently monitors:
- CuTeDSL cute.compile calls
- Triton ``@triton.autotune`` cache misses (via ``knobs.autotuning.print``)
- Triton ``@triton.jit`` first-time compilations
  (via ``knobs.runtime.jit_post_compile_hook``)
"""

import functools
import os
from typing import Literal

from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)
JitMonitorMode = Literal["warn", "error"]

_active: bool = False
_mode: JitMonitorMode = "warn"
_verbose: bool = False
_cutedsl_hook_installed: bool = False


def is_active() -> bool:
    """Return whether the JIT compilation monitor is currently active."""
    return _active


def activate(*, mode: JitMonitorMode = "warn", verbose: bool = False) -> None:
    """Enable JIT compilation monitoring after warmup.

    Call once per worker process at the end of
    :func:`compile_or_warm_up_model`. After activation every monitored kernel
    compilation or autotuning benchmark that happens during inference will be
    logged as a warning or raised as an error, depending on ``mode``.

    Safe to call multiple times; subsequent calls are no-ops.

    If the user has explicitly set ``TRITON_PRINT_AUTOTUNING=0`` in
    their environment, autotuning printing is left disabled; the JIT
    compilation hook is still registered regardless.
    """
    global _active, _mode, _verbose
    if _active:
        return
    if mode not in ("warn", "error"):
        raise ValueError(f"Unsupported JIT monitor mode: {mode!r}")
    _active = True
    _mode = mode
    _verbose = verbose

    _setup_triton_autotuning_print()
    _setup_triton_jit_hook()
    _setup_cutedsl_jit_hook()

    logger.info(
        "Kernel JIT monitor activated; monitored JIT compilations during "
        "inference will use mode=%s.",
        mode,
    )


# ------------------------------------------------------------------
# Triton autotuning print
# ------------------------------------------------------------------


def _setup_triton_autotuning_print() -> None:
    """Enable ``TRITON_PRINT_AUTOTUNING`` unless the user opted out."""
    if not HAS_TRITON:
        return
    from triton import knobs  # type: ignore[import-untyped]

    user_val = os.environ.get("TRITON_PRINT_AUTOTUNING")
    if user_val == "0":
        logger.debug(
            "TRITON_PRINT_AUTOTUNING=0 set by user; "
            "autotuning messages will stay suppressed."
        )
        return

    knobs.autotuning.print = True


# ------------------------------------------------------------------
# Triton JIT compilation hook
# ------------------------------------------------------------------


def _handle_jit_event(
    *,
    backend: str,
    event: str,
    fn_name: str,
    detail: str | None = None,
) -> None:
    message = (
        "%s %s during inference: %s%s. "
        "This causes a latency spike; consider extending warmup "
        "to cover this shape/config."
    )
    detail_suffix = f" ({detail})" if detail else ""
    args = (backend, event, fn_name, detail_suffix)

    if _mode == "error":
        raise RuntimeError(message % args)

    if _verbose:
        logger.warning(message, *args)
        return

    logger.warning_once(message, *args)


def _log_triton_jit_compile(fn_name: str, kwargs) -> None:
    compile_info = kwargs.get("compile")
    if not isinstance(compile_info, dict):
        compile_info = {}
    key = compile_info.get("key") or kwargs.get("key")
    detail = f"key={key}" if _verbose and key is not None else None
    event = (
        "autotune/warmup candidate JIT compilation"
        if kwargs.get("warmup")
        else "kernel JIT compilation"
    )
    _handle_jit_event(
        backend="Triton",
        event=event,
        fn_name=fn_name,
        detail=detail,
    )


def _setup_triton_jit_hook() -> None:
    """Register a ``jit_post_compile_hook`` that warns on compilation."""
    if not HAS_TRITON:
        return
    from triton import knobs  # type: ignore[import-untyped]

    existing_hook = knobs.runtime.jit_post_compile_hook

    def _on_jit_compile(**kwargs):
        # `jit_post_compile_hook` is Triton internal API and its
        # signature has changed across releases (kwargs added/renamed).
        # Accept **kwargs so an upstream change cannot crash this hook
        # with TypeError, and forward the full kwarg set to any
        # pre-existing hook unchanged.
        fn = kwargs.get("fn")
        fn_name = getattr(fn, "name", "<unknown>")
        _log_triton_jit_compile(fn_name, kwargs)
        if existing_hook is not None:
            return existing_hook(**kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = _on_jit_compile


# ------------------------------------------------------------------
# CuTeDSL JIT compilation hook
# ------------------------------------------------------------------


def _log_cutedsl_jit_compile(fn_name: str) -> None:
    _handle_jit_event(
        backend="CuTeDSL",
        event="JIT compilation",
        fn_name=fn_name,
    )


def _setup_cutedsl_jit_hook() -> None:
    """Wrap ``cutlass.cute.compile`` to warn on compilation."""
    global _cutedsl_hook_installed
    if _cutedsl_hook_installed:
        return

    try:
        import cutlass.cute as cute
    except Exception:
        logger.debug("CuTeDSL is not available; skipping CuTeDSL JIT monitor.")
        return

    original_compile = cute.compile

    @functools.wraps(original_compile)
    def _compile_with_monitor(*args, **kwargs):
        kernel = args[0] if args else kwargs.get("function")
        kernel_name = getattr(kernel, "__name__", None)
        if kernel_name is None:
            kernel_name = (
                kernel.__class__.__name__ if kernel is not None else "<unknown>"
            )
        _log_cutedsl_jit_compile(kernel_name)
        return original_compile(*args, **kwargs)

    cute.compile = _compile_with_monitor
    _cutedsl_hook_installed = True
