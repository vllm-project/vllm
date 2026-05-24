# SPDX-License-Identifier: Apache-2.0
"""Warmup-time skip context manager for Genesis kernels.

Pattern from vllm#41329 (FlashInfer autotune skip): some kernels
catastrophically fail on edge-case shapes during warmup `_dummy_run`
that never appear in steady-state. Provide an idempotent
`with warmup_active():` block scheduler/runner can wrap around
warmup; Genesis kernels can opt to fall through to upstream
fallback during warmup.

Example usage in a Genesis kernel:

    from vllm._genesis.utils.warmup_state import is_warmup_active

    def my_kernel_dispatch(q, k, v, ...):
        if is_warmup_active() and _shape_is_edge_case(q.shape):
            return _upstream_fallback(q, k, v, ...)
        return _genesis_fast_path(q, k, v, ...)

Why contextvars (not threading.local):
- Async-safe: vllm v1 worker uses asyncio loops
- Forks correctly across spawn workers
- Test-friendly (per-test isolation)

Memory precedent: feedback_27b_lorbus_compile_cache_regression
(2026-04-30) — stale compile cache vs new fixes broke long-form gen;
warmup-vs-steady-state separation would have caught earlier.

Author: Sandermage 2026-05-04, pattern from vllm#41329 (haosdent).
"""
from __future__ import annotations

import contextlib
import contextvars
import logging

log = logging.getLogger("genesis.utils.warmup_state")

_is_genesis_warmup: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "genesis_warmup_active", default=False,
)


def is_warmup_active() -> bool:
    """Returns True when current execution is inside a warmup context.

    Genesis kernels SHOULD check this and prefer upstream fallback for
    edge-case shapes during warmup (where dummy_run shapes may differ
    from steady-state and trigger kernel crashes).
    """
    return _is_genesis_warmup.get()


@contextlib.contextmanager
def warmup_active():
    """Mark current execution context as warmup (try/finally guaranteed reset).

    Wrap scheduler dummy_run / model_runner profile_run calls:

        with warmup_active():
            self.execute_model(dummy_inputs)
    """
    token = _is_genesis_warmup.set(True)
    try:
        yield
    finally:
        _is_genesis_warmup.reset(token)


def reset_warmup_state() -> None:
    """Force-reset warmup state. For test cleanup only — DO NOT use in prod."""
    try:
        _is_genesis_warmup.set(False)
    except Exception as e:
        log.warning("Failed to reset warmup state: %s", e)
