# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures for manual-fusion tests.

The counters here are installed *inside* each worker through ``collective_rpc``
rather than in the test process: with the V1 engine the model forward runs in
spawned worker processes, so a monkeypatch applied in the test process never
reaches the code under test and every count reads zero.

Two kinds of counters are installed:

- A ``TorchDispatchMode`` counting every ATen / vLLM C++ op dispatch, proving
  that fused *kernels* actually run (e.g. ``fused_add_rms_norm_static_fp8_quant``).
- Python-level spies wrapping the manual-fusion entry points, which the
  dispatch mode cannot see because they are plain Python functions -- most
  importantly ``flashinfer_trtllm_fused_allreduce_norm``.
"""

import os
from collections import Counter
from contextlib import contextmanager

import pytest

# Required so collective_rpc can ship our counter install/fetch closures to
# workers (vLLM defaults to a strict msgpack codec that rejects functions).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _install_python_spies(worker_self):
    """Wrap the manual-fusion Python entry points on this worker. Runs via RPC."""
    import vllm.model_executor.layers.fusion.ar_rms_quant as arq
    import vllm.model_executor.layers.fusion.residual_stream as rs
    import vllm.model_executor.models.qwen2 as qwen2

    state: Counter[str] = Counter()
    orig_fused = arq.fused_ar_rms_norm_quant
    orig_finalize = rs.finalize_norm
    orig_all_reduce = rs.tensor_model_parallel_all_reduce
    orig_flashinfer_ar = arq.flashinfer_trtllm_fused_allreduce_norm

    def counting_fused(hidden_states, residual, norm, consumer_linear, *, do_allreduce):
        state["prepare_calls"] += 1
        if do_allreduce:
            state["owed_allreduce"] += 1
        return orig_fused(
            hidden_states,
            residual,
            norm,
            consumer_linear,
            do_allreduce=do_allreduce,
        )

    def counting_finalize(norm, hidden_states, residual, *, incoming):
        # Once per model forward for every decoder, migrated or not, so it is
        # the forward counter the other counts are normalized against.
        state["forwards"] += 1
        return orig_finalize(norm, hidden_states, residual, incoming=incoming)

    def counting_all_reduce(x):
        state["explicit_allreduce"] += 1
        return orig_all_reduce(x)

    def counting_flashinfer_ar(**kwargs):
        state["flashinfer_ar_calls"] += 1
        return orig_flashinfer_ar(**kwargs)

    # ``residual_stream.py`` and ``ar_rms_quant.py`` imported the symbols
    # directly, so every binding must be patched for all call sites to be
    # counted -- including the all-reduce the helper itself falls back to.
    arq.fused_ar_rms_norm_quant = counting_fused
    rs.fused_ar_rms_norm_quant = counting_fused
    rs.finalize_norm = counting_finalize
    qwen2.finalize_norm = counting_finalize
    rs.tensor_model_parallel_all_reduce = counting_all_reduce
    arq.tensor_model_parallel_all_reduce = counting_all_reduce
    arq.flashinfer_trtllm_fused_allreduce_norm = counting_flashinfer_ar
    worker_self._python_spy_counts = state


def _fetch_python_spy_counts(worker_self) -> dict[str, int]:
    """Tear down the spy state and return its counts. Runs via RPC."""
    state = getattr(worker_self, "_python_spy_counts", None)
    if state is None:
        return {}
    del worker_self._python_spy_counts
    return dict(state)


def _install_dispatch_counter(worker_self):
    """Install a TorchDispatchMode op counter on this worker. Runs via RPC."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class _OpCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts: Counter[str] = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.counts[str(func)] += 1
            return func(*args, **(kwargs or {}))

    counter = _OpCounter()
    counter.__enter__()
    worker_self._dispatch_op_counter = counter


def _fetch_dispatch_counts(worker_self) -> dict[str, int]:
    """Tear down the counter and return its counts. Runs via RPC."""
    counter = getattr(worker_self, "_dispatch_op_counter", None)
    if counter is None:
        return {}
    counter.__exit__(None, None, None)
    del worker_self._dispatch_op_counter
    return dict(counter.counts)


@pytest.fixture
def ar_counter_session():
    """Aggregate manual-fusion and op-dispatch counts from every worker.

    The yielded ``Counter`` mixes two key spaces: short snake_case keys from
    the Python spies (``prepare_calls``, ``flashinfer_ar_calls``, ...) and full
    op names from the dispatch counter (``aten::...``, ``_C....``).

    Usage::

        with ar_counter_session(llm) as counts:
            llm.generate(...)
        assert counts["owed_allreduce"] > 0
    """

    @contextmanager
    def session(llm):
        llm.collective_rpc(_install_python_spies)
        llm.collective_rpc(_install_dispatch_counter)
        aggregated: Counter = Counter()
        try:
            yield aggregated
        finally:
            for spy_counts in llm.collective_rpc(_fetch_python_spy_counts):
                aggregated.update(spy_counts)
            for op_counts in llm.collective_rpc(_fetch_dispatch_counts):
                aggregated.update(op_counts)

    return session


def count_matching(counts: Counter, *needles: str) -> int:
    """Sum counts whose op name contains any of the given substrings."""
    return sum(c for op, c in counts.items() if any(n in op for n in needles))
