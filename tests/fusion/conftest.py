# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures for manual-fusion tests.

The harness lets a test prove that a manual fusion op actually runs in model
code, even when every compiler-driven fusion pass is disabled. The mechanism
is a ``TorchDispatchMode`` installed inside each worker via ``collective_rpc``
that counts every ATen / vLLM C++ op call, plus a per-worker registry of
Python-level wrappers (e.g. ``flashinfer_trtllm_fused_allreduce_norm``) that
TorchDispatchMode cannot see.
"""

import os
from collections import Counter
from contextlib import contextmanager

import pytest

# Required so collective_rpc can ship our counter install/fetch closures to
# workers (vLLM defaults to a strict msgpack codec that rejects functions).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _install_dispatch_counter(worker_self):
    """Install a TorchDispatchMode counter on this worker. Runs via RPC."""
    from collections import Counter

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
    worker_self._fusion_op_counter = counter


def _fetch_dispatch_counts(worker_self) -> dict[str, int]:
    """Tear down the counter and return its dict. Runs via RPC."""
    counter = getattr(worker_self, "_fusion_op_counter", None)
    if counter is None:
        return {}
    counter.__exit__(None, None, None)
    del worker_self._fusion_op_counter
    return dict(counter.counts)


@pytest.fixture
def op_count_session():
    """Context manager that yields a Counter aggregated across workers.

    Usage:

        def test_x(op_count_session):
            llm = LLM(...)
            with op_count_session(llm) as counts:
                llm.generate(...)
            assert counts["_C.fused_add_rms_norm_static_fp8_quant.default"] > 0
    """

    @contextmanager
    def session(llm):
        llm.collective_rpc(_install_dispatch_counter)
        aggregated: Counter[str] = Counter()
        try:
            yield aggregated
        finally:
            per_worker = llm.collective_rpc(_fetch_dispatch_counts)
            for d in per_worker:
                aggregated.update(d)

    return session


def count_matching(counts: Counter, *needles: str) -> int:
    """Sum counts whose op name contains any of the given substrings."""
    return sum(c for op, c in counts.items() if any(n in op for n in needles))
