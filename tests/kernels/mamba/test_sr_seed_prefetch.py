# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity tests for VLLM_MAMBA_SR_SEED_PREFETCH.

The N per-layer stochastic-rounding seeds drawn up front on a side stream
(StochasticRoundingSeedPrefetcher, event-joined per use) must equal the N
per-layer `torch.randint(0, 2**32, (1,))` draws of stock, eagerly and across
CUDA graph replays, with other work on the current stream between uses.
"""

import pytest
import torch

import vllm.model_executor.layers.mamba.ops.ssu_dispatch as ssu_dispatch
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    StochasticRoundingSeedPrefetcher,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NUM_LAYERS = 40


class _FakeForwardContext:
    pass


@pytest.fixture
def forward_context(monkeypatch):
    """A fresh fake ForwardContext per forward pass: call `.new()`."""

    class _Holder:
        current: _FakeForwardContext | None = None

        def new(self) -> None:
            self.current = _FakeForwardContext()

    holder = _Holder()
    monkeypatch.setattr(
        ssu_dispatch, "is_forward_context_available", lambda: holder.current is not None
    )
    monkeypatch.setattr(ssu_dispatch, "get_forward_context", lambda: holder.current)
    return holder


def _stock(out, work, num_calls):
    for k in range(num_calls):
        seed = torch.randint(0, 2**32, (1,), device="cuda")
        out[k].copy_(seed)
        work.mul_(1.0)  # layer work between draws


def _prefetched(prefetcher, out, work, num_calls):
    # What FlashInferSSUBackend.__call__ does per SSU call.
    for k in range(num_calls):
        seed = prefetcher.next_seed(torch.device("cuda"))
        if seed is None:
            seed = torch.randint(0, 2**32, (1,), device="cuda")
        out[k].copy_(seed)
        work.mul_(1.0)


def _run_eager(fn, seed, num_calls):
    torch.manual_seed(seed)
    out = torch.zeros(num_calls, 1, dtype=torch.int64, device="cuda")
    fn(out)
    torch.cuda.synchronize()
    return out.clone()


def _run_graph(fn, seed, num_calls, reps=3):
    out = torch.zeros(num_calls, 1, dtype=torch.int64, device="cuda")
    torch.manual_seed(seed)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn(out)
    torch.cuda.current_stream().wait_stream(stream)
    res = []
    for _ in range(reps):
        graph.replay()
        torch.cuda.synchronize()
        res.append(out.clone())
    return res


@pytest.mark.parametrize("extra_calls", [0, 2])
def test_sr_seed_prefetch_eager_identical(forward_context, extra_calls):
    work = torch.randn(4096, 4096, device="cuda")
    prefetcher = StochasticRoundingSeedPrefetcher(NUM_LAYERS)
    num_calls = NUM_LAYERS + extra_calls

    def ours(out):
        forward_context.new()
        _prefetched(prefetcher, out, work, num_calls)

    for seed in (123, 456):
        ref = _run_eager(lambda out: _stock(out, work, num_calls), seed, num_calls)
        got = _run_eager(ours, seed, num_calls)
        assert torch.equal(ref, got)


def test_sr_seed_prefetch_cuda_graph_identical(forward_context):
    work = torch.randn(4096, 4096, device="cuda")
    prefetcher = StochasticRoundingSeedPrefetcher(NUM_LAYERS)

    def ours(out):
        forward_context.new()
        _prefetched(prefetcher, out, work, NUM_LAYERS)

    # Create the side stream and events outside capture, as vLLM's eager
    # warmup pass does before capturing.
    _run_eager(ours, 0, NUM_LAYERS)
    ref = _run_graph(lambda out: _stock(out, work, NUM_LAYERS), 7, NUM_LAYERS)
    got = _run_graph(ours, 7, NUM_LAYERS)
    for r, g in zip(ref, got):
        assert torch.equal(r, g)
    # Fresh seeds on every replay, as stock.
    assert not torch.equal(got[0], got[-1])


def test_sr_seed_prefetch_outside_forward_context_falls_back(forward_context):
    prefetcher = StochasticRoundingSeedPrefetcher(NUM_LAYERS)
    assert prefetcher.next_seed(torch.device("cuda")) is None
