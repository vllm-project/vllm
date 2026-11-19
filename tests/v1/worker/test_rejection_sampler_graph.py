# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity and fallback tests for the graph-captured rejection sampler."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_graph import (
    RejectionSamplerGraph,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_inputs(
    device: str, n_reqs: int = 2, steps: int = 2, vocab: int = 64, seed: int = 0
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_logits = n_reqs * (steps + 1)
    max_reqs = 4

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.float32, generator=g).to(device)

    target_logits = randn(n_logits, vocab)
    draft_values = randn(n_reqs, steps, vocab)
    # Draft token ids per row; -1 marks the bonus position.
    draft_sampled = torch.randint(
        0, vocab, (n_logits,), dtype=torch.int64, generator=g
    ).to(device)
    for i in range(n_reqs):
        draft_sampled[i * (steps + 1) + steps] = -1
    cu_num_logits = torch.arange(
        0, (n_reqs + 1) * steps + steps, steps + 1, dtype=torch.int32
    ).to(device)
    pos = torch.arange(n_logits, dtype=torch.int64).to(device)
    idx_mapping = torch.arange(n_reqs, dtype=torch.int32).to(device)
    expanded_idx_mapping = torch.repeat_interleave(
        torch.arange(n_reqs, dtype=torch.int32), steps + 1
    ).to(device)
    expanded_local_pos = (
        torch.tile(torch.arange(steps + 1, dtype=torch.int32), (n_reqs,))
    ).to(device)
    temperature = torch.ones(max_reqs, dtype=torch.float32).to(device)
    seed = torch.arange(max_reqs, dtype=torch.int64).to(device)
    draft_cache = SimpleNamespace(values=draft_values, row_mapping=None)
    return (
        target_logits,
        draft_cache,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        steps,
    )


def _run_eager(args):
    return rejection_sample(*args)


def _run_graph(manager: RejectionSamplerGraph, args):
    return rejection_sample(*args, graph=manager)


def _assert_parity(result_a, result_b):
    """Compare two rejection_sample results on their valid prefix.

    `sampled` rows are valid only up to `num_sampled[r]`; the tail is
    uninitialized on the eager path (`new_empty`), so only the consumed
    prefix may be compared.
    """
    sampled_a, num_a = result_a
    sampled_b, num_b = result_b
    assert torch.equal(num_a, num_b)
    for r in range(num_a.shape[0]):
        k = int(num_a[r])
        assert torch.equal(sampled_a[r, :k], sampled_b[r, :k])


def test_graph_matches_eager_and_replays():
    device = "cuda"
    manager = RejectionSamplerGraph(torch.device(device))

    args_a = _make_inputs(device, seed=0)
    eager = _run_eager(args_a)

    graph = _run_graph(manager, args_a)
    # First graph call captures and must match eager bit-exactly.
    _assert_parity(graph, eager)

    # Same inputs replayed through the graph: still identical.
    _assert_parity(_run_graph(manager, args_a), eager)

    # Different inputs: the replay must track the new data, matching a
    # fresh eager run on the same inputs (guards against stale replays).
    args_b = _make_inputs(device, seed=1)
    _assert_parity(_run_graph(manager, args_b), _run_eager(args_b))


def test_capture_failure_falls_back_to_eager():
    device = "cuda"
    manager = RejectionSamplerGraph(torch.device(device))
    manager._fill_static_inputs = lambda *args, **kwargs: (raise_capture_error())

    args = _make_inputs(device, seed=2)
    result = _run_graph(manager, args)
    # The capture failed; the step must fall back to eager results and the
    # manager must be permanently disabled.
    eager = _run_eager(args)
    _assert_parity(result, eager)
    assert manager._disabled
    assert manager._graph is None
    assert manager._sampled_buf is None

    # Later steps keep running eagerly (no crash, consistent results).
    args2 = _make_inputs(device, seed=3)
    _assert_parity(_run_graph(manager, args2), _run_eager(args2))


def raise_capture_error():
    raise RuntimeError("forced capture failure")
