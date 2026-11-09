# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vocab-parallel argmax kernels behind ``LogitsProcessor.get_top_tokens``
must equal ``torch.argmax`` over the concatenated logits bit for bit.

TP shards are simulated on one GPU: per-shard candidates are concatenated in
the layout of a dim-0 all-gather and reduced by the pick kernel. Covers ties
within and across shards, -inf rows, +inf ties, NaN (first NaN wins),
-0.0/+0.0 ties, uneven shards and CUDA graph replay.
"""

import pytest
import torch

from vllm.model_executor.layers.vocab_parallel_argmax import (
    local_argmax_candidates,
    pick_from_candidates,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernels require CUDA."
)


def _select(logits: torch.Tensor, tp_size: int) -> torch.Tensor:
    num_rows, vocab_size = logits.shape
    shard = -(-vocab_size // tp_size)
    candidates = [
        local_argmax_candidates(logits[:, r * shard : (r + 1) * shard], r * shard)
        for r in range(tp_size)
    ]
    return pick_from_candidates(torch.cat(candidates, 0), num_rows, tp_size)


def _cases(num_rows, vocab_size, dtype, gen):
    def randn():
        return torch.randn(num_rows, vocab_size, generator=gen, device="cuda").to(dtype)

    yield "randn", randn()
    yield (
        "int-ties",
        torch.randint(-3, 4, (num_rows, vocab_size), generator=gen, device="cuda").to(
            dtype
        ),
    )
    x = randn()
    x[:, [vocab_size // 3, (2 * vocab_size) // 3, vocab_size - 1]] = 50
    yield "cross-shard-ties", x
    yield "all-neg-inf", torch.full_like(x, float("-inf"))
    x = randn()
    x[:, [7, vocab_size // 2]] = float("inf")
    yield "pos-inf-ties", x
    x = randn()
    x[0, [vocab_size // 2 + 5, vocab_size - 2]] = float("nan")
    x[:, 100] = 6e4
    yield "nan", x
    x = torch.full_like(x, -1.0)
    x[:, vocab_size // 4 + 3] = -0.0
    x[:, 3 * vocab_size // 4] = 0.0
    yield "signed-zero-ties", x


@pytest.mark.parametrize("tp_size", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("vocab_size", [131072, 32001])
def test_matches_torch_argmax(tp_size, dtype, vocab_size):
    gen = torch.Generator(device="cuda").manual_seed(0)
    for num_rows in (1, 3, 8, 64):
        for name, x in _cases(num_rows, vocab_size, dtype, gen):
            ref = torch.argmax(x, dim=-1)
            got = _select(x, tp_size)
            assert torch.equal(ref, got), (name, num_rows)


@pytest.mark.parametrize("layout", ["column-major", "strided-columns"])
def test_non_contiguous_shard(layout):
    gen = torch.Generator(device="cuda").manual_seed(0)
    num_rows, vocab_size, tp_size = 5, 32001, 4
    for name, x in _cases(num_rows, vocab_size, torch.bfloat16, gen):
        if layout == "column-major":
            view = x.t().contiguous().t()
        else:
            wide = torch.empty(num_rows, 2 * vocab_size, device="cuda", dtype=x.dtype)
            wide[:, ::2] = x
            view = wide[:, ::2]
        assert not view.is_contiguous()
        assert torch.equal(_select(view, tp_size), torch.argmax(x, dim=-1)), name


def test_cuda_graph_replay():
    tp_size, num_rows, vocab_size = 4, 8, 131072
    gen = torch.Generator(device="cuda").manual_seed(0)
    static = torch.randn(num_rows, vocab_size, device="cuda").to(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _select(static, tp_size)  # warm-up / Triton compile
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _select(static, tp_size)
    for it in range(50):
        fresh = torch.randint(
            -2, 3, (num_rows, vocab_size), generator=gen, device="cuda"
        )
        static.copy_(fresh.to(torch.bfloat16))
        graph.replay()
        assert torch.equal(out, torch.argmax(static, -1)), it
