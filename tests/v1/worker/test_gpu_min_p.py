# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Min-p filtering preserves request mapping, logits and graph replay semantics."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for min-p tests", allow_module_level=True)

from vllm.v1.worker.gpu.sample.min_p import apply_min_p


def reference(logits, mapping, min_p):
    p = min_p[mapping.long()]
    threshold = logits.float().amax(-1) + p.log()
    return logits.masked_fill(
        (logits.float() < threshold[:, None]) & (p[:, None] != 0), -float("inf")
    )


@pytest.mark.parametrize("vocab_size", [128, 16384, 16385, 32000, 128256, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [0, 13])
def test_min_p(vocab_size, dtype, padding):
    torch.manual_seed(0)
    backing = torch.randn(8, vocab_size + padding, device="cuda", dtype=dtype)
    logits = backing[:, :vocab_size]
    # Repeated slots also cover expanded speculative positions, not just decode.
    mapping = torch.tensor([5, 1, 5, 2, 0, 1, 4, 3], device="cuda", dtype=torch.int32)
    min_p = torch.tensor([0.1, 0.0, 1.0, 1e-6, 0.5, 0.1], device="cuda")
    logits[4].fill_(-float("inf"))
    logits[6, -1] = float("inf")
    logits[7, -1] = float("nan")
    expected = reference(logits, mapping, min_p)
    padding_before = backing[:, vocab_size:].clone()
    apply_min_p(logits, mapping, min_p)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(backing[:, vocab_size:], padding_before, rtol=0, atol=0)


def test_min_p_graph_replay():
    torch.manual_seed(0)
    logits = torch.randn(4, 128256, device="cuda")
    mapping = torch.tensor([2, 0, 2, 1], device="cuda", dtype=torch.int32)
    min_p = torch.tensor([0.0, 0.1, 1.0], device="cuda")
    apply_min_p(logits, mapping, min_p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_min_p(logits, mapping, min_p)
    # Reused slots must read new parameters; neither row maxima nor no-op rows
    # can retain stale values from capture or the previous generation step.
    for values in ([0.5, 0.0, 0.1], [0.0, 1.0, 0.0]):
        logits.normal_()
        min_p.copy_(torch.tensor(values, device="cuda"))
        expected = reference(logits, mapping, min_p)
        graph.replay()
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [8, 9])
def test_min_p_batch_boundary(batch_size):
    torch.manual_seed(0)
    logits = torch.randn(batch_size, 32000, device="cuda")
    mapping = torch.arange(batch_size, device="cuda", dtype=torch.int32)
    min_p = torch.full((batch_size,), 0.1, device="cuda")
    expected = reference(logits, mapping, min_p)
    apply_min_p(logits, mapping, min_p)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
