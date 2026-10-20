# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""draft_lm_head.compute_logits, the DSpark draft's base logits on gfx942.

With --specialized-decode on gfx942, the draft's 5 rows of one request get a sixth
row of zeros, so that vLLM's dispatch (rocm_unquantized_gemm) takes the torch
GEMM instead of wvSplitK, and the logits of the first 5 rows come back. Every
other row count, and every call without --specialized-decode, keeps vLLM's dispatch,
which sends up to 5 rows to wvSplitK. The tests count the wvSplitK calls of
vLLM's dispatch on the draft's lm_head shard at TP 4 and compare the logits
with an fp32 GEMM.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the helper is for ROCm"
)

# The DSpark draft's lm_head shard at tensor parallel size 4.
VOCAB_SHARD, HIDDEN = 32320, 5120


def _skip_unless_gfx942():
    from vllm.platforms.rocm import on_gfx942

    if not on_gfx942():
        pytest.skip("the helper runs on gfx942 only")


def _set_gate(monkeypatch, on: bool):
    from vllm.models.deepseek_v41.amd import ops

    # The model records --specialized-decode when it is created. These tests have no
    # model, so they set the recorded flag themselves.
    monkeypatch.setattr(ops, "_specialized_decode", on)
    ops.enabled.cache_clear()


class _LMHead(torch.nn.Module):
    """The parts of ParallelLMHead that vLLM's GEMM dispatch uses."""

    def __init__(self, w):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


def _processor(head, x):
    """The logits processor without its tensor parallel gather, which needs a
    process group. The GEMM is the same."""
    return head.quant_method.apply(head, x)


@pytest.fixture
def wvsplitk_rows(monkeypatch):
    """The row counts of the wvSplitK calls that vLLM's dispatch makes."""
    from vllm.model_executor.layers import utils

    rows: list[int] = []
    real = utils.ops.wvSplitK

    def counted(weight, x, *args, **kwargs):
        rows.append(x.shape[0])
        return real(weight, x, *args, **kwargs)

    monkeypatch.setattr(utils.ops, "wvSplitK", counted)
    return rows


@pytest.mark.parametrize("specialized_decode", [False, True])
@pytest.mark.parametrize("rows", [1, 4, 5, 6, 10])
def test_draft_lm_head_takes_5_rows_off_wvsplitk(
    monkeypatch, wvsplitk_rows, rows, specialized_decode
):
    from vllm.models.deepseek_v41.amd.ops import draft_lm_head

    _skip_unless_gfx942()
    g = torch.Generator(device="cuda").manual_seed(rows)
    w = torch.randn(VOCAB_SHARD, HIDDEN, generator=g, device="cuda") * 0.02
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda")
    head = _LMHead(w.to(torch.bfloat16))

    _set_gate(monkeypatch, on=specialized_decode)
    try:
        got = draft_lm_head.compute_logits(_processor, head, x.to(torch.bfloat16))
        torch.accelerator.synchronize()
    finally:
        _set_gate(monkeypatch, on=False)

    # Only the 5 rows with --specialized-decode leave wvSplitK. Every other call
    # keeps vLLM's dispatch, which takes wvSplitK for up to 5 rows.
    padded = specialized_decode and rows == draft_lm_head.ROWS
    assert wvsplitk_rows == ([] if padded or rows > 5 else [rows])
    assert got.shape == (rows, VOCAB_SHARD)
    want = x.to(torch.bfloat16).float() @ w.to(torch.bfloat16).float().T
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)


def test_draft_lm_head_passes_on_a_rank_without_logits(monkeypatch):
    from vllm.models.deepseek_v41.amd.ops import draft_lm_head

    _skip_unless_gfx942()
    # The logits processor returns None on a rank that does not get the
    # gathered logits. The padded path must return None there too, instead
    # of slicing it.
    x = torch.zeros(draft_lm_head.ROWS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    seen: list[int] = []

    def processor(head, h):
        seen.append(h.shape[0])
        return None

    _set_gate(monkeypatch, on=True)
    try:
        assert draft_lm_head.compute_logits(processor, None, x) is None
    finally:
        _set_gate(monkeypatch, on=False)
    assert seen == [draft_lm_head.ROWS + 1]
