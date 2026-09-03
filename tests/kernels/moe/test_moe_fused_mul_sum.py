# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused MoE topk weight-and-reduce kernel.

Run `pytest tests/kernels/moe/test_moe_fused_mul_sum.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)
from vllm.utils.torch_utils import set_random_seed

NUM_TOKENS = [1, 17, 256]
TOP_KS = [2, 8]
HIDDEN_SIZE = 512
# One BLOCK_K tile, a partial trailing tile, and Kimi-K3's real dims
# (moe_intermediate_size 3072, hidden_size 7168) -- exercises the per-tile loop
# and its tail mask (BLOCK_K caps at 512 for <=2-byte dtypes).
HIDDEN_SIZES = [512, 1000, 3072, 7168]
DTYPES = [torch.bfloat16, torch.float32]
NUM_EXPERTS = 16


def _reference(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    valid = topk_ids >= 0
    if expert_map is not None:
        valid &= expert_map[topk_ids.clamp(min=0)] >= 0
    weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))
    return (inputs.float() * weights.float().unsqueeze(-1)).sum(dim=1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("use_expert_map", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_invalid_slots_excluded(num_tokens: int, top_k: int, use_expert_map: bool):
    """Slots marked -1 must not contribute, and must not be dereferenced.

    The expert GEMM never writes rows for topk slots whose expert id is -1
    (a non-local expert under EP, or an all2all padding row), so those rows
    hold stale workspace data. Filling them with NaN here means any failure to
    mask them shows up as NaN in the output rather than a small numeric drift.
    Indexing `expert_map` with -1 is also an out-of-bounds read.
    """
    set_random_seed(0)
    device = "cuda"

    inputs = torch.randn(
        num_tokens, top_k, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    # Mark roughly half the slots invalid and poison the rows behind them.
    invalid = torch.rand(num_tokens, top_k, device=device) < 0.5
    topk_ids[invalid] = -1
    inputs[invalid] = float("nan")

    expert_map = None
    if use_expert_map:
        # Second half of the experts is non-local, so valid ids can still map
        # to -1 -- exercising the second masking condition.
        expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
        local = NUM_EXPERTS // 2
        expert_map[:local] = torch.arange(local, dtype=torch.int32, device=device)
        inputs[(topk_ids >= 0) & (expert_map[topk_ids.clamp(min=0)] < 0)] = float("nan")

    # The _reference implementation zeros out outputs. We need to match that here,
    # otherwise the test fails.
    outputs = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    assert not out.isnan().any(), "invalid slots leaked into the reduction"
    ref = _reference(inputs.nan_to_num(), topk_weights, topk_ids, expert_map)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def _local_expert_map(device: str) -> torch.Tensor:
    """First half of the experts is local, second half maps to -1 (non-local)."""
    expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    local = NUM_EXPERTS // 2
    expert_map[:local] = torch.arange(local, dtype=torch.int32, device=device)
    return expert_map


SENTINEL = 42.0  # exact in bf16; marks output the kernel must leave untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nonlocal_row_is_zeroed():
    """A real row whose positive ids all map to non-local experts is written 0.

    Such a row is not padding (its ids are >= 0, so row_valid > 0 and the block
    is not early-returned). Every slot is then masked out by expert_map, so the
    accumulator stays 0 and 0 is stored -- the row must not be left holding stale
    buffer data, because downstream reduction treats it as a real token.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k = 17, 8
    local = NUM_EXPERTS // 2
    expert_map = _local_expert_map(device)

    # Poison inputs so any failure to mask the non-local slots surfaces as NaN.
    inputs = torch.full(
        (num_tokens, top_k, HIDDEN_SIZE),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    # Every id lands in the non-local half [local, NUM_EXPERTS).
    topk_ids = torch.randint(
        local, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    outputs = torch.full(
        (num_tokens, HIDDEN_SIZE), SENTINEL, dtype=torch.bfloat16, device=device
    )
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    assert not out.isnan().any(), "non-local row leaked NaN into the output"
    torch.testing.assert_close(out.float(), torch.zeros_like(out.float()))


@pytest.mark.parametrize("use_expert_map", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_padding_row_left_untouched(use_expert_map: bool):
    """An all -1 padding row is skipped: its output row is left untouched.

    This is the CUDA-graph decode contract -- padding rows past num_recv are
    never read downstream, so the kernel elides them (whole-padding blocks
    early-return) rather than zeroing. The skip is gated on topk_ids alone, so
    it must hold whether or not expert_map is passed (the humming path passes
    None); we pin both.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k = 17, 8
    expert_map = _local_expert_map(device) if use_expert_map else None

    inputs = torch.full(
        (num_tokens, top_k, HIDDEN_SIZE),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    topk_ids = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device=device)

    outputs = torch.full(
        (num_tokens, HIDDEN_SIZE), SENTINEL, dtype=torch.bfloat16, device=device
    )
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    assert torch.all(out == SENTINEL), "padding row output was modified"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nonlocal_and_padding_adjacent():
    """A non-local real row next to an all -1 padding row.

    Each CTA owns one token row: the non-local row must be zeroed (its slots are
    all masked by expert_map) while the padding row beside it is left untouched.
    Guards against a change that lets one row's fate leak into its neighbour.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k = 17, 8
    local = NUM_EXPERTS // 2
    expert_map = _local_expert_map(device)

    inputs = torch.full(
        (num_tokens, top_k, HIDDEN_SIZE),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    # All rows are non-local real rows except row 1, an all -1 padding row.
    topk_ids = torch.randint(
        local, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_ids[1] = -1

    outputs = torch.full(
        (num_tokens, HIDDEN_SIZE), SENTINEL, dtype=torch.bfloat16, device=device
    )
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    padding = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    padding[1] = True
    assert torch.all(out[1] == SENTINEL), "padding row beside a real row was modified"
    written = out[~padding].float()
    assert not written.isnan().any(), "non-local row leaked NaN into the output"
    torch.testing.assert_close(written, torch.zeros_like(written))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weighted_sum_matches_reference(
    num_tokens: int, top_k: int, hidden_size: int, dtype: torch.dtype
):
    """Weighted sum over live slots matches the reference across hidden tiles.

    Covers the deployed humming path (globalized ids, expert_map=None): a random
    -1 mix with NaN-poisoned dead slots, over hidden sizes that span a single
    tile, a partial trailing tile, and many tiles, in both bf16 and fp32.
    """
    set_random_seed(0)
    device = "cuda"

    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    invalid = torch.rand(num_tokens, top_k, device=device) < 0.5
    topk_ids[invalid] = -1
    inputs[invalid] = float("nan")

    # All-padding rows early-return untouched, so pre-zero to match the reference.
    outputs = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=None,
    )

    assert not out.isnan().any(), "invalid slots leaked into the reduction"
    ref = _reference(inputs.nan_to_num(), topk_weights, topk_ids, expert_map=None)
    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_no_topk_ids_sums_all_slots(dtype: torch.dtype):
    """With topk_ids=None every slot is live: plain weighted sum over top_k."""
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k, hidden_size = 17, 8, 1000

    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
    out = moe_fused_mul_sum(inputs=inputs, topk_weights=topk_weights)

    ref = (inputs.float() * topk_weights.float().unsqueeze(-1)).sum(dim=1)
    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("hidden_size", [512, 7168])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_map_is_required_for_nonlocal_ids(hidden_size: int, dtype: torch.dtype):
    """expert_map is load-bearing, not redundant: dropping it leaks non-local ids.

    A positive id whose expert is non-local marks a slot the GEMM never wrote, so
    its `inputs` row is stale. `topk_ids >= 0` alone cannot tell it apart from a
    real slot -- only expert_map can. This pins the contract the humming path
    relies on (globalized ids are still routed through expert_map as a safety
    net): with the map the stale slot is masked; without it, it corrupts the sum.
    Regression guard against re-dropping expert_map as an optimization.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k = 8, 8
    local = NUM_EXPERTS // 2
    expert_map = _local_expert_map(device)

    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
    # Each row keeps at least one local slot (so the mapped output is finite) and
    # at least one non-local slot whose stale row is poisoned with NaN.
    half = top_k // 2
    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
    topk_ids[:, :half] = torch.randint(
        0, local, (num_tokens, half), dtype=torch.int32, device=device
    )
    topk_ids[:, half:] = torch.randint(
        local, NUM_EXPERTS, (num_tokens, top_k - half), dtype=torch.int32, device=device
    )
    inputs[:, half:] = float("nan")

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)

    mapped = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device),
        topk_ids=topk_ids,
        expert_map=expert_map,
    )
    assert not mapped.isnan().any(), "expert_map failed to mask stale non-local slot"
    ref = _reference(inputs.nan_to_num(), topk_weights, topk_ids, expert_map)
    torch.testing.assert_close(mapped.float(), ref, atol=atol, rtol=rtol)

    # Same call without the map: the non-local slot is summed and the NaN leaks.
    unmapped = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device),
        topk_ids=topk_ids,
        expert_map=None,
    )
    assert unmapped.isnan().any(), "expert_map is not redundant: dropping it must leak"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_num_valid_tokens_bounds_stale_padding(dtype: torch.dtype):
    """Rows past num_valid_tokens are left untouched, even with stale ids/NaN.

    Under cudagraph decode the grid is the static padded token count and the
    tail rows past num_recv hold stale (non -1) ids and stale NaN in `inputs`
    from a prior replay -- the -1 mask alone cannot spot them. Bounding on
    num_valid_tokens must skip the tail before any load so its NaN never reaches
    the output; without the bound it is summed and a masked finalize can then
    propagate the NaN. Regression guard for the decode accuracy fix.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, num_recv, top_k, hidden_size = 17, 10, 8, 512

    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
    # All ids valid (>= 0): the stale tail is indistinguishable by the -1 mask.
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    inputs[num_recv:] = float("nan")  # stale down_output behind the padding tail
    num_valid = torch.tensor([num_recv], dtype=torch.int32, device=device)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)

    bounded = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=torch.full(
            (num_tokens, hidden_size), SENTINEL, dtype=dtype, device=device
        ),
        topk_ids=topk_ids,
        num_valid_tokens=num_valid,
    )
    assert torch.all(bounded[num_recv:] == SENTINEL), (
        "padding tail past num_recv written"
    )
    assert not bounded[:num_recv].isnan().any(), "real rows leaked NaN"
    ref = _reference(
        inputs[:num_recv].nan_to_num(),
        topk_weights[:num_recv],
        topk_ids[:num_recv],
        expert_map=None,
    )
    torch.testing.assert_close(bounded[:num_recv].float(), ref, atol=atol, rtol=rtol)

    # Without the bound the stale tail is summed and leaks NaN into its rows.
    unbounded = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=torch.full(
            (num_tokens, hidden_size), SENTINEL, dtype=dtype, device=device
        ),
        topk_ids=topk_ids,
    )
    assert unbounded[num_recv:].isnan().any(), "num_valid_tokens bound is load-bearing"
