# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU correctness for the FA4 mm_prefix CuTe mask_mod.

The CPU tests in ``test_mm_prefix_query_ranges.py`` only cover metadata
construction. These run the actual kernel and compare against a dense float32
reference, which is the only way to check the two things the mask_mod does that
cannot be reasoned about from the Python side:

* ``q_ranges[token_idx, 0]`` indexing an aux tensor with a runtime ``Int32``.
* ``cu_seqlens_q[b] + q_local`` matching FA4's own packing of a varlen batch.

Passing ``mask_mod`` makes FA4 resolve causal and local to False
(``_resolve_causal_local_window``), so the mask_mod is the *entire* mask and
the reference has to reproduce the causal and sliding-window terms too.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import _make_mm_prefix_mask_mod
from vllm.v1.attention.backends.utils import fill_mm_prefix_query_ranges

if not current_platform.is_cuda():
    pytest.skip("FA4 mm_prefix mask_mod is CUDA only", allow_module_level=True)

from vllm.v1.attention.backends.fa_utils import (  # noqa: E402
    is_fa_version_supported,
)

if not is_fa_version_supported(4):
    pytest.skip("FA4 not supported on this device", allow_module_level=True)

from vllm.vllm_flash_attn.flash_attn_interface import (  # noqa: E402
    flash_attn_varlen_func,
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HEAD_SIZE = 128
NUM_HEADS = 8
NUM_KV_HEADS = 8


def _cu_seqlens(lens: list[int]) -> torch.Tensor:
    out = torch.zeros(len(lens) + 1, dtype=torch.int32)
    out[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    return out


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_lens: list[int],
    seq_lens: list[int],
    mm_ranges: dict[int, list[tuple[int, int]]],
    sliding_window_left: int | None,
    scale: float,
    mm_clamp_sw: int = 0,
) -> torch.Tensor:
    """Dense float32 ``(causal AND window) OR mm_prefix`` per request."""
    out = torch.empty_like(q, dtype=torch.float32)
    q_off = 0
    k_off = 0
    for req_idx, (q_len, k_len) in enumerate(zip(query_lens, seq_lens)):
        ctx = k_len - q_len
        q_pos = torch.arange(q_len, device=DEVICE) + ctx
        k_pos = torch.arange(k_len, device=DEVICE)
        delta = q_pos[:, None] - k_pos[None, :]

        keep = delta >= 0
        if sliding_window_left is not None:
            keep &= delta < sliding_window_left

        for start, end in mm_ranges.get(req_idx, []):
            if start >= end:
                continue
            q_in = (q_pos >= start) & (q_pos <= end)
            k_in = (k_pos >= start) & (k_pos <= end)
            mm = q_in[:, None] & k_in[None, :]
            if mm_clamp_sw > 0:
                mm &= delta < mm_clamp_sw
            keep |= mm

        q_i = q[q_off : q_off + q_len].float().transpose(0, 1)
        k_i = k[k_off : k_off + k_len].float().transpose(0, 1)
        v_i = v[k_off : k_off + k_len].float().transpose(0, 1)
        if NUM_HEADS != NUM_KV_HEADS:
            repeats = NUM_HEADS // NUM_KV_HEADS
            k_i = k_i.repeat_interleave(repeats, dim=0)
            v_i = v_i.repeat_interleave(repeats, dim=0)

        scores = (q_i @ k_i.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~keep[None], float("-inf"))
        out[q_off : q_off + q_len] = (scores.softmax(-1) @ v_i).transpose(0, 1)

        q_off += q_len
        k_off += k_len
    return out


def _run_kernel(
    q,
    k,
    v,
    query_lens,
    seq_lens,
    mm_ranges,
    sliding_window_left,
    scale,
    mm_clamp_sw=0,
):
    cu_q = _cu_seqlens(query_lens)
    cu_k = _cu_seqlens(seq_lens)

    staging = torch.full((int(cu_q[-1]), 2), 12345, dtype=torch.int32).numpy()
    num_rows = fill_mm_prefix_query_ranges(
        staging,
        mm_ranges,
        cu_q,
        torch.tensor(seq_lens, dtype=torch.int32),
    )
    assert num_rows > 0, "test case must have at least one in-range query token"
    q_ranges = torch.from_numpy(staging[:num_rows]).to(DEVICE)
    cu_q_gpu = cu_q.to(DEVICE)

    mask_mod = _make_mm_prefix_mask_mod(
        sliding_window=mm_clamp_sw,
        sliding_window_left=sliding_window_left,
    )
    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q_gpu,
        cu_seqlens_k=cu_k.to(DEVICE),
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(seq_lens),
        softmax_scale=scale,
        causal=True,
        fa_version=4,
        mask_mod=mask_mod,
        aux_tensors=[q_ranges, cu_q_gpu],
    )
    return out


# (name, query_lens, seq_lens, mm_ranges)
CASES = [
    # Production pooling shape: query and key positions coincide.
    (
        "single_no_context",
        [256],
        [256],
        {0: [(16, 79), (100, 163)]},
    ),
    # Exercises cu_seqlens_q[b] + q_local. Unequal query lens across the batch
    # is the packing the pooling workload never produced.
    (
        "varlen_batch",
        [128, 64, 200],
        [128, 64, 200],
        {0: [(8, 71)], 1: [(0, 31), (40, 55)], 2: [(64, 191)]},
    ),
    # seqlen_k > seqlen_q: q_abs = q_idx + (seqlen_k - seqlen_q). Ranges are
    # absolute prompt positions, so part of each range sits in the context.
    (
        "context_offset",
        [64],
        [320],
        {0: [(32, 95), (200, 287)]},
    ),
    # A prefill chunk landing entirely inside one range, plus one whose range
    # is fully behind the chunk (must degrade to pure causal for that request).
    (
        "chunked_prefill_mix",
        [96, 48],
        [352, 200],
        {0: [(256, 351)], 1: [(0, 63)]},
    ),
    # Decode rows: query token is generated, so it is in no range and must get
    # the (-1, -1) sentinel. One prefill row keeps the batch mixed.
    (
        "mixed_prefill_decode",
        [160, 1, 1, 1],
        [160, 512, 300, 97],
        {0: [(32, 127)], 1: [(4, 259)], 2: [(8, 71)], 3: [(1, 64)]},
    ),
    # A 320-token range, wider than SLIDING_WINDOW_LEFT. Every other case has
    # ranges narrower than the window, so this is the only one where the Gemma4
    # clamp changes the mask -- images larger than the window are exactly why
    # mm_prefix_clamp_sliding_window exists.
    (
        "range_wider_than_window",
        [384],
        [384],
        {0: [(32, 351)]},
    ),
]

SLIDING_WINDOW_LEFT = 129

# (sliding_window_left, mm_clamp_sw). Gemma4 sets mm_clamp_sw == sw_val on
# sliding layers that opt in; the unclamped variant ignores mm_clamp_sw
# entirely, so a clamp without a window is not a reachable configuration.
WINDOW_MODES = [
    (None, 0),
    (SLIDING_WINDOW_LEFT, 0),
    (SLIDING_WINDOW_LEFT, SLIDING_WINDOW_LEFT),
]
WINDOW_IDS = ["full_causal", "window", "window_clamped"]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("window_mode", WINDOW_MODES, ids=WINDOW_IDS)
def test_mm_prefix_mask_mod_matches_dense_reference(case, window_mode):
    sliding_window_left, mm_clamp_sw = window_mode
    _, query_lens, seq_lens, mm_ranges = case
    torch.manual_seed(0)

    total_q = sum(query_lens)
    total_k = sum(seq_lens)
    q = torch.randn(total_q, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    k = torch.randn(total_k, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    v = torch.randn(total_k, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    scale = HEAD_SIZE**-0.5

    actual = _run_kernel(
        q,
        k,
        v,
        query_lens,
        seq_lens,
        mm_ranges,
        sliding_window_left,
        scale,
        mm_clamp_sw=mm_clamp_sw,
    )
    expected = _reference(
        q,
        k,
        v,
        query_lens,
        seq_lens,
        mm_ranges,
        sliding_window_left,
        scale,
        mm_clamp_sw=mm_clamp_sw,
    )

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_clamp_narrows_a_range_wider_than_the_window():
    """Negative control for the ``window_clamped`` parametrization.

    Every case whose ranges fit inside the window is clamp-insensitive, so
    without this the clamped axis could silently be testing nothing. Asserts the
    clamp changes the reference at all, then that the kernel tracks the change.
    """
    _, query_lens, seq_lens, mm_ranges = next(
        c for c in CASES if c[0] == "range_wider_than_window"
    )
    torch.manual_seed(0)
    total = sum(seq_lens)
    q = torch.randn(total, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    scale = HEAD_SIZE**-0.5

    args = (q, k, v, query_lens, seq_lens, mm_ranges, SLIDING_WINDOW_LEFT, scale)
    unclamped = _reference(*args, mm_clamp_sw=0)
    clamped = _reference(*args, mm_clamp_sw=SLIDING_WINDOW_LEFT)
    assert not torch.allclose(unclamped, clamped, atol=2e-2, rtol=2e-2), (
        "case does not actually exercise the clamp branch"
    )

    actual = _run_kernel(*args, mm_clamp_sw=SLIDING_WINDOW_LEFT)
    torch.testing.assert_close(actual.float(), clamped, atol=2e-2, rtol=2e-2)


def _legacy_range_id_mask_mod(sliding_window_left: int | None):
    """The range-id mask_mod as committed at c0d60dce8, for A/B only.

    Reconstructed here rather than imported because the query-range rewrite
    replaced it. Reads ``range_ids[b, q_abs]`` and ``range_ids[b, kv_idx]``,
    i.e. two dependent loads inside the per-element loop.
    """
    import cutlass.cute as cute
    from cutlass import Int32

    # `vllm.vllm_flash_attn.cute` is a build artifact, so isort sorts it as
    # third-party in a clean checkout.
    # isort: split
    from vllm.vllm_flash_attn.cute.utils import scalar_to_ssa

    if sliding_window_left is not None:

        @cute.jit
        def legacy_mask_mod(
            batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
        ):
            ctx_off = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
            q_abs = q_idx + ctx_off
            sw = scalar_to_ssa(Int32(sliding_window_left), Int32)
            keep = (kv_idx <= q_abs) & ((q_abs - kv_idx) < sw)
            range_ids = aux_tensors[0]
            b = batch_idx[0]
            q_range_id = scalar_to_ssa(range_ids[b, q_abs[0]], Int32)
            k_range_id = scalar_to_ssa(range_ids[b, kv_idx[0]], Int32)
            keep = keep | ((q_range_id >= Int32(0)) & (q_range_id == k_range_id))
            return keep

    else:

        @cute.jit
        def legacy_mask_mod(
            batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
        ):
            ctx_off = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
            q_abs = q_idx + ctx_off
            keep = kv_idx <= q_abs
            range_ids = aux_tensors[0]
            b = batch_idx[0]
            q_range_id = scalar_to_ssa(range_ids[b, q_abs[0]], Int32)
            k_range_id = scalar_to_ssa(range_ids[b, kv_idx[0]], Int32)
            keep = keep | ((q_range_id >= Int32(0)) & (q_range_id == k_range_id))
            return keep

    legacy_mask_mod.use_fast_sampling = True
    return legacy_mask_mod


def _legacy_range_ids(mm_ranges, num_seqs, max_seq_len):
    range_ids = torch.full((num_seqs, max_seq_len), -1, dtype=torch.int32)
    for seq_idx in range(num_seqs):
        for range_id, (start, end) in enumerate(sorted(mm_ranges.get(seq_idx, []))):
            range_ids[seq_idx, start : end + 1] = range_id
    return range_ids.to(DEVICE)


# Only the regime the legacy path is defined on: query and key positions
# coincide, and every range fits inside the request's own sequence.
LEGACY_CASES = [c for c in CASES if c[0] in ("single_no_context", "varlen_batch")]


@pytest.mark.parametrize("case", LEGACY_CASES, ids=[c[0] for c in LEGACY_CASES])
@pytest.mark.parametrize("sliding_window_left", [None, SLIDING_WINDOW_LEFT])
def test_matches_legacy_range_id_kernel(case, sliding_window_left):
    """Generalize the model-level bit-exactness to varlen batches.

    The production pooling A/B only ever fed single-segment shapes, so the
    per-request packing offset was never compared between the two paths.
    """
    _, query_lens, seq_lens, mm_ranges = case
    torch.manual_seed(0)

    total = sum(query_lens)
    q = torch.randn(total, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    scale = HEAD_SIZE**-0.5

    new_out = _run_kernel(
        q, k, v, query_lens, seq_lens, mm_ranges, sliding_window_left, scale
    )

    cu = _cu_seqlens(seq_lens).to(DEVICE)
    legacy_out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=legacy_out,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(seq_lens),
        softmax_scale=scale,
        causal=True,
        fa_version=4,
        mask_mod=_legacy_range_id_mask_mod(sliding_window_left),
        aux_tensors=[_legacy_range_ids(mm_ranges, len(seq_lens), max(seq_lens))],
    )

    assert torch.equal(new_out, legacy_out)


def test_ranges_outside_chunk_degrade_to_causal():
    """A range fully behind the scheduled chunk must not widen the mask.

    Under chunked prefill a request's ranges routinely sit entirely in the
    already-computed context. Indexing by query token makes those a no-op; the
    result must be identical to running with no ranges at all.
    """
    torch.manual_seed(0)
    query_lens, seq_lens = [64], [512]
    q = torch.randn(64, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    k = torch.randn(512, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    v = torch.randn(512, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    scale = HEAD_SIZE**-0.5

    # Range (16, 79) is entirely below the chunk start (448), plus one live
    # range so the fill still reports rows.
    mm_ranges = {0: [(16, 79), (448, 511)]}
    actual = _run_kernel(q, k, v, query_lens, seq_lens, mm_ranges, None, scale)
    expected = _reference(q, k, v, query_lens, seq_lens, {0: [(448, 511)]}, None, scale)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
