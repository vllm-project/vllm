# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only used by ROCm"
)


def _on_split_decode_arch() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950

        return bool(_ON_GFX942 or _ON_GFX950)
    except Exception:
        return False


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX950

        return _ON_GFX950
    except ImportError:
        return False


# The flash-decode split-K decode path is only tuned for AMD gfx942/gfx950; other
# architectures take the fallback decode kernel, so its tests are skipped there.
requires_split_decode_arch = pytest.mark.skipif(
    not _on_split_decode_arch(),
    reason="split-K decode kernel is only tuned for AMD gfx942/gfx950",
)
requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(),
    reason="optimized sparse decode partial is gfx950-only",
)

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM


def _ref_global_topk_ragged(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk = topk_indices.reshape(topk_indices.shape[0], -1)
    valid = (topk >= 0) & is_valid_token[:, None]
    lens = valid.sum(dim=1, dtype=torch.int32)
    indptr = torch.zeros(lens.shape[0] + 1, dtype=torch.int32, device=topk.device)
    torch.cumsum(lens, dim=0, out=indptr[1:])

    safe_topk = torch.clamp(topk, min=0)
    block_indices = safe_topk // block_size
    block_offsets = safe_topk % block_size
    req_indices = token_to_req_indices[:, None].expand_as(topk)
    slot_ids = block_table[req_indices, block_indices] * block_size + block_offsets

    offsets = torch.arange(topk.shape[1], dtype=torch.int32, device=topk.device)
    positions = indptr[:-1, None] + offsets[None, :]
    return slot_ids[valid], positions[valid].to(torch.long), indptr, lens


def _ref_sparse_prefill_ragged(
    q: torch.Tensor,
    kv: torch.Tensor,
    rows: list[list[int]],
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    q_f32 = q.float()
    kv_f32 = kv.float()
    out = torch.empty_like(q_f32)

    for query_idx in range(q.shape[0]):
        row_indices = rows[query_idx]
        for head_idx in range(q.shape[1]):
            if row_indices:
                selected_kv = kv_f32[row_indices]
                scores = torch.mv(selected_kv, q_f32[query_idx, head_idx]) * scale
                if attn_sink is not None:
                    scores_with_sink = torch.cat(
                        [scores, attn_sink[head_idx].float().reshape(1)]
                    )
                    probs = torch.softmax(scores_with_sink, dim=0)[:-1]
                else:
                    probs = torch.softmax(scores, dim=0)
                out[query_idx, head_idx] = torch.sum(
                    probs[:, None] * selected_kv, dim=0
                )
            else:
                out[query_idx, head_idx] = 0
    return out.to(torch.bfloat16)


def _pack_fp8_ds_mla_cache(
    kv: torch.Tensor, block_size: int, use_fnuz: bool
) -> torch.Tensor:
    assert kv.shape[-1] == HEAD_DIM
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, 584),
        dtype=torch.uint8,
        device=kv.device,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=kv.device)
    quantize_and_insert_k_cache(
        kv,
        cache,
        slot_mapping,
        block_size=block_size,
        use_fnuz=use_fnuz,
    )
    return cache


def _poison_fp8_ds_mla_cache_row(
    cache: torch.Tensor, block_size: int, slot: int = 0
) -> None:
    flat = cache.flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576
    scale_base = block_base + block_size * 576 + pos * 8
    flat[token_base] = 0x7F
    flat[scale_base : scale_base + 7] = 255
    flat[token_base + NOPE_HEAD_DIM : token_base + 576].view(torch.bfloat16)[0] = float(
        "nan"
    )


def _read_fp8_ds_mla_cache_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
    use_fnuz: bool,
) -> torch.Tensor:
    cache_flat = cache.view(torch.uint8).flatten()
    block_idx = slots // block_size
    pos = slots % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576
    scale_base = block_base + block_size * 576 + pos * 8

    fp8_dtype = torch.float8_e4m3fnuz if use_fnuz else torch.float8_e4m3fn
    nope_offsets = torch.arange(NOPE_HEAD_DIM, device=cache.device)
    nope_u8 = cache_flat[token_base[:, None] + nope_offsets]
    nope = nope_u8.view(fp8_dtype).to(torch.float32)
    scale_offsets = torch.arange(7, device=cache.device)
    scales = torch.exp2(
        cache_flat[scale_base[:, None] + scale_offsets].to(torch.float32) - 127.0
    )
    nope = nope * scales.repeat_interleave(64, dim=1)
    rope_offsets = torch.arange(ROPE_HEAD_DIM * 2, device=cache.device)
    rope_u8 = cache_flat[token_base[:, None] + NOPE_HEAD_DIM + rope_offsets]
    rope = rope_u8.contiguous().view(torch.bfloat16).to(torch.float32)
    return torch.cat([nope, rope], dim=1)


def _ref_sparse_decode_ragged(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_rows: list[list[int]],
    scale: float,
    attn_sink: torch.Tensor | None,
    block_size: int,
    extra_cache: torch.Tensor | None = None,
    extra_rows: list[list[int]] | None = None,
    main_use_fnuz: bool = False,
    extra_use_fnuz: bool = False,
    extra_block_size: int | None = None,
) -> torch.Tensor:
    q_f32 = q.float()
    out = torch.empty_like(q_f32)

    for query_idx in range(q.shape[0]):
        row_kv = []
        if main_rows[query_idx]:
            main_slots = torch.tensor(
                main_rows[query_idx], dtype=torch.int64, device=q.device
            )
            row_kv.append(
                _read_fp8_ds_mla_cache_rows(
                    main_cache, main_slots, block_size, main_use_fnuz
                )
            )
        if extra_cache is not None and extra_rows is not None and extra_rows[query_idx]:
            extra_slots = torch.tensor(
                extra_rows[query_idx], dtype=torch.int64, device=q.device
            )
            row_kv.append(
                _read_fp8_ds_mla_cache_rows(
                    extra_cache,
                    extra_slots,
                    extra_block_size or block_size,
                    extra_use_fnuz,
                )
            )

        if not row_kv:
            out[query_idx] = 0
            continue
        kv = torch.cat(row_kv)
        for head_idx in range(q.shape[1]):
            scores = torch.mv(kv, q_f32[query_idx, head_idx]) * scale
            if attn_sink is not None:
                scores_with_sink = torch.cat(
                    [scores, attn_sink[head_idx].float().reshape(1)]
                )
                probs = torch.softmax(scores_with_sink, dim=0)[:-1]
            else:
                probs = torch.softmax(scores, dim=0)
            out[query_idx, head_idx] = torch.sum(probs[:, None] * kv, dim=0)
    return out.to(torch.bfloat16)


def _ragged_from_rows(
    rows: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten per-query slot lists into ragged (indices, indptr) tensors."""
    flat = [slot for row in rows for slot in row]
    indptr = [0]
    for row in rows:
        indptr.append(indptr[-1] + len(row))
    return (
        torch.tensor(flat, dtype=torch.int32, device=device),
        torch.tensor(indptr, dtype=torch.int32, device=device),
    )


def _rows_from_ragged(indices: torch.Tensor, indptr: torch.Tensor) -> list[list[int]]:
    ends = indptr.cpu().tolist()
    values = indices[: ends[-1]].cpu().tolist()
    return [values[start:end] for start, end in zip(ends, ends[1:])]


def _launch_sparse_decode_reduce(
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    adaptive_splits: bool,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the reduce kernel, with the inverse-RoPE epilogue if given positions.

    Passing ``out_scale`` selects the MXFP8 epilogue (``out_dtype`` e4m3).
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    num_queries, num_splits, num_heads = part_m.shape
    out = torch.empty(
        (num_queries, num_heads, HEAD_DIM),
        dtype=out_dtype,
        device=part_m.device,
    )
    attn_sink = torch.empty(1, dtype=torch.float32, device=part_m.device)
    mod._sparse_attn_decode_reduce_kernel[(num_queries, num_heads)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        positions,
        cos_sin_cache,
        out_scale,
        out.stride(0),
        out.stride(1),
        out_scale.stride(0) if out_scale is not None else 0,
        HEAD_DIM // 32,
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        cos_sin_cache.stride(0) if cos_sin_cache is not None else 0,
        num_heads,
        HAS_ATTN_SINK=False,
        ADAPTIVE_SPLITS=adaptive_splits,
        COMB_DIM=HEAD_DIM,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=1 << (num_splits - 1).bit_length(),
        FUSE_INV_ROPE=positions is not None,
        NOPE=NOPE_HEAD_DIM,
        HALF=ROPE_HEAD_DIM // 2,
        QUANT_OUT=out_scale is not None,
        num_warps=4,
    )
    return out


@torch.inference_mode()
def test_compute_global_topk_ragged_indices_and_indptr() -> None:
    from vllm.models.deepseek_v4.amd.rocm import (
        compute_global_topk_ragged_indices_and_indptr,
    )

    device = torch.device("cuda")
    block_size = 4
    topk_indices = torch.tensor(
        [
            [0, 3, 4, -1],
            [5, 8, -1, -1],
            [2, 7, 9, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    token_to_req_indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ],
        dtype=torch.int32,
        device=device,
    )
    is_valid_token = torch.tensor([True, False, True], dtype=torch.bool, device=device)

    actual_ragged, actual_indptr, actual_lens = (
        compute_global_topk_ragged_indices_and_indptr(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        )
    )
    expected_values, expected_positions, expected_indptr, expected_lens = (
        _ref_global_topk_ragged(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        )
    )

    torch.testing.assert_close(actual_ragged[expected_positions], expected_values)
    torch.testing.assert_close(actual_indptr, expected_indptr)
    torch.testing.assert_close(actual_lens, expected_lens)


@torch.inference_mode()
def test_combine_topk_swa_indices_adds_image_visibility() -> None:
    from vllm.models.deepseek_v4.amd.rocm import combine_topk_swa_indices

    device = torch.device("cuda")
    num_tokens = 8
    topk_indices = torch.full((num_tokens, 1), -1, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    gather_lens = seq_lens.clone()
    left_visible = torch.tensor(
        [0, 0, 0, 1, 2, 3, 4, 0], dtype=torch.int32, device=device
    )
    right_visible = torch.tensor(
        [0, 0, 4, 3, 2, 1, 0, 0], dtype=torch.int32, device=device
    )

    indices, lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=4,
        compress_ratio=1,
        topk=0,
        M=16,
        N=0,
        max_image_tokens=5,
        left_visible=left_visible,
        right_visible=right_visible,
    )

    expected_rows = [
        [0],
        [0, 1],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    for token_idx, expected in enumerate(expected_rows):
        actual = indices[token_idx, : lens[token_idx]].cpu().tolist()
        assert actual == expected


@torch.inference_mode()
def test_combine_topk_swa_indices_apc_hit_inside_image() -> None:
    from vllm.models.deepseek_v4.amd.rocm import combine_topk_swa_indices

    device = torch.device("cuda")
    indices, lens = combine_topk_swa_indices(
        torch.full((2, 1), -1, dtype=torch.int32, device=device),
        torch.tensor([0, 2], dtype=torch.int32, device=device),
        torch.tensor([10], dtype=torch.int32, device=device),
        # Only positions [5, 10) exist in the gathered SWA workspace.
        torch.tensor([5], dtype=torch.int32, device=device),
        window_size=4,
        compress_ratio=1,
        topk=0,
        M=10,
        N=0,
        max_image_tokens=10,
        left_visible=torch.tensor([8, 9], dtype=torch.int32, device=device),
        # Deliberately extends beyond seq_len to exercise the upper clamp too.
        right_visible=torch.tensor([5, 5], dtype=torch.int32, device=device),
    )

    assert lens.cpu().tolist() == [5, 5]
    assert indices[:, :5].cpu().tolist() == [list(range(5)), list(range(5))]


@torch.inference_mode()
def test_combine_topk_swa_indices_keeps_vision_row_width_without_images() -> None:
    from vllm.models.deepseek_v4.amd.rocm import combine_topk_swa_indices

    device = torch.device("cuda")
    indices, lens = combine_topk_swa_indices(
        torch.full((1, 1), -1, dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        window_size=120,
        compress_ratio=1,
        topk=0,
        M=256,
        N=0,
        max_image_tokens=16,
    )

    assert indices.shape == (1, 256)
    assert lens.item() == 1


def test_extra_cache_nan_free_provenance_gate(monkeypatch) -> None:
    from vllm.models.deepseek_v4.amd import rocm as mod

    monkeypatch.setattr(mod, "_ON_GFX950", True)
    assert mod._trust_dsv4_extra_cache_nan_free("fp8_ds_mla", False, True)
    assert not mod._trust_dsv4_extra_cache_nan_free("fp8_ds_mla", True, True)
    assert not mod._trust_dsv4_extra_cache_nan_free("bfloat16", False, True)
    assert not mod._trust_dsv4_extra_cache_nan_free("fp8_ds_mla", False, False)

    monkeypatch.setattr(mod, "_ON_GFX950", False)
    assert not mod._trust_dsv4_extra_cache_nan_free("fp8_ds_mla", False, True)


@torch.inference_mode()
def test_sparse_attn_prefill_ragged_kernel() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(3, 3, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    kv = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    indices = torch.tensor([0, 2, 1, 3, 4], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 5, 5], dtype=torch.int32, device=device)
    attn_sink = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float32, device=device)
    scale = HEAD_DIM**-0.5

    actual = _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
    )
    expected = _ref_sparse_prefill_ragged(
        q, kv, [[0, 2], [1, 3, 4], []], scale, attn_sink
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("num_queries", "on_gfx950", "expected"),
    [(1023, True, False), (1024, True, True), (1024, False, False)],
)
def test_aiter_sparse_prefill_opus_selection(
    num_queries: int, on_gfx950: bool, expected: bool
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _can_use_aiter_sparse_prefill_opus,
    )

    device = torch.device("cuda")
    q = torch.empty(num_queries, 16, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.empty(num_queries, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.empty(16, dtype=torch.float32, device=device)
    output = torch.empty_like(q)

    assert (
        _can_use_aiter_sparse_prefill_opus(
            q, kv, attn_sink, output, on_gfx950=on_gfx950
        )
        is expected
    )


def test_aiter_sparse_prefill_opus_selection_rejects_incompatible_inputs() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _can_use_aiter_sparse_prefill_opus,
    )

    device = torch.device("cuda")
    q = torch.empty(1024, 16, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.empty(1024, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.empty(16, dtype=torch.float32, device=device)
    output = torch.empty_like(q)

    assert _can_use_aiter_sparse_prefill_opus(q, kv, attn_sink, output, on_gfx950=True)
    assert not _can_use_aiter_sparse_prefill_opus(
        q,
        kv,
        attn_sink,
        torch.empty(16, 1024, HEAD_DIM, dtype=q.dtype, device=device).transpose(0, 1),
        on_gfx950=True,
    )
    assert not _can_use_aiter_sparse_prefill_opus(
        q, kv, attn_sink[:-1], output, on_gfx950=True
    )
    assert not _can_use_aiter_sparse_prefill_opus(
        q.cpu(), kv.cpu(), attn_sink.cpu(), output.cpu(), on_gfx950=True
    )


def test_sparse_attn_prefill_aiter_opus_routing(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    q = torch.empty(2, 1, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.empty(2, 1, HEAD_DIM, dtype=torch.bfloat16)
    indices = torch.tensor([[0], [1]], dtype=torch.int32)
    topk_length = torch.ones(2, dtype=torch.int32)
    attn_sink = torch.empty(1, dtype=torch.float32)
    output = torch.empty_like(q)
    opus_calls = 0

    def fake_opus(*args, out):
        nonlocal opus_calls
        opus_calls += 1
        assert args[2].dtype == torch.int32
        assert args[3].dtype == torch.int32
        assert args[5].numel() == 0
        assert torch.count_nonzero(args[6]) == 0
        out.zero_()
        return out

    monkeypatch.setattr(mod, "_can_use_aiter_sparse_prefill_opus", lambda *args: True)
    monkeypatch.setattr(mod, "_get_aiter_sparse_prefill_opus", lambda: fake_opus)
    monkeypatch.setattr(
        mod,
        "build_ragged_indices_from_dense",
        lambda *args, **kwargs: (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        mod,
        "_rocm_sparse_attn_prefill_triton",
        lambda *args, **kwargs: pytest.fail("unexpected dense Triton fallback"),
    )
    monkeypatch.setattr(
        mod,
        "_rocm_sparse_attn_prefill_ragged_triton",
        lambda *args, **kwargs: pytest.fail("unexpected ragged Triton fallback"),
    )

    mod.rocm_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=topk_length,
        scale=HEAD_DIM**-0.5,
        head_dim=HEAD_DIM,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        attn_sink=attn_sink,
        output=output,
    )

    assert opus_calls == 1
    assert torch.count_nonzero(output) == 0


def test_sparse_attn_prefill_preserves_dense_triton_fallback(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    q = torch.empty(2, 1, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.empty(2, 1, HEAD_DIM, dtype=torch.bfloat16)
    indices = torch.tensor([[0], [1]], dtype=torch.int32)
    topk_length = torch.ones(2, dtype=torch.int32)
    attn_sink = torch.empty(1, dtype=torch.float32)
    output = torch.empty_like(q)
    dense_fallback_calls = 0

    def fake_dense_fallback(*args, **kwargs):
        nonlocal dense_fallback_calls
        dense_fallback_calls += 1
        return torch.zeros_like(q)

    monkeypatch.setattr(mod, "_can_use_aiter_sparse_prefill_opus", lambda *args: True)
    monkeypatch.setattr(mod, "_get_aiter_sparse_prefill_opus", lambda: None)
    monkeypatch.setattr(mod, "_rocm_sparse_attn_prefill_triton", fake_dense_fallback)
    monkeypatch.setattr(
        mod,
        "_rocm_sparse_attn_prefill_ragged_triton",
        lambda *args, **kwargs: pytest.fail("unexpected ragged Triton fallback"),
    )

    mod.rocm_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=topk_length,
        scale=HEAD_DIM**-0.5,
        head_dim=HEAD_DIM,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        attn_sink=attn_sink,
        output=output,
    )

    assert dense_fallback_calls == 1
    assert torch.count_nonzero(output) == 0


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_prefill_ragged_aiter_opus(monkeypatch) -> None:
    opus_mod = pytest.importorskip("aiter.ops.pa_sparse_prefill_opus")
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(
        mod, "_get_aiter_sparse_prefill_opus", lambda: opus_mod.pa_sparse_prefill_opus
    )

    device = torch.device("cuda")
    torch.manual_seed(4)
    num_queries = 8
    num_heads = 16
    q = (
        torch.randn(
            num_queries,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    kv = torch.randn(num_queries, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    rows = [list(range(query_idx + 1)) for query_idx in range(num_queries)]
    indices = torch.tensor(
        [index for row in rows for index in row], dtype=torch.int32, device=device
    )
    indptr = torch.tensor(
        [0]
        + [
            sum(len(row) for row in rows[: query_idx + 1])
            for query_idx in range(num_queries)
        ],
        dtype=torch.int32,
        device=device,
    )
    attn_sink = torch.linspace(
        -0.25, 0.25, num_heads, dtype=torch.float32, device=device
    )
    output = torch.empty_like(q)
    scale = HEAD_DIM**-0.5

    assert mod._rocm_sparse_attn_prefill_ragged_aiter_opus(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=attn_sink,
        output=output,
    )
    expected = _ref_sparse_prefill_ragged(q, kv, rows, scale, attn_sink)

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_sparse_attn_decode_ragged_kernel() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
    )

    device = torch.device("cuda")
    torch.manual_seed(1)
    block_size = 4
    main_use_fnuz = current_platform.is_fp8_fnuz()
    q = torch.randn(2, 3, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_kv = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    extra_kv = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, use_fnuz=main_use_fnuz)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, use_fnuz=False)
    main_indices = torch.tensor([0, 2, 4, 1], dtype=torch.int32, device=device)
    main_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    extra_indices = torch.tensor([1, 3, 0], dtype=torch.int32, device=device)
    extra_indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    attn_sink = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
    scale = HEAD_DIM**-0.5

    out = torch.empty_like(q)
    actual = _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        out=out,
    )
    expected = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=[[0, 2], [4, 1]],
        scale=scale,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=[[1], [3, 0]],
        main_use_fnuz=main_use_fnuz,
    )

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_decode_scrubs_untrusted_cache_by_default() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
    )

    device = torch.device("cuda")
    block_size = 4
    main_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8, device=device)
    extra_cache = torch.zeros_like(main_cache)
    _poison_fp8_ds_mla_cache_row(main_cache, block_size)
    _poison_fp8_ds_mla_cache_row(extra_cache, block_size)
    indices = torch.zeros(1, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)

    actual = _rocm_sparse_attn_decode_ragged_triton(
        q=torch.ones(1, 1, HEAD_DIM, dtype=torch.bfloat16, device=device),
        main_cache=main_cache,
        main_indices=indices,
        main_indptr=indptr,
        scale=HEAD_DIM**-0.5,
        attn_sink=None,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=indices,
        extra_indptr=indptr,
    )

    assert not torch.isnan(actual).any()
    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("on_gfx950", [False, True])
@torch.inference_mode()
def test_rocm_ragged_graph_buffer_view_tracks_source_width(
    monkeypatch, on_gfx950: bool
) -> None:
    from vllm.models.deepseek_v4.amd import rocm as rocm_mod

    monkeypatch.setattr(rocm_mod, "_ON_GFX950", on_gfx950)

    indices_buffer = torch.full((16,), -1, dtype=torch.int32)
    indptr_buffer = torch.full((3,), -1, dtype=torch.int32)
    first_indices = torch.tensor([3, 5, 7], dtype=torch.int32)
    first_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
    first_view, first_indptr_view = rocm_mod._copy_ragged_to_graph_buffers(
        first_indices,
        first_indptr,
        indices_buffer,
        indptr_buffer,
        num_rows=2,
        max_entries_per_row=8,
    )

    second_indices = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
    second_indptr = torch.tensor([0, 2, 6], dtype=torch.int32)
    second_view, second_indptr_view = rocm_mod._copy_ragged_to_graph_buffers(
        second_indices,
        second_indptr,
        indices_buffer,
        indptr_buffer,
        num_rows=2,
        max_entries_per_row=8,
    )

    expected_first_entries = (
        first_indices.numel() if on_gfx950 else indices_buffer.numel()
    )
    expected_second_entries = (
        second_indices.numel() if on_gfx950 else indices_buffer.numel()
    )
    assert first_view.numel() == expected_first_entries
    assert second_view.numel() == expected_second_entries
    assert first_view.data_ptr() == second_view.data_ptr() == indices_buffer.data_ptr()
    assert first_indptr_view.data_ptr() == second_indptr_view.data_ptr()
    assert torch.equal(second_view[: second_indices.numel()], second_indices)
    assert torch.equal(second_indptr_view, second_indptr)


def test_rocm_capture_metadata_sets_adaptive_marker(monkeypatch) -> None:
    from vllm.models.deepseek_v4.amd import rocm as rocm_mod
    from vllm.models.deepseek_v4.sparse_mla import (
        DeepseekV4SparseMLAMetadataBuilder,
    )

    metadata = SimpleNamespace(for_cudagraph_capture=False)
    monkeypatch.setattr(
        DeepseekV4SparseMLAMetadataBuilder,
        "build_for_cudagraph_capture",
        lambda *_: metadata,
    )
    builder = object.__new__(rocm_mod.DeepseekV4ROCMAiterMLASparseMetadataBuilder)

    actual = builder.build_for_cudagraph_capture(SimpleNamespace())

    assert actual is metadata
    assert actual.for_cudagraph_capture is _on_gfx950()


@requires_split_decode_arch
@torch.inference_mode()
def test_decode_num_splits_heuristic(monkeypatch) -> None:
    """Split-count heuristic added with the flash-decode split-K decode path."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    # Pin the CU count so the heuristic is deterministic off-device.
    monkeypatch.setattr(mod, "_decode_cu_count", lambda: 256)

    # A batch that already fills the device should not be split.
    assert mod._decode_num_splits(256, 1, avg_main_len=128.0, avg_extra_len=0.0) == 1
    # A tiny batch on a large device should split to add parallelism.
    assert mod._decode_num_splits(2, 1, avg_main_len=256.0, avg_extra_len=0.0) > 1

    # The shared gfx942 selector retains its original 16-split ceiling.
    assert mod._decode_num_splits(1, 1, 128.0, 8192.0) == 16

    # The chosen count always stays within the searched [1, 16] range, and a
    # zero-length workload never splits (no work to parallelize).
    for num_queries in (1, 4, 24, 224, 1024):
        splits = mod._decode_num_splits(
            num_queries, 1, avg_main_len=512.0, avg_extra_len=128.0
        )
        assert 1 <= splits <= 16
    assert mod._decode_num_splits(2, 1, avg_main_len=0.0, avg_extra_len=0.0) >= 1


@torch.inference_mode()
def test_decode_num_splits_gfx950(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod, "_decode_cu_count", lambda: 256)
    assert mod._decode_gfx950_num_splits(1, 1, 128, 8192) == 32
    assert mod._decode_gfx950_num_splits(17, 1, 128, 32) == 4
    assert mod._decode_gfx950_num_splits(512, 1, 128, 7812) == 1


@requires_split_decode_arch
@pytest.mark.parametrize("num_splits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("with_extra", [True, False])
@pytest.mark.parametrize("with_sink", [True, False])
@torch.inference_mode()
def test_sparse_attn_decode_split_k_kernel(
    monkeypatch, num_splits: int, with_extra: bool, with_sink: bool
) -> None:
    """Flash-decode split-K decode path (partial + reduce kernels).

    This path is the gfx942/gfx950 production path, so the test only runs on
    those architectures. The split count is pinned so the partial/reduce kernels are
    exercised across split counts. ``num_splits=8`` drives splits past the
    shortest segment length, covering the empty-split edge case handled by the
    reduce kernel.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")
    torch.manual_seed(7)
    block_size = 4
    num_heads = 3
    main_use_fnuz = current_platform.is_fp8_fnuz()

    main_rows = [[0, 2, 4, 6, 1, 3, 7, 5], [4, 1, 6, 0, 2]]
    num_queries = len(main_rows)
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = torch.randn(8, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, use_fnuz=main_use_fnuz)
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)

    extra_rows: list[list[int]] | None = None
    extra_cache: torch.Tensor | None = None
    extra_indices: torch.Tensor | None = None
    extra_indptr: torch.Tensor | None = None
    if with_extra:
        rows = [[1, 3, 0, 5, 2, 4], [3, 0, 6]]
        extra_kv = torch.randn(7, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
        extra_rows = rows
        extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, use_fnuz=False)
        extra_indices, extra_indptr = _ragged_from_rows(rows, device)

    attn_sink = (
        torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    # Pin the split count so each parametrized value is exercised deterministically.
    split_fn = "_decode_gfx950_num_splits" if _on_gfx950() else "_decode_num_splits"
    other_split_fn = (
        "_decode_num_splits" if _on_gfx950() else "_decode_gfx950_num_splits"
    )
    monkeypatch.setattr(mod, split_fn, lambda *args, **kwargs: num_splits)
    monkeypatch.setattr(
        mod, other_split_fn, lambda *args, **kwargs: pytest.fail("wrong selector")
    )

    actual = mod._rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )
    expected = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=main_rows,
        scale=scale,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=extra_rows,
        main_use_fnuz=main_use_fnuz,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_decode_gfx950_adaptive_reduce_ignores_stale_scratch() -> None:
    device = torch.device("cuda")
    part_m = torch.full(
        (1, 8, 1),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=device,
    )
    part_l = torch.zeros_like(part_m)
    part_acc = torch.full(
        (1, 8, 1, HEAD_DIM),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    part_m[:, :2] = 0
    part_l[:, :2] = 1
    part_acc[:, 0] = 1
    part_acc[:, 1] = 3

    actual = _launch_sparse_decode_reduce(part_m, part_l, part_acc, True)

    assert torch.isfinite(actual).all()
    assert torch.equal(actual, torch.full_like(actual, 2))


@requires_gfx950
@pytest.mark.parametrize("adaptive_splits", [False, True])
@torch.inference_mode()
def test_sparse_attn_decode_reduce_inverse_rope_epilogue(adaptive_splits: bool) -> None:
    """FUSE_INV_ROPE must land where the standalone rotation would.

    The epilogue rotates a whole [H, nope + rope] row with one expression by
    feeding cos=1/sin=0 to the NoPE lanes, which only holds if the pair split
    lands on ``nope_head_dim // 2``; the NoPE lanes therefore have to come
    back untouched, not merely close.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

    device = torch.device("cuda")
    torch.manual_seed(7)
    num_queries, num_splits, num_heads, max_pos = 5, 4, 8, 64
    shape = (num_queries, num_splits, num_heads)
    part_m = torch.randn(shape, dtype=torch.float32, device=device)
    part_l = torch.rand(shape, dtype=torch.float32, device=device) + 0.5
    part_acc = torch.randn((*shape, HEAD_DIM), dtype=torch.float32, device=device)
    positions = torch.randint(
        0, max_pos, (num_queries,), dtype=torch.int64, device=device
    )
    angle = torch.randn(max_pos, ROPE_HEAD_DIM // 2, device=device)
    cos_sin_cache = torch.cat((angle.cos(), angle.sin()), dim=-1).contiguous()

    unfused = _launch_sparse_decode_reduce(part_m, part_l, part_acc, adaptive_splits)
    expected = _fused_inverse_rope_gptj(
        unfused, positions, cos_sin_cache, ROPE_HEAD_DIM
    )
    actual = _launch_sparse_decode_reduce(
        part_m, part_l, part_acc, adaptive_splits, positions, cos_sin_cache
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(actual[..., :NOPE_HEAD_DIM], unfused[..., :NOPE_HEAD_DIM])
    # An epilogue that quietly did nothing would satisfy everything above.
    assert not torch.equal(actual[..., NOPE_HEAD_DIM:], unfused[..., NOPE_HEAD_DIM:])


def _mxfp8_dequant(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    blocks = data.float().view(*data.shape[:-1], -1, 32)
    return (blocks * torch.exp2(scale.float() - 127.0)[..., None]).view(data.shape)


def _random_cos_sin_cache(max_pos: int, device: torch.device) -> torch.Tensor:
    angle = torch.randn(max_pos, ROPE_HEAD_DIM // 2, device=device)
    return torch.cat((angle.cos(), angle.sin()), dim=-1).contiguous()


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_decode_reduce_mxfp8_epilogue() -> None:
    """QUANT_OUT must equal MXFP8-quantizing the reduce's fp32 output."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )

    device = torch.device("cuda")
    torch.manual_seed(7)
    num_queries, num_splits, num_heads = 5, 4, 8
    shape = (num_queries, num_splits, num_heads)
    part_m = torch.randn(shape, device=device)
    part_l = torch.rand(shape, device=device) + 0.5
    part_acc = torch.randn((*shape, HEAD_DIM), device=device)
    positions = torch.randint(0, 64, (num_queries,), device=device)
    cos_sin_cache = _random_cos_sin_cache(64, device)
    args = (part_m, part_l, part_acc, False, positions, cos_sin_cache)

    rotated = _launch_sparse_decode_reduce(*args, out_dtype=torch.float32)
    expected_data, expected_scale = _mxfp8_e4m3_quantize_torch(
        rotated.view(num_queries, -1)
    )
    scale = torch.empty_like(expected_scale)
    data = _launch_sparse_decode_reduce(
        *args, out_dtype=torch.float8_e4m3fn, out_scale=scale
    )

    assert torch.equal(scale, expected_scale)
    assert torch.equal(
        data.view(num_queries, -1).view(torch.uint8), expected_data.view(torch.uint8)
    )


@requires_gfx950
@torch.inference_mode()
def test_inverse_rope_mxfp8_rows() -> None:
    """Prefill rows get the same inverse RoPE + MXFP8 as the decode epilogue."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_inverse_rope_mxfp8_rows,
    )

    device = torch.device("cuda")
    torch.manual_seed(5)
    num_tokens, num_heads = 7, 4
    o = torch.randn(num_tokens, num_heads, HEAD_DIM, device=device).bfloat16()
    positions = torch.randint(0, 64, (num_tokens,), device=device)
    cos_sin_cache = _random_cos_sin_cache(64, device)

    ref = o.float()
    cos, sin = cos_sin_cache[positions, None].chunk(2, dim=-1)
    even, odd = ref[..., NOPE_HEAD_DIM::2].clone(), ref[..., NOPE_HEAD_DIM + 1 :: 2]
    ref[..., NOPE_HEAD_DIM::2] = even * cos + odd * sin
    ref[..., NOPE_HEAD_DIM + 1 :: 2] = odd * cos - even * sin
    ref = ref.view(num_tokens, -1)
    _, expected_scale = _mxfp8_e4m3_quantize_torch(ref)

    data = torch.empty_like(ref, dtype=torch.float8_e4m3fn)
    scale = torch.empty_like(expected_scale)
    rocm_inverse_rope_mxfp8_rows(
        o, positions, cos_sin_cache, ROPE_HEAD_DIM, data, scale
    )

    assert torch.equal(scale, expected_scale)
    # Within half an e4m3 step at the top of each block's range.
    half_step = 16.0 * torch.exp2(scale.float() - 127.0).repeat_interleave(32, -1)
    assert bool(((_mxfp8_dequant(data, scale) - ref).abs() <= half_step).all())


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_decode_mxfp8_output() -> None:
    """``out_mxfp8`` routes the real decode through the MXFP8 epilogue."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
    )

    device = torch.device("cuda")
    torch.manual_seed(1)
    num_heads = 16
    q = torch.randn(2, num_heads, HEAD_DIM, device=device).bfloat16() * 0.125
    kv = torch.randn(6, HEAD_DIM, device=device).bfloat16() * 0.125
    decode = functools.partial(
        _rocm_sparse_attn_decode_ragged_triton,
        q=q,
        main_cache=_pack_fp8_ds_mla_cache(kv, 4, current_platform.is_fp8_fnuz()),
        main_indices=torch.tensor([0, 2, 4, 1], dtype=torch.int32, device=device),
        main_indptr=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
        scale=HEAD_DIM**-0.5,
        attn_sink=None,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        inv_rope_positions=torch.tensor([3, 9], device=device),
        inv_rope_cos_sin_cache=_random_cos_sin_cache(16, device),
    )

    expected = decode().float().view(2, -1)
    data = torch.empty(
        2, num_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    scale = torch.empty(2, num_heads * HEAD_DIM // 32, dtype=torch.uint8, device=device)
    decode(out_mxfp8=(data, scale))

    torch.testing.assert_close(
        _mxfp8_dequant(data, scale), expected, atol=1e-3, rtol=2**-4
    )


@requires_gfx950
@pytest.mark.parametrize(
    "num_tokens, n_groups",
    # One case per tile tier of _mxfp8_wo_a_bmm_config, with partial M tiles.
    [
        (1, 4),
        (20, 4),
        (48, 4),
        (77, 4),
        (130, 4),
        (300, 4),
        (700, 4),
        (1000, 4),
        (1100, 8),
    ],
)
@torch.inference_mode()
def test_rocm_mxfp8_wo_a_bmm(num_tokens: int, n_groups: int) -> None:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_mxfp8_wo_a_bmm

    device = torch.device("cuda")
    torch.manual_seed(3)
    o_lora_rank, group_dim = 128, 2048
    a, a_scale = _mxfp8_e4m3_quantize_torch(
        torch.randn(num_tokens, n_groups * group_dim, device=device)
    )
    w, w_scale = _mxfp8_e4m3_quantize_torch(
        torch.randn(n_groups * o_lora_rank, group_dim, device=device)
    )
    wo_a = SimpleNamespace(weight=w, weight_scale=w_scale)

    out = rocm_mxfp8_wo_a_bmm(a, a_scale, wo_a, n_groups, o_lora_rank)

    expected = torch.einsum(
        "tgd,grd->tgr",
        _mxfp8_dequant(a, a_scale).view(num_tokens, n_groups, group_dim),
        _mxfp8_dequant(w, w_scale).view(n_groups, o_lora_rank, group_dim),
    )
    torch.testing.assert_close(out.float(), expected.flatten(1), atol=5e-2, rtol=1e-2)


@requires_gfx950
@pytest.mark.parametrize("extra_len", [0, 1, 31, 32, 33, 63, 64, 65])
@torch.inference_mode()
def test_sparse_attn_decode_gfx950_outer64_boundaries(
    monkeypatch, extra_len: int
) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")
    torch.manual_seed(13)
    block_size = 4
    num_heads = 16
    num_extra_rows = 80
    q = torch.randn(2, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    q *= 0.125
    main_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8, device=device)
    main_indices = torch.empty(0, dtype=torch.int32, device=device)
    main_indptr = torch.zeros(3, dtype=torch.int32, device=device)
    extra_cache = _pack_fp8_ds_mla_cache(
        torch.randn(num_extra_rows, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.125,
        block_size,
        use_fnuz=False,
    )
    _poison_fp8_ds_mla_cache_row(extra_cache, block_size)

    raw_row = list(range(1, extra_len + 1))
    if extra_len > 3:
        raw_row[3] = -1
    if extra_len > 40:
        raw_row[40] = num_extra_rows
    if extra_len > 64:
        raw_row[64] = num_extra_rows + 1024
    extra_indices, extra_indptr = _ragged_from_rows([raw_row, []], device)
    valid_row = [slot for slot in raw_row if 0 <= slot < num_extra_rows]
    attn_sink = torch.linspace(-0.1, 0.1, num_heads, dtype=torch.float32, device=device)

    monkeypatch.setattr(mod, "_decode_gfx950_num_splits", lambda *args: 1)
    actual = mod._rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=HEAD_DIM**-0.5,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )
    expected = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=[[], []],
        scale=HEAD_DIM**-0.5,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=[valid_row, []],
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_decode_gfx950_graph_replay(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")
    torch.manual_seed(17)
    block_size = 64
    num_queries = 16
    num_heads = 16
    num_splits = 8
    extra_per_query = 65 * num_splits
    max_extra_per_query = 8192
    q = (
        torch.randn(
            num_queries,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    main_cache = _pack_fp8_ds_mla_cache(
        torch.randn(num_queries, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125,
        block_size,
        use_fnuz=False,
    )
    extra_cache = _pack_fp8_ds_mla_cache(
        torch.randn(
            num_queries * extra_per_query,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125,
        block_size,
        use_fnuz=False,
    )
    main_rows = [[query_idx] for query_idx in range(num_queries)]
    extra_rows = [
        list(range(query_idx * extra_per_query, (query_idx + 1) * extra_per_query))
        for query_idx in range(num_queries)
    ]
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)
    short_extra_rows = [row[:64] for row in extra_rows]
    long_indices, long_indptr = _ragged_from_rows(extra_rows, device)
    short_indices, short_indptr = _ragged_from_rows(short_extra_rows, device)
    extra_indices = torch.full(
        (num_queries * max_extra_per_query,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    extra_indices[: long_indices.numel()].copy_(long_indices)
    extra_indptr = long_indptr.clone()
    extra_indices_ptr = extra_indices.data_ptr()
    attn_sink = torch.linspace(-0.1, 0.1, num_heads, dtype=torch.float32, device=device)
    out = torch.empty_like(q)

    monkeypatch.setattr(mod, "_decode_gfx950_num_splits", lambda *args: num_splits)

    def run_decode() -> torch.Tensor:
        return mod._rocm_sparse_attn_decode_ragged_triton(
            q=q,
            main_cache=main_cache,
            main_indices=main_indices,
            main_indptr=main_indptr,
            scale=HEAD_DIM**-0.5,
            attn_sink=attn_sink,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_indptr=extra_indptr,
            out=out,
            extra_cache_nan_free=True,
            adaptive_splits=True,
        )

    run_decode()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = run_decode()
    torch.accelerator.synchronize()
    captured_long = out.clone()
    expected_long = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=main_rows,
        scale=HEAD_DIM**-0.5,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=extra_rows,
    )
    torch.testing.assert_close(captured_long, expected_long, atol=2e-2, rtol=2e-2)

    extra_indices[: short_indices.numel()].copy_(short_indices)
    extra_indptr.copy_(short_indptr)
    graph.replay()
    torch.accelerator.synchronize()
    short_out = out.clone()
    expected_short = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=main_rows,
        scale=HEAD_DIM**-0.5,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=short_extra_rows,
    )

    assert captured_out.data_ptr() == out.data_ptr()
    assert extra_indices.data_ptr() == extra_indices_ptr
    assert extra_indices.numel() == num_queries * max_extra_per_query
    assert not torch.equal(short_out, captured_long)
    torch.testing.assert_close(short_out, expected_short, atol=2e-2, rtol=2e-2)

    extra_indices[: long_indices.numel()].copy_(long_indices)
    extra_indptr.copy_(long_indptr)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, expected_long, atol=2e-2, rtol=2e-2)


@requires_gfx950
@torch.inference_mode()
def test_dsv4_adaptive_mla_swa_metadata_graph_replay(monkeypatch) -> None:
    """Replay the production ROCm decode path after device-only reallocation."""
    from tests.v1.attention.utils import create_vllm_config
    from vllm.models.deepseek_v4.amd.rocm import (
        DeepseekV4ROCMAiterMLASparseMetadataBuilder,
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )
    from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

    device = torch.device("cuda")
    num_reqs = 3
    upper_query_len = 8
    graph_tokens = num_reqs * upper_query_len
    block_size = 256
    compressed_block_size = block_size // 128
    window_size = 32

    vllm_config = create_vllm_config(
        model_name="facebook/opt-125m",
        max_model_len=1024,
        block_size=block_size,
        max_num_seqs=num_reqs,
        max_num_batched_tokens=graph_tokens,
        hf_config_override={
            "compress_ratios": [128],
            "index_topk": 2048,
            "sliding_window": window_size,
        },
    )
    vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens=upper_query_len - 1,
        parallel_drafting=False,
        enable_adaptive_verification=True,
        use_dspark=lambda: True,
    )
    mla_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        tokens_per_state=128,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        sliding_window=window_size,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    mla_builder = DeepseekV4ROCMAiterMLASparseMetadataBuilder(
        mla_spec, ["c128a"], vllm_config, device
    )
    swa_builder = DeepseekV4ROCMAiterSparseSWAMetadataBuilder(
        swa_spec, ["c128a"], vllm_config, device
    )
    assert (
        mla_builder.get_cudagraph_support(vllm_config, mla_spec)
        == AttentionCGSupport.ALWAYS
    )
    assert (
        swa_builder.get_cudagraph_support(vllm_config, swa_spec)
        == AttentionCGSupport.ALWAYS
    )

    seq_lens = torch.tensor([520, 528, 536], dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    cpu_query_start_loc = torch.arange(
        0,
        graph_tokens + 1,
        upper_query_len,
        dtype=torch.int32,
    )
    max_blocks = (int(seq_lens_cpu.max()) + block_size - 1) // block_size
    block_table = torch.arange(
        num_reqs * max_blocks, dtype=torch.int32, device=device
    ).view(num_reqs, max_blocks)

    def build_metadata(query_lens: list[int]):
        query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32, device=device)
        query_start_loc = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                query_lens_tensor.cumsum(0),
            ]
        )
        active_tokens = sum(query_lens)
        positions = torch.zeros(graph_tokens, dtype=torch.int64, device=device)
        position_rows = [
            torch.arange(
                seq_len - query_len,
                seq_len,
                dtype=torch.int64,
                device=device,
            )
            for seq_len, query_len in zip(seq_lens_cpu.tolist(), query_lens)
        ]
        positions[:active_tokens] = torch.cat(position_rows)
        slot_mapping = torch.full((graph_tokens,), -1, dtype=torch.int64, device=device)
        slot_mapping[:active_tokens] = torch.arange(
            active_tokens, dtype=torch.int64, device=device
        )
        common = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=cpu_query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=graph_tokens,
            max_query_len=upper_query_len,
            max_seq_len=int(seq_lens_cpu.max()),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            positions=positions,
            causal=True,
        )
        return (
            mla_builder.build_for_cudagraph_capture(common),
            swa_builder.build_for_cudagraph_capture(common),
        )

    mla_metadata, swa_metadata = build_metadata([8, 8, 8])
    assert mla_metadata.for_cudagraph_capture
    metadata_ptrs = (
        mla_metadata.c128a_decode_topk_ragged_indices.data_ptr(),
        mla_metadata.c128a_decode_topk_ragged_indptr.data_ptr(),
        swa_metadata.decode_swa_ragged_indices.data_ptr(),
        swa_metadata.decode_swa_ragged_indptr.data_ptr(),
    )

    torch.manual_seed(19)
    num_heads = 16
    q = (
        torch.randn(
            graph_tokens,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    num_cache_blocks = num_reqs * max_blocks
    swa_cache = _pack_fp8_ds_mla_cache(
        torch.randn(
            num_cache_blocks * block_size,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125,
        block_size,
        use_fnuz=False,
    )
    compressed_cache = _pack_fp8_ds_mla_cache(
        torch.randn(
            num_cache_blocks * compressed_block_size,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125,
        compressed_block_size,
        use_fnuz=False,
    )
    attn_sink = torch.linspace(-0.1, 0.1, num_heads, dtype=torch.float32, device=device)
    out = torch.empty_like(q)
    monkeypatch.setattr(mod, "_decode_gfx950_num_splits", lambda *args: 1)

    def run_decode(mla_md, swa_md) -> None:
        mod.rocm_sparse_attn_decode(
            q=q,
            kv_cache=compressed_cache,
            swa_k_cache=swa_cache,
            swa_only=False,
            topk_indices=mla_md.c128a_global_decode_topk_indices,
            topk_lens=mla_md.c128a_decode_topk_lens,
            swa_indices=swa_md.decode_swa_indices,
            swa_lens=swa_md.decode_swa_lens,
            swa_ragged_indices=swa_md.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_md.decode_swa_ragged_indptr,
            topk_ragged_indices=mla_md.c128a_decode_topk_ragged_indices,
            topk_ragged_indptr=mla_md.c128a_decode_topk_ragged_indptr,
            attn_sink=attn_sink,
            scale=HEAD_DIM**-0.5,
            head_dim=HEAD_DIM,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            output=out,
            extra_cache_nan_free=True,
            adaptive_splits=True,
        )

    def reference(mla_md, swa_md) -> torch.Tensor:
        return _ref_sparse_decode_ragged(
            q=q,
            main_cache=swa_cache,
            main_rows=_rows_from_ragged(
                swa_md.decode_swa_ragged_indices,
                swa_md.decode_swa_ragged_indptr,
            ),
            scale=HEAD_DIM**-0.5,
            attn_sink=attn_sink,
            block_size=block_size,
            extra_cache=compressed_cache,
            extra_rows=_rows_from_ragged(
                mla_md.c128a_decode_topk_ragged_indices,
                mla_md.c128a_decode_topk_ragged_indptr,
            ),
            extra_block_size=compressed_block_size,
        )

    run_decode(mla_metadata, swa_metadata)
    torch.accelerator.synchronize()
    expected_full = reference(mla_metadata, swa_metadata)
    torch.testing.assert_close(out, expected_full, atol=2e-2, rtol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_decode(mla_metadata, swa_metadata)
    torch.accelerator.synchronize()
    captured_full = out.clone()

    reallocated_mla, reallocated_swa = build_metadata([3, 8, 5])
    assert metadata_ptrs == (
        reallocated_mla.c128a_decode_topk_ragged_indices.data_ptr(),
        reallocated_mla.c128a_decode_topk_ragged_indptr.data_ptr(),
        reallocated_swa.decode_swa_ragged_indices.data_ptr(),
        reallocated_swa.decode_swa_ragged_indptr.data_ptr(),
    )
    expected_reallocated = reference(reallocated_mla, reallocated_swa)
    run_decode(reallocated_mla, reallocated_swa)
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, expected_reallocated, atol=2e-2, rtol=2e-2)
    assert not torch.equal(captured_full, expected_reallocated)

    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, expected_reallocated, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# o-projection: fused inverse-RoPE + cached bf16 wo_a (rocm_inv_rope_einsum)
# ---------------------------------------------------------------------------


# Cache rows = max_position_embeddings * scaling_factor.
_ROTARY_MAX_POS = 1024
_ROTARY_SCALING_FACTOR = 4.0
_ROTARY_CACHE_LEN = int(_ROTARY_MAX_POS * _ROTARY_SCALING_FACTOR)


def _make_dsv4_rotary(device: torch.device):
    """The official DSv4 rotary embedding, sized down for unit tests."""
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekV4ScalingRotaryEmbedding,
    )

    # The model loader constructs layers under a default-device context;
    # mirror that so the fp32 cos_sin_cache lands on the GPU.
    with torch.device(device):
        rotary_emb = DeepseekV4ScalingRotaryEmbedding(
            head_size=ROPE_HEAD_DIM,
            rotary_dim=ROPE_HEAD_DIM,
            max_position_embeddings=_ROTARY_MAX_POS,
            base=10000,
            is_neox_style=False,
            scaling_factor=_ROTARY_SCALING_FACTOR,
            dtype=torch.bfloat16,
            mscale=1.0,
            mscale_all_dim=1.0,
        )
    rotary_emb = rotary_emb.to(device)
    assert rotary_emb.cos_sin_cache.shape == (_ROTARY_CACHE_LEN, ROPE_HEAD_DIM)
    return rotary_emb


def _inv_rope_via_rotary_native(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Reference: the official ``forward_native(inverse=True)`` path."""
    expected, _ = rotary_emb.forward_native(positions, o.clone(), None, inverse=True)
    return expected.to(torch.bfloat16)


class _FakeWoA(torch.nn.Module):
    """Stand-in for the wo_a linear layer holding the (optionally fp8) weight."""

    def __init__(
        self, weight: torch.Tensor, weight_scale_inv: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.weight = weight
        if weight_scale_inv is not None:
            self.weight_scale_inv = weight_scale_inv


@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_fused_inverse_rope_gptj_matches_rotary_native(
    num_tokens: int, num_heads: int, pos_dtype: torch.dtype, default_vllm_config
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

    device = torch.device("cuda")
    torch.manual_seed(0)
    rotary_emb = _make_dsv4_rotary(device)
    o = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(
        0, _ROTARY_CACHE_LEN, (num_tokens,), dtype=pos_dtype, device=device
    )

    actual = _fused_inverse_rope_gptj(
        o, positions, rotary_emb.cos_sin_cache, ROPE_HEAD_DIM
    )
    expected = _inv_rope_via_rotary_native(rotary_emb, o, positions)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == o.shape
    # NoPE lanes are a pure bf16 passthrough -> must be bit-exact.
    assert torch.equal(actual[..., :NOPE_HEAD_DIM], expected[..., :NOPE_HEAD_DIM])
    # RoPE lanes: tolerate at most ~1 bf16 ulp from fp32 fma ordering.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_fused_inverse_rope_gptj_empty(default_vllm_config) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

    device = torch.device("cuda")
    rotary_emb = _make_dsv4_rotary(device)
    o = torch.empty(0, 8, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.empty(0, dtype=torch.int32, device=device)

    out = _fused_inverse_rope_gptj(
        o, positions, rotary_emb.cos_sin_cache, ROPE_HEAD_DIM
    )
    assert out.shape == (0, 8, HEAD_DIM)
    assert out.dtype == torch.bfloat16


@torch.inference_mode()
def test_rocm_inv_rope_einsum_matches_rotary_native(default_vllm_config) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum

    device = torch.device("cuda")
    torch.manual_seed(2)
    num_tokens, num_heads = 5, 8
    n_local_groups = num_heads
    o_lora_rank = 16
    hidden_dim = num_heads * HEAD_DIM // n_local_groups  # 512

    rotary_emb = _make_dsv4_rotary(device)
    o = (
        torch.randn(
            num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    positions = torch.randint(
        0, _ROTARY_CACHE_LEN, (num_tokens,), dtype=torch.int32, device=device
    )
    weight = (
        torch.randn(n_local_groups * o_lora_rank, hidden_dim, device=device) * 0.125
    ).to(torch.bfloat16)
    wo_a = _FakeWoA(weight)

    actual = rocm_inv_rope_einsum(
        rotary_emb, o, positions, ROPE_HEAD_DIM, n_local_groups, o_lora_rank, wo_a
    )

    o_ref = _inv_rope_via_rotary_native(rotary_emb, o, positions)
    o_ref = o_ref.view(num_tokens, n_local_groups, -1)
    wo_a_ref = weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)
    expected = torch.einsum("tgd,grd->tgr", o_ref, wo_a_ref)

    assert actual.shape == (num_tokens, n_local_groups, o_lora_rank)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_get_cached_wo_a_bf16_plain_caches() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    device = torch.device("cuda")
    torch.manual_seed(4)
    n_local_groups, o_lora_rank, hidden_dim = 2, 4, 8
    weight = torch.randn(
        n_local_groups * o_lora_rank, hidden_dim, dtype=torch.bfloat16, device=device
    )
    wo_a = _FakeWoA(weight)

    out1 = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim)
    expected = weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)
    assert out1.shape == (n_local_groups, o_lora_rank, hidden_dim)
    torch.testing.assert_close(out1, expected, atol=0, rtol=0)
    assert hasattr(wo_a, "_dsv4_wo_a_bf16")

    # Mutate the source weight: the cached tensor must be returned unchanged
    # (proving the dequant is not recomputed per call).
    wo_a.weight.zero_()
    out2 = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim)
    assert out2 is out1
    torch.testing.assert_close(out2, expected, atol=0, rtol=0)


@pytest.mark.parametrize("scale_attr", ["weight_scale", "weight_scale_inv"])
@torch.inference_mode()
def test_get_cached_wo_a_bf16_dequantized_ignores_retained_scale(
    scale_attr: str,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    device = torch.device("cuda")
    torch.manual_seed(6)
    n_local_groups, o_lora_rank, hidden_dim = 2, 4, 8
    weight = torch.randn(
        n_local_groups * o_lora_rank, hidden_dim, dtype=torch.bfloat16, device=device
    )
    retained_scale = torch.full(
        (n_local_groups, 1), 0.25, dtype=torch.float32, device=device
    )
    wo_a = _FakeWoA(weight)
    setattr(wo_a, scale_attr, retained_scale)

    out = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim)

    expected = weight.view(n_local_groups, o_lora_rank, hidden_dim)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_get_cached_wo_a_bf16_fp8_blockscale_caches() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    device = torch.device("cuda")
    torch.manual_seed(5)
    n_local_groups, o_lora_rank, hidden_dim = 2, 4, 8
    row_block, col_block = 2, 2
    row_blocks = o_lora_rank // row_block
    col_blocks = hidden_dim // col_block

    fp8_dtype = current_platform.fp8_dtype()
    weight_f32 = (
        torch.randn(
            n_local_groups, o_lora_rank, hidden_dim, dtype=torch.float32, device=device
        )
        * 0.1
    )
    weight_fp8 = weight_f32.to(fp8_dtype)
    scale = (
        torch.rand(
            n_local_groups, row_blocks, col_blocks, dtype=torch.float32, device=device
        )
        * 0.5
        + 0.5
    )
    wo_a = _FakeWoA(
        weight_fp8.reshape(n_local_groups * o_lora_rank, hidden_dim),
        weight_scale_inv=scale.reshape(n_local_groups * row_blocks, col_blocks),
    )

    out = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim)

    scale_full = scale.repeat_interleave(row_block, dim=-2).repeat_interleave(
        col_block, dim=-1
    )
    expected = (weight_fp8.to(torch.float32) * scale_full).to(torch.bfloat16)
    assert out.shape == (n_local_groups, o_lora_rank, hidden_dim)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)

    # Second call returns the same cached object.
    assert _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim) is out


@requires_split_decode_arch
@torch.inference_mode()
def test_fp8_paged_mqa_logits_triton_matches_torch_ref() -> None:
    """Block-flat Triton decode logits vs the eager torch reference.

    Compare only valid key positions: the kernel skips the -inf pad that
    the torch ref writes past seq_len.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda")
    init_workspace_manager(device)
    torch.manual_seed(0)

    fp8_dtype = current_platform.fp8_dtype()
    batch_size = 2
    next_n = 1
    num_heads = 4
    head_size = 128
    block_size = 64
    max_model_len = 256
    seq_lens = [180, 64]
    num_pages = [(seq_len + block_size - 1) // block_size for seq_len in seq_lens]
    num_blocks = sum(num_pages)

    q = torch.randn(
        batch_size, next_n, num_heads, head_size, device=device, dtype=torch.bfloat16
    ).to(fp8_dtype)
    weights = torch.rand(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    context_lens = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    block_tables = torch.full(
        (batch_size, max(num_pages)), 0, device=device, dtype=torch.int32
    )
    page = 0
    for i, n_pages in enumerate(num_pages):
        block_tables[i, :n_pages] = torch.arange(
            page, page + n_pages, device=device, dtype=torch.int32
        )
        page += n_pages

    kv_cache = torch.empty(
        num_blocks, block_size, 1, head_size + 4, device=device, dtype=torch.uint8
    )
    values = torch.randn(
        num_blocks, block_size, head_size, device=device, dtype=torch.bfloat16
    ).to(fp8_dtype)
    scales = torch.rand(
        num_blocks, block_size, device=device, dtype=torch.float32
    ).clamp(min=1e-3)
    kv_flat = kv_cache.view(num_blocks, -1)
    scale_off = block_size * head_size
    kv_flat[:, :scale_off] = values.reshape(num_blocks, -1).view(torch.uint8)
    kv_flat[:, scale_off:] = scales.contiguous().view(torch.uint8).view(num_blocks, -1)

    got = mod.rocm_fp8_paged_mqa_logits_triton(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )
    ref = mod.fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )
    for i, seq_len in enumerate(seq_lens):
        torch.testing.assert_close(
            got[i, :seq_len], ref[i, :seq_len], atol=2e-3, rtol=2e-3
        )


def test_indexer_k_is_c4a_block_flat_gate() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _indexer_k_is_c4a_block_flat,
    )

    assert not _indexer_k_is_c4a_block_flat(1)
    assert not _indexer_k_is_c4a_block_flat(2)
    assert _indexer_k_is_c4a_block_flat(4)
    assert not _indexer_k_is_c4a_block_flat(128)


@requires_split_decode_arch
@torch.inference_mode()
def test_paged_mqa_logits_gate_v41_shuffle_vs_c4a_flat() -> None:
    """Ratio 1/2 stay on AITER; ratio 4 takes Triton."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
        indexer_k_quant_and_cache_triton,
        rocm_fp8_paged_mqa_logits,
        rocm_fp8_paged_mqa_logits_triton,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda")
    init_workspace_manager(device)
    torch.manual_seed(1)

    fp8_dtype = current_platform.fp8_dtype()
    batch_size = 1
    next_n = 1
    num_heads = 32
    head_size = 128
    block_size = 64
    seq_len = 180
    max_model_len = 256
    num_pages = (seq_len + block_size - 1) // block_size
    num_tokens = num_pages * block_size

    q = torch.randn(
        batch_size, next_n, num_heads, head_size, device=device, dtype=torch.bfloat16
    ).to(fp8_dtype)
    weights = torch.rand(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    context_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    block_tables = torch.arange(num_pages, device=device, dtype=torch.int32).view(1, -1)
    dummy_sched = torch.zeros(8, 2, device=device, dtype=torch.int32)

    k_bf16 = torch.randn(num_tokens, head_size, device=device, dtype=torch.bfloat16)
    slot = torch.arange(num_tokens, device=device, dtype=torch.int32)
    cache_shuffle = torch.zeros(
        num_pages, block_size, head_size + 4, device=device, dtype=torch.uint8
    )
    indexer_k_quant_and_cache_triton(k_bf16, cache_shuffle, slot, 128, "ue8m0")
    cache_shuffle = cache_shuffle.unsqueeze(2)

    aiter = rocm_fp8_paged_mqa_logits(
        q,
        cache_shuffle,
        weights,
        context_lens,
        block_tables,
        dummy_sched,
        max_model_len,
        compress_ratio=1,
    )
    for ratio in (1, 2):
        got = rocm_fp8_paged_mqa_logits(
            q,
            cache_shuffle,
            weights,
            context_lens,
            block_tables,
            dummy_sched,
            max_model_len,
            compress_ratio=ratio,
        )
        torch.testing.assert_close(got[:, :seq_len], aiter[:, :seq_len], atol=0, rtol=0)

    cache_flat = torch.empty(
        num_pages, block_size, 1, head_size + 4, device=device, dtype=torch.uint8
    )
    values = torch.randn(
        num_pages, block_size, head_size, device=device, dtype=torch.bfloat16
    ).to(fp8_dtype)
    scales = torch.rand(
        num_pages, block_size, device=device, dtype=torch.float32
    ).clamp(min=1e-3)
    kv_flat = cache_flat.view(num_pages, -1)
    scale_off = block_size * head_size
    kv_flat[:, :scale_off] = values.reshape(num_pages, -1).view(torch.uint8)
    kv_flat[:, scale_off:] = scales.contiguous().view(torch.uint8).view(num_pages, -1)

    gated_c4a = rocm_fp8_paged_mqa_logits(
        q,
        cache_flat,
        weights,
        context_lens,
        block_tables,
        dummy_sched,
        max_model_len,
        compress_ratio=4,
    )
    triton = rocm_fp8_paged_mqa_logits_triton(
        q, cache_flat, weights, context_lens, block_tables, max_model_len
    )
    ref = fp8_paged_mqa_logits_torch(
        q, cache_flat, weights, context_lens, block_tables, max_model_len
    )
    torch.testing.assert_close(
        gated_c4a[:, :seq_len], triton[:, :seq_len], atol=0, rtol=0
    )
    torch.testing.assert_close(
        gated_c4a[:, :seq_len], ref[:, :seq_len], atol=2e-3, rtol=2e-3
    )
