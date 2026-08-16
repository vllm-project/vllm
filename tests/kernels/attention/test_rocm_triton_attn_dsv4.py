# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def _canonicalize_poisoned_fp8_ds_mla_cache_row(
    cache: torch.Tensor, block_size: int, slot: int = 0
) -> None:
    flat = cache.flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576
    scale_base = block_base + block_size * 576 + pos * 8
    flat[token_base] = 0
    flat[scale_base : scale_base + 7] = 254
    flat[token_base + NOPE_HEAD_DIM : token_base + 576].view(torch.bfloat16)[0] = 0


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
                    extra_cache, extra_slots, block_size, extra_use_fnuz
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


def _launch_gfx950_partial(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_indptr: torch.Tensor,
    num_splits: int,
    adaptive_splits: bool = False,
    poison_scratch: bool = False,
    one_wave_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    num_queries, num_heads, _ = q.shape
    part_m = torch.empty(
        num_queries,
        num_splits,
        num_heads,
        dtype=torch.float32,
        device=q.device,
    )
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        num_queries,
        num_splits,
        num_heads,
        HEAD_DIM,
        dtype=torch.float32,
        device=q.device,
    )
    if poison_scratch:
        part_m.fill_(float("nan"))
        part_l.fill_(float("nan"))
        part_acc.fill_(float("nan"))
        part_acc[:, 1::2].fill_(float("inf"))
    assert part_m.is_contiguous()
    assert part_l.is_contiguous()
    assert part_acc.is_contiguous()

    mod._sparse_attn_decode_gfx950_partial_kernel[
        (num_queries, num_splits, (num_heads + 15) // 16)
    ](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        part_m,
        part_l,
        part_acc,
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        HEAD_DIM**-0.5,
        num_heads,
        HAS_EXTRA=True,
        NOPE_DIM=NOPE_HEAD_DIM,
        ROPE_DIM=ROPE_HEAD_DIM,
        IS_FNUZ_MAIN=current_platform.is_fp8_fnuz(),
        IS_FNUZ_EXTRA=False,
        TRUST_EXTRA_CACHE_NAN_FREE=True,
        ADAPTIVE_SPLITS=adaptive_splits,
        ONE_WAVE_SPLITS=one_wave_splits or num_splits,
        BLOCK_H=16,
        BLOCK_K=32,
        NUM_SPLITS=num_splits,
        NUM_STAGES=1,
        num_warps=4,
        waves_per_eu=0,
    )
    return part_m, part_l, part_acc


def _launch_sparse_decode_reduce(
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    adaptive_splits: bool,
) -> torch.Tensor:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    num_queries, num_splits, num_heads = part_m.shape
    out = torch.empty(
        (num_queries, num_heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=part_m.device,
    )
    attn_sink = torch.empty(1, dtype=torch.float32, device=part_m.device)
    mod._sparse_attn_decode_reduce_kernel[(num_queries, num_heads)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=False,
        ADAPTIVE_SPLITS=adaptive_splits,
        COMB_DIM=HEAD_DIM,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=1 << (num_splits - 1).bit_length(),
        num_warps=4,
    )
    return out


@torch.inference_mode()
def test_paged_mqa_logits_do_not_contain_nan(monkeypatch) -> None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")

    class FakeWorkspaceManager:
        def get_simultaneous(self, *shapes_and_dtypes):
            return [
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in shapes_and_dtypes
            ]

    def fake_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        out_logits,
        context_lens,
        block_tables,
        max_seq_len,
        **kwargs,
    ):
        del (
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            max_seq_len,
            kwargs,
        )
        out_logits.fill_(float("nan"))

    monkeypatch.setattr(mod, "_ON_GFX942", False)
    monkeypatch.setattr(mod, "_ON_GFX950", True)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        mod,
        "paged_mqa_logits_module",
        lambda: SimpleNamespace(deepgemm_fp8_paged_mqa_logits=fake_paged_mqa_logits),
    )
    monkeypatch.setattr(
        mod, "current_workspace_manager", lambda: FakeWorkspaceManager()
    )

    q_fp8 = torch.empty((1, 1, 1, 1), dtype=torch.uint8, device=device)
    kv_cache_fp8 = torch.empty((1, 1, 1, 5), dtype=torch.uint8, device=device)
    logits = mod.rocm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        torch.empty((1, 1), dtype=torch.float32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        1,
    )

    assert not torch.isnan(logits).any()


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


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_decode_trusted_extra_matches_legacy_scrub() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
    )

    device = torch.device("cuda")
    block_size = 4
    main_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8, device=device)
    legacy_extra = torch.zeros_like(main_cache)
    _poison_fp8_ds_mla_cache_row(main_cache, block_size)
    _poison_fp8_ds_mla_cache_row(legacy_extra, block_size)
    canonical_extra = legacy_extra.clone()
    _canonicalize_poisoned_fp8_ds_mla_cache_row(canonical_extra, block_size)
    indices = torch.zeros(1, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    q = torch.ones(1, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kwargs = {
        "q": q,
        "main_cache": main_cache,
        "main_indices": indices,
        "main_indptr": indptr,
        "scale": HEAD_DIM**-0.5,
        "attn_sink": None,
        "nope_head_dim": NOPE_HEAD_DIM,
        "rope_head_dim": ROPE_HEAD_DIM,
        "extra_indices": indices,
        "extra_indptr": indptr,
    }

    legacy = _rocm_sparse_attn_decode_ragged_triton(
        **kwargs,
        extra_cache=legacy_extra,
    )
    trusted = _rocm_sparse_attn_decode_ragged_triton(
        **kwargs,
        extra_cache=canonical_extra,
        extra_cache_nan_free=True,
    )

    assert torch.equal(trusted, legacy)
    assert torch.equal(trusted, torch.zeros_like(trusted))


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
    assert mod._decode_num_splits(8, 1, 128.0, 8192.0) == 16
    assert mod._decode_num_splits(64, 1, 128.0, 8192.0) == 4

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
    for num_queries in (1, 8):
        assert mod._decode_gfx950_num_splits(num_queries, 1, 128, 32) == 32
        assert mod._decode_gfx950_num_splits(num_queries, 1, 128, 7812) == 32

    assert mod._decode_gfx950_num_splits(17, 1, 128, 32) == 4
    assert mod._decode_gfx950_num_splits(16, 1, 128, 781) == 13
    assert mod._decode_gfx950_num_splits(48, 1, 128, 3906) == 10
    for extra_rows in (32, 256):
        assert mod._decode_gfx950_num_splits(64, 1, 128, extra_rows) == 4
    assert mod._decode_gfx950_num_splits(64, 1, 128, 781) == 7
    for extra_rows in (3906, 7812):
        assert mod._decode_gfx950_num_splits(64, 1, 128, extra_rows) == 8
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
    monkeypatch.setattr(mod, split_fn, lambda *args, **kwargs: num_splits)

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
@pytest.mark.parametrize("num_splits", [1, 2, 3, 8, 32])
@torch.inference_mode()
def test_sparse_attn_decode_gfx950_partial_buffer_layout(num_splits: int) -> None:
    device = torch.device("cuda")
    block_size = 64
    num_heads = 16
    entries_per_split = 65
    num_extra = num_splits * entries_per_split
    q = torch.full(
        (2, num_heads, HEAD_DIM),
        0.125,
        dtype=torch.bfloat16,
        device=device,
    )
    main_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8, device=device)
    main_indices = torch.empty(0, dtype=torch.int32, device=device)
    main_indptr = torch.zeros(3, dtype=torch.int32, device=device)
    extra_cache = _pack_fp8_ds_mla_cache(
        torch.full(
            (num_extra, HEAD_DIM),
            0.125,
            dtype=torch.bfloat16,
            device=device,
        ),
        block_size,
        use_fnuz=False,
    )
    extra_indices = torch.arange(num_extra, dtype=torch.int32, device=device)
    extra_indptr = torch.tensor(
        [0, num_extra, num_extra], dtype=torch.int32, device=device
    )

    batch = _launch_gfx950_partial(
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        num_splits,
    )
    single_nonempty = _launch_gfx950_partial(
        q[:1],
        main_cache,
        main_indices,
        torch.zeros(2, dtype=torch.int32, device=device),
        extra_cache,
        extra_indices,
        torch.tensor([0, num_extra], dtype=torch.int32, device=device),
        num_splits,
    )
    single_empty = _launch_gfx950_partial(
        q[1:],
        main_cache,
        main_indices,
        torch.zeros(2, dtype=torch.int32, device=device),
        extra_cache,
        torch.empty(0, dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        num_splits,
    )

    for batch_part, nonempty_part, empty_part in zip(
        batch, single_nonempty, single_empty
    ):
        assert torch.equal(batch_part[:1], nonempty_part)
        assert torch.equal(batch_part[1:], empty_part)

    part_m, part_l, part_acc = batch
    expected_m = HEAD_DIM * 0.125**2 * HEAD_DIM**-0.5
    torch.testing.assert_close(
        part_m[0],
        torch.full_like(part_m[0], expected_m),
        rtol=1e-4,
        atol=1e-4,
    )
    assert torch.equal(part_l[0], torch.full_like(part_l[0], float(entries_per_split)))
    torch.testing.assert_close(
        part_acc[0],
        torch.full_like(part_acc[0], entries_per_split * 0.125),
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        part_m[1], torch.full_like(part_m[1], torch.finfo(torch.float32).min)
    )
    assert torch.equal(part_l[1], torch.zeros_like(part_l[1]))
    assert torch.equal(part_acc[1], torch.zeros_like(part_acc[1]))


@requires_gfx950
@torch.inference_mode()
def test_sparse_attn_decode_gfx950_adaptive_reduce_ignores_stale_scratch() -> None:
    device = torch.device("cuda")
    torch.manual_seed(19)
    block_size = 64
    num_heads = 16
    num_splits = 32
    main_use_fnuz = current_platform.is_fp8_fnuz()
    q = torch.randn(3, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    q *= 0.125
    main_kv = torch.randn(3, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, main_use_fnuz)
    main_rows = [[0], [1], [2]]
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)
    extra_kv = torch.randn(2112, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, use_fnuz=False)
    extra_rows = [
        list(range(64)),
        list(range(64, 576)),
        list(range(576, 2112)),
    ]
    extra_indices, extra_indptr = _ragged_from_rows(extra_rows, device)

    part_m, part_l, part_acc = _launch_gfx950_partial(
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        num_splits,
        adaptive_splits=True,
        poison_scratch=True,
        one_wave_splits=16,
    )

    neg_large = torch.finfo(torch.float32).min
    assert torch.equal(part_m[0, 4:], torch.full_like(part_m[0, 4:], neg_large))
    assert torch.equal(part_l[0, 4:], torch.zeros_like(part_l[0, 4:]))
    assert torch.isnan(part_acc[0, 4::2]).all()
    assert torch.isinf(part_acc[0, 5::2]).all()
    assert (part_l[1, :16] > 0).all()
    assert torch.equal(part_m[1, 16:], torch.full_like(part_m[1, 16:], neg_large))
    assert torch.equal(part_l[1, 16:], torch.zeros_like(part_l[1, 16:]))
    assert torch.isnan(part_acc[1, 16::2]).all()
    assert torch.isinf(part_acc[1, 17::2]).all()
    assert (part_l[2] > 0).all()
    assert torch.isfinite(part_acc[2]).all()

    actual = _launch_sparse_decode_reduce(
        part_m,
        part_l,
        part_acc,
        adaptive_splits=True,
    )
    expected = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=main_rows,
        scale=HEAD_DIM**-0.5,
        attn_sink=None,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=extra_rows,
        main_use_fnuz=main_use_fnuz,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


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


@requires_split_decode_arch
@torch.inference_mode()
def test_sparse_attn_decode_long_context(monkeypatch) -> None:
    """Validate the production C128A maximum of 8192 selected KV rows."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")
    torch.manual_seed(11)
    block_size = 64
    num_kv = 8192
    num_heads = 16
    q = torch.randn(1, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(
        torch.zeros(1, HEAD_DIM, dtype=torch.bfloat16, device=device),
        block_size,
        use_fnuz=current_platform.is_fp8_fnuz(),
    )
    extra_cache = _pack_fp8_ds_mla_cache(
        torch.randn(num_kv, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125,
        block_size,
        use_fnuz=False,
    )
    empty_indices = torch.empty(0, dtype=torch.int32, device=device)
    empty_indptr = torch.zeros(2, dtype=torch.int32, device=device)
    extra_indices = torch.arange(num_kv, dtype=torch.int32, device=device)
    extra_indptr = torch.tensor([0, num_kv], dtype=torch.int32, device=device)
    scale = HEAD_DIM**-0.5

    split_fn = "_decode_gfx950_num_splits" if _on_gfx950() else "_decode_num_splits"
    num_splits = 32 if _on_gfx950() else 16
    monkeypatch.setattr(mod, split_fn, lambda *args, **kwargs: num_splits)
    actual = mod._rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=empty_indices,
        main_indptr=empty_indptr,
        scale=scale,
        attn_sink=None,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )

    kv = _read_fp8_ds_mla_cache_rows(
        extra_cache, extra_indices.to(torch.int64), block_size, use_fnuz=False
    )
    scores = torch.matmul(q[0].float(), kv.T) * scale
    expected = torch.matmul(torch.softmax(scores, dim=-1), kv)
    torch.testing.assert_close(
        actual[0], expected.to(torch.bfloat16), atol=2e-2, rtol=2e-2
    )


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
