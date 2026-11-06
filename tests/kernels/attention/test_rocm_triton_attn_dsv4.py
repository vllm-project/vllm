# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

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


def _on_cdna3() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import get_cdna_version

        return get_cdna_version() == 3
    except ImportError:
        return False


# The flash-decode split-K decode path is only tuned for AMD gfx942/gfx950; other
# architectures take the fallback decode kernel, so its tests are skipped there.
requires_split_decode_arch = pytest.mark.skipif(
    not _on_split_decode_arch(),
    reason="split-K decode and the V4.1 combine tests need an AMD gfx942/gfx950",
)
requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(),
    reason="optimized sparse decode partial is gfx950-only",
)
requires_cdna3 = pytest.mark.skipif(
    not _on_cdna3(),
    reason="the BF16 mega-attention prefill is CDNA3-only",
)

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM


@pytest.fixture
def enable_aiter_mqa(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops

    with monkeypatch.context() as context:
        context.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()
        yield
    rocm_aiter_ops.refresh_env_variables()


@requires_gfx950
@pytest.mark.parametrize("column_scales", [False, True], ids=["vector", "column"])
@torch.inference_mode()
def test_unpaged_mqa_logits_preserves_intervals_and_scale_layout(
    column_scales, monkeypatch, enable_aiter_mqa
) -> None:
    """The vLLM AITER dispatch honors ragged intervals and per-token scales."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    assert rocm_aiter_ops.is_enabled(), "AITER is required on gfx950"
    assert mod.mqa_logits_module() is not None, "AITER MQA logits are required"
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = current_platform.fp8_dtype()
    q = torch.randn(7, 32, 128, device=device).to(dtype)
    k = torch.randn(259, 128, device=device).to(dtype)
    scales = torch.linspace(0.25, 1.25, k.shape[0], device=device)
    if column_scales:
        scales = scales[:, None]
    weights = torch.rand(q.shape[:2], device=device) / q.shape[1]
    starts = torch.tensor([0, 3, 128, 5, 0, 257, 259], device=device, dtype=torch.int32)
    ends = torch.tensor(
        [1, 129, 259, 5, 259, 259, 259], device=device, dtype=torch.int32
    )

    module = mod.mqa_logits_module()
    implementation = module.fp8_mqa_logits
    calls = 0

    def traced_implementation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return implementation(*args, **kwargs)

    monkeypatch.setattr(module, "fp8_mqa_logits", traced_implementation)
    actual = mod.rocm_fp8_mqa_logits(q, (k, scales), weights, starts, ends)
    assert calls == 1
    score = torch.einsum("mhd,nd->mhn", q.float(), k.float()) * scales.reshape(-1)
    expected = (score.relu() * weights[..., None]).sum(dim=1)
    positions = torch.arange(k.shape[0], device=device)
    valid = (positions >= starts[:, None]) & (positions < ends[:, None])
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert not torch.isneginf(actual[valid]).any()
    torch.testing.assert_close(actual[valid], expected[valid], rtol=1e-3, atol=1e-3)


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
    if not use_fnuz and current_platform.is_fp8_fnuz():
        # As the compressed-cache writer stores OCP records on FNUZ platforms.
        records = cache.view(num_blocks, -1)[:, : block_size * 576]
        nope = records.view(num_blocks, block_size, 576)[..., :448]
        nope.masked_fill_(nope == 0x80, 0)
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


@pytest.mark.parametrize("width", [16, 640, 2176])
@pytest.mark.parametrize("num_queries", [1, 70, 1000])
@torch.inference_mode()
def test_build_ragged_indices_from_dense_drops_invalid_entries(
    width: int, num_queries: int
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    device = torch.device("cuda")
    set_random_seed(width + num_queries)
    num_rows = 5000
    indices = torch.randint(0, num_rows + 100, (num_queries, width))
    indices[torch.rand(num_queries, width) < 0.3] = -1
    lengths = torch.randint(0, width + 8, (num_queries,))

    flat, indptr = build_ragged_indices_from_dense(
        indices.to(device, torch.int32),
        lengths.to(device, torch.int32),
        num_rows=num_rows,
    )

    # Each row keeps its in-range entries within its length, in order.
    expected = [
        [x for x in row[:n] if 0 <= x < num_rows]
        for row, n in zip(indices.tolist(), lengths.tolist())
    ]
    assert _rows_from_ragged(flat, indptr) == expected


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


NOPE_ONLY_HEAD_DIM = 512


def _nope_only_ragged_inputs(
    kv_lens: list[int], num_heads: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a GLM-like 512+0 NoPE decode batch with ragged KV segments."""
    num_kv = 4096
    kv = (
        torch.randn(num_kv, NOPE_ONLY_HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.125
    )
    q = (
        torch.randn(
            len(kv_lens),
            num_heads,
            NOPE_ONLY_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    rows = [torch.randperm(num_kv)[:length].tolist() for length in kv_lens]
    indices, indptr = _ragged_from_rows(rows, device)
    return q, kv, indices, indptr


# Lengths below the split count leave splits empty; non-multiples force a tail.
_NOPE_ONLY_KV_LENS = [0, 1, 3, 37, 129, 2048]


@requires_split_decode_arch
@pytest.mark.parametrize("num_splits", [2, 32])
@pytest.mark.parametrize("with_sink", [True, False])
@torch.inference_mode()
def test_sparse_attn_decode_bf16_split_k_matches_ragged(
    num_splits: int, with_sink: bool
) -> None:
    """Split-K decode must reproduce the single-workgroup ragged kernel."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_bf16_triton,
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    device = torch.device("cuda")
    set_random_seed(0)
    num_heads = 16
    q, kv, indices, indptr = _nope_only_ragged_inputs(
        _NOPE_ONLY_KV_LENS, num_heads, device
    )
    attn_sink = (
        torch.randn(num_heads, dtype=torch.float32, device=device)
        if with_sink
        else None
    )
    scale = NOPE_ONLY_HEAD_DIM**-0.5

    expected = _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_ONLY_HEAD_DIM,
        rope_head_dim=0,
    )
    actual = _rocm_sparse_attn_decode_ragged_bf16_triton(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_ONLY_HEAD_DIM,
        rope_head_dim=0,
        num_splits=num_splits,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _sparse_prefill_ragged_inputs(nope_dim: int, rope_dim: int) -> dict:
    device = torch.device("cuda")
    set_random_seed(7)
    head_dim = nope_dim + rope_dim
    return dict(
        q=torch.randn(3, 3, head_dim, dtype=torch.bfloat16, device=device) * 0.125,
        kv=torch.randn(5, head_dim, dtype=torch.bfloat16, device=device) * 0.125,
        indices=torch.tensor([0, 2, 1, 3, 4], dtype=torch.int32, device=device),
        indptr=torch.tensor([0, 2, 5, 5], dtype=torch.int32, device=device),
        scale=head_dim**-0.5,
        attn_sink=torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float32, device=device),
        nope_head_dim=nope_dim,
        rope_head_dim=rope_dim,
    )


# 448+64 is DeepSeek V4/V4.1, 512+64 is V3.2, and 256+0 stands in for the
# NoPE-only layouts where the destination spans the whole head dim.
SPARSE_PREFILL_DIMS = [(NOPE_HEAD_DIM, ROPE_HEAD_DIM), (512, 64), (256, 0)]


@pytest.mark.parametrize(("nope_dim", "rope_dim"), SPARSE_PREFILL_DIMS)
@torch.inference_mode()
def test_sparse_attn_prefill_ragged_out_matches_allocated(
    nope_dim: int, rope_dim: int
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    kwargs = _sparse_prefill_ragged_inputs(nope_dim, rope_dim)
    allocated = _rocm_sparse_attn_prefill_ragged_triton(**kwargs)

    q = kwargs["q"]
    dest = torch.empty(q.shape[0], q.shape[1], nope_dim, dtype=q.dtype, device=q.device)
    returned = _rocm_sparse_attn_prefill_ragged_triton(**kwargs, out=dest)

    assert returned is dest
    torch.testing.assert_close(dest, allocated[..., :nope_dim], atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("nope_dim", "rope_dim"), SPARSE_PREFILL_DIMS)
@torch.inference_mode()
def test_sparse_attn_prefill_ragged_out_view_leaves_neighbors_untouched(
    nope_dim: int, rope_dim: int
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    kwargs = _sparse_prefill_ragged_inputs(nope_dim, rope_dim)
    q = kwargs["q"]
    num_queries, num_heads = q.shape[0], q.shape[1]

    buffer = torch.full(
        (num_queries + 4, num_heads + 2, nope_dim),
        -7.0,
        dtype=q.dtype,
        device=q.device,
    )
    dest = buffer[2 : 2 + num_queries, :num_heads]
    assert not dest.is_contiguous()
    _rocm_sparse_attn_prefill_ragged_triton(**kwargs, out=dest)

    expected = _rocm_sparse_attn_prefill_ragged_triton(**kwargs)[..., :nope_dim]
    torch.testing.assert_close(dest, expected, atol=0.0, rtol=0.0)

    untouched = torch.ones_like(buffer, dtype=torch.bool)
    untouched[2 : 2 + num_queries, :num_heads] = False
    assert torch.all(buffer[untouched] == -7.0)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_sparse_attn_prefill_ragged_out_casts_to_dest_dtype(
    out_dtype: torch.dtype,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    kwargs = _sparse_prefill_ragged_inputs(512, 64)
    q = kwargs["q"]
    dest = torch.empty(q.shape[0], q.shape[1], 512, dtype=out_dtype, device=q.device)
    _rocm_sparse_attn_prefill_ragged_triton(**kwargs, out=dest)

    expected = _rocm_sparse_attn_prefill_ragged_triton(**kwargs)[..., :512]
    assert dest.dtype == out_dtype
    torch.testing.assert_close(dest.float(), expected.float(), atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_sparse_attn_prefill_head_dim_wide_out_keeps_rope_rows() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    kwargs = _sparse_prefill_ragged_inputs(NOPE_HEAD_DIM, ROPE_HEAD_DIM)
    q = kwargs["q"]
    dest = torch.empty_like(q)
    _rocm_sparse_attn_prefill_ragged_triton(**kwargs, out=dest)

    expected = _ref_sparse_prefill_ragged(
        q, kwargs["kv"], [[0, 2], [1, 3, 4], []], kwargs["scale"], kwargs["attn_sink"]
    )
    torch.testing.assert_close(dest, expected, atol=2e-2, rtol=2e-2)


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

    def fake_dense_fallback(*args, out, **kwargs):
        nonlocal dense_fallback_calls
        dense_fallback_calls += 1
        out.zero_()
        return out

    monkeypatch.setattr(mod, "_can_use_aiter_sparse_prefill_opus", lambda *args: True)
    monkeypatch.setattr(mod, "_get_aiter_sparse_prefill_opus", lambda: None)
    monkeypatch.setattr(mod, "_can_use_aiter_pa_prefill_sparse", lambda *args: False)
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


@pytest.mark.parametrize(
    ("num_queries", "cdna_version", "mla_enabled", "dtype", "kv_dtype", "expected"),
    [
        (512, 3, True, torch.bfloat16, torch.bfloat16, True),
        (512, 3, True, torch.float16, torch.float16, True),
        (511, 3, True, torch.bfloat16, torch.bfloat16, False),
        (512, 4, True, torch.bfloat16, torch.bfloat16, False),
        (512, 0, True, torch.bfloat16, torch.bfloat16, False),
        (512, 3, False, torch.bfloat16, torch.bfloat16, False),
        (512, 3, True, torch.float32, torch.float32, False),
        (512, 3, True, torch.bfloat16, torch.float16, False),
    ],
)
def test_can_use_aiter_pa_prefill_sparse(
    monkeypatch,
    num_queries: int,
    cdna_version: int,
    mla_enabled: bool,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    expected: bool,
) -> None:
    """Literal gate. Expected values are not derived from the cutoff constant."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod, "get_cdna_version", lambda: cdna_version)
    monkeypatch.setattr(rocm_aiter_ops, "is_mla_enabled", lambda: mla_enabled)
    q = torch.empty(num_queries, 1, HEAD_DIM, dtype=dtype)
    kv = torch.empty(4, 1, HEAD_DIM, dtype=kv_dtype)
    assert mod._can_use_aiter_pa_prefill_sparse(q, kv) is expected


@pytest.mark.parametrize(
    ("num_queries", "mla_enabled", "expect_aiter"),
    [(512, True, True), (511, True, False), (512, False, False)],
)
def test_sparse_attn_prefill_routes_gfx942_pa_prefill_sparse(
    monkeypatch, num_queries: int, mla_enabled: bool, expect_aiter: bool
) -> None:
    """Call site launches only when the gate passes."""
    from unittest.mock import Mock

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    kernel = None
    if expect_aiter:
        pytest.importorskip("aiter.ops.triton.attention.pa_prefill_sparse")
        from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse

        kernel = pa_prefill_sparse

    monkeypatch.setattr(rocm_aiter_ops, "is_mla_enabled", lambda: mla_enabled)
    monkeypatch.setattr(mod, "get_cdna_version", lambda: 3)
    monkeypatch.setattr(mod, "_can_use_aiter_sparse_prefill_opus", lambda *args: False)

    def zero_output(*args, **kwargs):
        kwargs["output"].zero_()

    def zero_out(*args, out, **kwargs):
        out.zero_()
        return out

    fake_aiter = Mock(side_effect=zero_output)
    fake_triton = Mock(side_effect=zero_out)
    monkeypatch.setattr(mod, "_rocm_sparse_attn_prefill_aiter", fake_aiter)
    monkeypatch.setattr(mod, "_rocm_sparse_attn_prefill_triton", fake_triton)
    monkeypatch.setattr(
        mod,
        "_rocm_sparse_attn_prefill_ragged_triton",
        lambda *args, **kwargs: pytest.fail("unexpected ragged Triton fallback"),
    )
    monkeypatch.setattr(
        mod,
        "build_ragged_indices_from_dense",
        lambda *args, **kwargs: (
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        ),
    )
    q = torch.empty(num_queries, 1, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.empty(4, 1, HEAD_DIM, dtype=torch.bfloat16)
    output = torch.empty_like(q)
    mod.rocm_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=torch.zeros(num_queries, 1, dtype=torch.int32),
        topk_length=torch.ones(num_queries, dtype=torch.int32),
        scale=HEAD_DIM**-0.5,
        head_dim=HEAD_DIM,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        attn_sink=torch.empty(1, dtype=torch.float32),
        output=output,
    )

    if expect_aiter:
        fake_aiter.assert_called_once()
        assert fake_aiter.call_args.args[0] is kernel
        fake_triton.assert_not_called()
    else:
        fake_aiter.assert_not_called()
        fake_triton.assert_called_once()


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
    assert mod._decode_gfx950_num_splits(4, 1, 2048.0, 0.0, 32) == 32


def test_sparse_decode_bf16_num_splits_floor_and_tile_clamp(monkeypatch) -> None:
    """Short rows skip splitting, and splits never exceed the tiles they walk."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod, "_decode_num_splits", lambda *args: 32)
    monkeypatch.setattr(mod, "_decode_gfx950_num_splits", lambda *args: 32)

    floor = mod._SPARSE_DECODE_BF16_MIN_SPLIT_LEN
    block_k = mod._SPARSE_DECODE_BF16_BLOCK_K
    assert mod.rocm_sparse_decode_bf16_num_splits(1, 1, floor - 1) == 1
    assert mod.rocm_sparse_decode_bf16_num_splits(1, 1, floor) == floor // block_k
    assert mod.rocm_sparse_decode_bf16_num_splits(1, 1, 2048) == 32


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
@pytest.mark.parametrize("enable_q_norm", [False, True])
@pytest.mark.parametrize("apply_q_rope", [False, True])
@pytest.mark.parametrize("optional_metadata", [False, True])
@pytest.mark.parametrize("index_repeats", [1, 33])
@pytest.mark.parametrize("heads", [8, 32])
@torch.inference_mode()
def test_hip_mega_prefill_mxfp8_output(
    enable_q_norm: bool,
    apply_q_rope: bool,
    optional_metadata: bool,
    index_repeats: int,
    heads: int,
) -> None:
    """Fused sparse prefill honors masks and writes only its MXFP8 row views."""
    device = torch.device("cuda")
    torch.manual_seed(21)
    tokens, width = 3, heads * HEAD_DIM
    q = torch.randn(tokens, heads, HEAD_DIM + 16, device=device).bfloat16()
    q = q[..., :HEAD_DIM]
    kv = torch.randn(6, 1, HEAD_DIM + 16, device=device).bfloat16() * 0.25
    kv = kv[..., :HEAD_DIM]
    indices = torch.tensor(
        [[-1, -1, -1, -1, -1], [0, -1, 3, 99, 5], [1, 4, 2, 3, 1]],
        dtype=torch.int32,
        device=device,
    )
    indices = indices.repeat(1, index_repeats)
    lengths = (
        None
        if optional_metadata
        else torch.tensor([0, indices.shape[1], 2], dtype=torch.int32, device=device)
    )
    sink = None if optional_metadata else torch.linspace(-1, 1, heads, device=device)
    positions = torch.tensor(
        [3, 8, 9], dtype=torch.int32 if apply_q_rope else torch.int64, device=device
    )
    cos_sin = _random_cos_sin_cache(16, device)
    q_ref = q.float()
    if enable_q_norm:
        q_ref *= torch.rsqrt(q_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
    cos, sin = cos_sin[positions.long(), None].chunk(2, dim=-1)
    if apply_q_rope:
        even = q_ref[..., NOPE_HEAD_DIM::2].clone()
        odd = q_ref[..., NOPE_HEAD_DIM + 1 :: 2].clone()
        q_ref[..., NOPE_HEAD_DIM::2] = even * cos - odd * sin
        q_ref[..., NOPE_HEAD_DIM + 1 :: 2] = odd * cos + even * sin
    limits = [indices.shape[1]] * tokens if lengths is None else lengths.cpu().tolist()
    rows = [
        [slot for slot in row[:limit] if 0 <= slot < kv.shape[0]]
        for row, limit in zip(indices.cpu().tolist(), limits)
    ]
    ref = _ref_sparse_prefill_ragged(
        q_ref.bfloat16(), kv[:, 0], rows, HEAD_DIM**-0.5, sink
    ).float()
    even = ref[..., NOPE_HEAD_DIM::2].clone()
    odd = ref[..., NOPE_HEAD_DIM + 1 :: 2].clone()
    ref[..., NOPE_HEAD_DIM::2] = even * cos + odd * sin
    ref[..., NOPE_HEAD_DIM + 1 :: 2] = odd * cos - even * sin

    data = torch.full(
        (tokens + 2, width + 16), 1.0, dtype=torch.float8_e4m3fn, device=device
    )
    scale = torch.full(
        (tokens + 2, width // 32 + 4), 17, dtype=torch.uint8, device=device
    )
    out_data, out_scale = data[1:-1, :width], scale[1:-1, : width // 32]
    torch.ops._rocm_C.fused_norm_rope_attn_rope_cast_fwd(
        q,
        kv,
        indices,
        lengths,
        sink,
        HEAD_DIM**-0.5,
        positions,
        cos_sin,
        enable_q_norm,
        1e-6,
        apply_q_rope,
        out_data,
        out_scale,
    )
    torch.testing.assert_close(
        _mxfp8_dequant(out_data, out_scale), ref.flatten(1), atol=3e-3, rtol=0.08
    )
    assert torch.count_nonzero(out_data[0].float()) == 0
    assert torch.all(data[[0, -1]].float() == 1)
    assert torch.all(data[:, width:].float() == 1)
    assert torch.all(scale[[0, -1]] == 17)
    assert torch.all(scale[:, width // 32 :] == 17)


def _rope_q(q: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """GPT-J RoPE on the last 64 dims in FP32, rounded to BF16, as the Q prep
    applies it."""
    cos, sin = cos_sin[positions.long(), None].chunk(2, dim=-1)
    x = q.float()
    even, odd = x[..., 448::2].clone(), x[..., 449::2].clone()
    x[..., 448::2] = even * cos - odd * sin
    x[..., 449::2] = even * sin + odd * cos
    return x.bfloat16()


def _mega_paged_prefill_case(
    heads: int,
    request_repeats: int,
    cache_padding: int,
    with_extra: bool,
    main_use_fnuz: bool,
    with_sink: bool,
    rope_q: bool = False,
) -> SimpleNamespace:
    """Paged prefill inputs with request boundaries, causal top-k and SWA
    replay, plus the inverse-RoPE'd reference rows. With ``rope_q`` the
    reference rotates Q first, for a kernel that takes Q before RoPE."""
    device = torch.device("cuda")
    torch.manual_seed(23)
    tokens = 5
    q = torch.randn(tokens, heads, HEAD_DIM + 16, device=device).bfloat16()
    q = q[..., :HEAD_DIM]
    main = _pack_fp8_ds_mla_cache(
        torch.randn(24, HEAD_DIM, device=device).bfloat16() * 0.25, 4, False
    )
    if main_use_fnuz:
        cpu = main.cpu().flatten(1)
        payload = cpu[:, : 4 * 576].reshape(6, 4, 576)[..., :448]
        values = payload.contiguous().view(torch.float8_e4m3fn).float() * 0.5
        payload.copy_(values.to(torch.float8_e4m3fnuz).view(torch.uint8))
        cpu[:, 4 * 576 :].reshape(6, 4, 8)[..., :7] += 1
        main.copy_(cpu.reshape_as(main))
    extra = (
        _pack_fp8_ds_mla_cache(
            torch.randn(12, HEAD_DIM, device=device).bfloat16() * 0.25, 2, False
        )
        if with_extra
        else None
    )
    main_pages = [[4, 1, 5], [5, 0, 1], [2, 0, 3]]
    extra_pages = [[3, 4, 1], [2, 0, 1], [5, 1, 4]]
    candidates = [
        [0, -1, 1, 0, 1],
        [1, 0, 99, -1, 0],
        [0, 2, -1, 99, 1],
        [1, 0, 4, 2, -1],
        [-1, -1, -1, -1, -1],
    ]
    topk = torch.tensor(candidates, dtype=torch.int32, device=device)
    lengths, gathers, request_ids = [5, 0, 10], [2, 0, 3], [0, 0, 2, 2, 2]
    logical_positions = [3, 4, 7, 8, 9]
    main_rows, extra_rows = [], []
    for t, (req, pos) in enumerate(zip(request_ids, logical_positions)):
        start = max(pos - 3, 0, lengths[req] - gathers[req])
        main_rows.append(
            [main_pages[req][i // 4] * 4 + i % 4 for i in range(start, pos + 1)]
        )
        selected = candidates[t][: min((pos + 1) // 2, len(candidates[t]))]
        extra_rows.append(
            [
                extra_pages[req][i // 2] * 2 + i % 2
                for i in selected
                if 0 <= i < lengths[req] // 2
            ]
        )
    sink = torch.linspace(-1, 1, heads, device=device) if with_sink else None
    positions = torch.tensor(
        [3, 4, 7, 8, 9], device=device, dtype=torch.int32 if with_sink else torch.int64
    )
    cos_sin = _random_cos_sin_cache(16, device)
    ref = (
        _ref_sparse_decode_ragged(
            _rope_q(q, positions, cos_sin) if rope_q else q,
            main,
            main_rows,
            HEAD_DIM**-0.5,
            sink,
            4,
            extra,
            extra_rows if with_extra else None,
            main_use_fnuz,
            extra_block_size=2,
        )
        .bfloat16()
        .float()
    )
    cos, sin = cos_sin[positions.long(), None].chunk(2, dim=-1)
    even, odd = ref[..., 448::2].clone(), ref[..., 449::2].clone()
    ref[..., 448::2] = even * cos + odd * sin
    ref[..., 449::2] = odd * cos - even * sin

    # Cross the CTA threshold to exercise both KV tile sizes.
    if request_repeats > 1:
        q = q.repeat(request_repeats, 1, 1)
        positions = positions.repeat(request_repeats)
        topk = topk.repeat(request_repeats, 1)
        ref = ref.repeat(request_repeats, 1, 1)
    tokens *= request_repeats
    main_pages *= request_repeats
    extra_pages *= request_repeats
    lengths *= request_repeats
    gathers *= request_repeats

    # Keep gaps between pages, metadata rows and output rows.
    page_stride = 4 * 584 + cache_padding
    storage = torch.zeros(
        12 * page_stride + cache_padding, dtype=torch.uint8, device=device
    )
    padded_cache = storage.as_strided(
        (12, 4, 584), (page_stride, 584, 1), storage_offset=cache_padding
    )
    padded_cache[::2].copy_(main)
    tables = torch.tensor(
        [row + [-1] for row in main_pages], device=device, dtype=torch.int32
    )[:, :3]
    extra_table = (
        torch.tensor(extra_pages, device=device, dtype=torch.int32)
        if with_extra
        else None
    )
    starts = torch.tensor(
        [19 + 5 * i + offset for i in range(request_repeats) for offset in (0, 2, 2)]
        + [19 + tokens],
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    gather_lens = torch.tensor(gathers, device=device, dtype=torch.int32)
    args = (
        q,
        padded_cache[::2],
        tables,
        extra,
        extra_table,
        topk,
        starts,
        seq_lens,
        gather_lens,
        4,
        2 if with_extra else 0,
        sink,
        HEAD_DIM**-0.5,
        positions,
        cos_sin,
        main_use_fnuz,
    )
    return SimpleNamespace(
        args=args, ref=ref, tokens=tokens, gather_lens=gather_lens, topk=topk
    )


_MEGA_PAGED_PREFILL_CASES = pytest.mark.parametrize(
    "heads,request_repeats,cache_padding,with_extra,main_use_fnuz,with_sink",
    [
        (heads, repeats, padding, extra, fnuz, sink)
        for heads in (8, 32)
        for repeats in (1, 103)
        for padding in (0, 2)
        for extra in (False, True)
        for fnuz in (False, True)
        for sink in (False, True)
    ],
)


@requires_gfx950
@_MEGA_PAGED_PREFILL_CASES
@torch.inference_mode()
def test_hip_mega_paged_prefill_mxfp8_output(
    heads: int,
    request_repeats: int,
    cache_padding: int,
    with_extra: bool,
    main_use_fnuz: bool,
    with_sink: bool,
) -> None:
    """Fused prefill respects request boundaries, causal top-k and SWA replay."""
    case = _mega_paged_prefill_case(
        heads, request_repeats, cache_padding, with_extra, main_use_fnuz, with_sink
    )
    device, tokens, width = case.ref.device, case.tokens, heads * HEAD_DIM
    data = torch.full(
        (tokens + 2, width + 16), 1.0, dtype=torch.float8_e4m3fn, device=device
    )
    scales = torch.full(
        (tokens + 2, width // 32 + 4), 17, dtype=torch.uint8, device=device
    )
    out, out_scale = data[1:-1, :width], scales[1:-1, : width // 32]

    def run():
        torch.ops._rocm_C.fused_norm_rope_attn_rope_cast_prefill(
            *case.args, out, out_scale
        )

    run()
    torch.testing.assert_close(
        _mxfp8_dequant(out, out_scale), case.ref.flatten(1), atol=3e-3, rtol=0.08
    )
    # A shorter SWA span must not retain a previous invocation's KV.
    case.gather_lens.zero_()
    case.topk.fill_(-1)
    run()
    assert torch.count_nonzero(out.float()) == 0
    assert torch.all(data[[0, -1]].float() == 1)
    assert torch.all(data[:, width:].float() == 1)
    assert torch.all(scales[[0, -1]] == 17)
    assert torch.all(scales[:, width // 32 :] == 17)


@requires_cdna3
@_MEGA_PAGED_PREFILL_CASES
@pytest.mark.parametrize("rotate_q", [False, True])
@torch.inference_mode()
def test_hip_mega_paged_prefill_bf16_output(
    heads: int,
    request_repeats: int,
    cache_padding: int,
    with_extra: bool,
    main_use_fnuz: bool,
    with_sink: bool,
    rotate_q: bool,
) -> None:
    """The CDNA3 fused prefill applies Q RoPE when asked and stores BF16 rows
    with inverse RoPE applied."""
    case = _mega_paged_prefill_case(
        heads,
        request_repeats,
        cache_padding,
        with_extra,
        main_use_fnuz,
        with_sink,
        rope_q=rotate_q,
    )
    device, tokens = case.ref.device, case.tokens
    # Pad heads and rows so out-of-view writes show up.
    buffer = torch.full(
        (tokens + 2, heads + 1, HEAD_DIM + 16),
        3.0,
        dtype=torch.bfloat16,
        device=device,
    )
    out = buffer[1:-1, :heads, :HEAD_DIM]

    def run():
        torch.ops._rocm_C.fused_norm_rope_attn_rope_prefill_bf16(
            *case.args, rotate_q, out
        )

    run()
    torch.testing.assert_close(out.float(), case.ref, atol=3e-3, rtol=0.02)
    case.gather_lens.zero_()
    case.topk.fill_(-1)
    run()
    assert torch.count_nonzero(out.float()) == 0
    assert torch.all(buffer[[0, -1]].float() == 3)
    assert torch.all(buffer[:, heads:].float() == 3)
    assert torch.all(buffer[..., HEAD_DIM:].float() == 3)


@requires_cdna3
@torch.inference_mode()
def test_hip_mega_paged_prefill_bf16_long_ramp() -> None:
    """A chunk with the model's window and top-k, whose key magnitudes grow
    along the sequence so the running max keeps rising across many steps."""
    device = torch.device("cuda")
    torch.manual_seed(29)
    heads, window, ratio, topk_len, block = 16, 128, 2, 512, 64
    seq_len, tokens = 2048, 640
    ramp = torch.linspace(0.1, 6.0, seq_len, device=device)[:, None]
    main = _pack_fp8_ds_mla_cache(
        (torch.randn(seq_len, HEAD_DIM, device=device) * ramp).bfloat16(), block, False
    )
    extra_rows = seq_len // ratio
    extra = _pack_fp8_ds_mla_cache(
        (torch.randn(extra_rows, HEAD_DIM, device=device) * ramp[::ratio]).bfloat16(),
        block,
        False,
    )
    q = torch.randn(tokens, heads, HEAD_DIM, device=device).bfloat16()
    positions = torch.arange(seq_len - tokens, seq_len, device=device)
    # Distinct causal top-k rows in random order.
    order = torch.rand(tokens, extra_rows, device=device)
    order.masked_fill_(
        torch.arange(extra_rows, device=device) >= (positions[:, None] + 1) // ratio, 2
    )
    topk = order.argsort(1)[:, :topk_len].int()
    sink = torch.linspace(-1, 1, heads, device=device)
    cos_sin = _random_cos_sin_cache(seq_len, device)
    scale = HEAD_DIM**-0.5

    main_kv = _read_fp8_ds_mla_cache_rows(
        main, torch.arange(seq_len, device=device), block, False
    )
    extra_kv = _read_fp8_ds_mla_cache_rows(
        extra, torch.arange(extra_rows, device=device), block, False
    )
    window_rows = positions[:, None] - window + 1 + torch.arange(window, device=device)
    q_rope = _rope_q(q, positions, cos_sin).float()
    ref = torch.empty(tokens, heads, HEAD_DIM, device=device)
    mag = torch.empty(tokens, device=device)
    for s in range(0, tokens, 128):
        kv = torch.cat(
            [main_kv[window_rows[s : s + 128]], extra_kv[topk[s : s + 128]]], 1
        )
        mag[s : s + 128] = kv.abs().amax((1, 2))
        scores = torch.einsum("thd,tkd->thk", q_rope[s : s + 128], kv) * scale
        scores = torch.cat([scores, sink[None, :, None].expand(len(kv), -1, 1)], -1)
        probs = torch.softmax(scores, -1)[..., :-1]
        ref[s : s + 128] = torch.einsum("thk,tkd->thd", probs, kv)
    ref = ref.bfloat16().float()
    cos, sin = cos_sin[positions, None].chunk(2, dim=-1)
    even, odd = ref[..., 448::2].clone(), ref[..., 449::2].clone()
    ref[..., 448::2] = even * cos + odd * sin
    ref[..., 449::2] = odd * cos - even * sin

    out = torch.empty(tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    torch.ops._rocm_C.fused_norm_rope_attn_rope_prefill_bf16(
        q,
        main,
        torch.arange(seq_len // block, device=device, dtype=torch.int32)[None],
        extra,
        torch.arange(extra_rows // block, device=device, dtype=torch.int32)[None],
        topk,
        torch.tensor([0, tokens], device=device, dtype=torch.int32),
        torch.tensor([seq_len], device=device, dtype=torch.int32),
        torch.tensor([seq_len], device=device, dtype=torch.int32),
        window,
        ratio,
        sink,
        scale,
        positions,
        cos_sin,
        False,
        True,
        out,
    )
    mag = mag[:, None, None]
    torch.testing.assert_close(out.float() / mag, ref / mag, atol=3e-3, rtol=0.02)
    # The max error is set by BF16/FP8 rounding. A row sum built from P values
    # other than the rounded BF16 ones fed to PV shows up only in the RMS:
    # over seeds 1-12 it is 2.1-2.3e-4 with the rounded P, 3.1-3.4e-4 without.
    rms = ((out.float() - ref) / mag).pow(2).mean().sqrt().item()
    assert rms < 2.7e-4, rms


def _mega_layer(n_decode: int, n_prefill: int, device: torch.device):
    """A CDNA3 mega layer with only the attention state, over one prefill
    request and ``n_decode`` one-token decode requests."""
    from vllm.forward_context import ForwardContext
    from vllm.models.deepseek_v41.amd.rocm_mega_attn import (
        DeepseekV41ROCMAiterMegaAttnAttention,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    heads, window, ratio, topk, block = 16, 32, 2, 64, 16
    n_tokens = n_decode + n_prefill
    seq_len = ratio * topk + n_prefill
    layer = DeepseekV41ROCMAiterMegaAttnAttention.__new__(
        DeepseekV41ROCMAiterMegaAttnAttention
    )
    torch.nn.Module.__init__(layer)
    layer.n_local_heads = layer.padded_heads = heads
    layer.head_dim, layer.rope_head_dim = HEAD_DIM, ROPE_HEAD_DIM
    layer.nope_head_dim = NOPE_HEAD_DIM
    layer.scale, layer.eps, layer.kv_mxfp8 = HEAD_DIM**-0.5, 1e-6, False
    layer.window_size, layer.compress_ratio = window, ratio
    layer.max_model_len, layer.max_num_batched_tokens = seq_len, n_tokens
    layer.PREFILL_CHUNK_SIZE = 1
    layer._use_aiter_sparse_mla = layer._has_kv_transfer = False
    layer._attn_rotates_q_from, layer._prefill_rotate_q = None, False
    layer.kv_cache_dtype = "fp8_ds_mla"
    layer.compressed_cache_prefix = "compressed"
    fnuz = current_platform.is_fp8_fnuz()
    swa_kv = torch.randn(seq_len, HEAD_DIM, device=device).bfloat16() * 0.25
    extra_kv = torch.randn(seq_len // ratio, HEAD_DIM, device=device).bfloat16() * 0.25
    layer.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=_pack_fp8_ds_mla_cache(swa_kv, block, fnuz)
    )
    extra = _pack_fp8_ds_mla_cache(extra_kv, block, False)
    layer._compressed_kv_cache = lambda: extra
    layer.rotary_emb = SimpleNamespace(
        cos_sin_cache=_random_cos_sin_cache(seq_len, device)
    )
    layer.attn_sink = torch.linspace(-1, 1, heads, device=device)
    layer._index_source_prefix = "source"
    layer._static_forward_context = {"source": layer}
    layer._topk_ragged_cache = {}
    prefill_start = seq_len - n_prefill
    topk_rows = torch.randint(
        0, prefill_start // ratio, (n_tokens, topk), device=device, dtype=torch.int32
    )
    topk_rows[1::2, topk // 2 :] = -1
    layer.topk_indices_buffer = topk_rows
    swa_rows = torch.randint(
        0, seq_len, (n_decode, window), device=device, dtype=torch.int32
    )
    swa_lens = torch.full((n_decode,), window, device=device, dtype=torch.int32)
    ragged, indptr = build_ragged_indices_from_dense(swa_rows, swa_lens)
    starts = torch.tensor([*range(n_decode + 1), n_tokens], dtype=torch.int32)
    n_reqs = n_decode + 1
    swa_table = torch.arange(
        layer.swa_cache_layer.kv_cache.shape[0], device=device, dtype=torch.int32
    ).repeat(n_reqs, 1)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    gather_len = window + n_prefill
    compressed_rows = seq_len // ratio
    swa_metadata = SimpleNamespace(
        num_decodes=n_decode,
        num_decode_tokens=n_decode,
        num_prefills=1,
        num_prefill_tokens=n_prefill,
        block_size=block,
        slot_mapping=torch.empty(0, device=device, dtype=torch.int64),
        block_table=swa_table,
        decode_swa_indices=swa_rows,
        decode_swa_lens=swa_lens,
        decode_swa_ragged_indices=ragged,
        decode_swa_ragged_indptr=indptr,
        token_to_req_indices=torch.cat(
            (
                torch.arange(n_decode, device=device, dtype=torch.int32),
                torch.full((n_prefill,), n_decode, device=device, dtype=torch.int32),
            )
        ),
        is_valid_token=torch.ones(n_tokens, device=device, dtype=torch.bool),
        prefill_seq_lens=seq_lens,
        prefill_seq_lens_cpu=seq_lens.cpu(),
        prefill_gather_lens=torch.tensor([gather_len], device=device).int(),
        query_start_loc_cpu=starts,
        query_start_loc=starts.to(device),
        get_prefill_chunk_plan=lambda **kwargs: [
            (0, 1, compressed_rows, compressed_rows + gather_len)
        ],
    )
    compressed_metadata = SimpleNamespace(
        block_size=block * ratio,
        block_table=torch.arange(
            extra.shape[0], device=device, dtype=torch.int32
        ).repeat(n_reqs, 1),
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"swa": swa_metadata, "compressed": compressed_metadata},
        slot_mapping={},
    )
    positions = torch.cat(
        (
            torch.full((n_decode,), seq_len - 1, device=device),
            torch.arange(prefill_start, seq_len, device=device),
        )
    )

    # FP32 attention over the dequantized caches, then inverse RoPE.
    swa = _read_fp8_ds_mla_cache_rows(
        layer.swa_cache_layer.kv_cache,
        torch.arange(seq_len, device=device),
        block,
        fnuz,
    )
    comp = _read_fp8_ds_mla_cache_rows(
        extra, torch.arange(compressed_rows, device=device), block, False
    )

    def reference(q_rope: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n_tokens, heads, HEAD_DIM, device=device)
        for row in range(n_tokens):
            if row < n_decode:
                slots = swa_rows[row].tolist()
                candidates = topk_rows[row].tolist()
            else:
                pos = prefill_start + row - n_decode
                slots = list(range(max(0, pos - window + 1), pos + 1))
                candidates = topk_rows[row, : (pos + 1) // ratio].tolist()
            kv = torch.cat([swa[slots], comp[[i for i in candidates if i >= 0]]])
            logits = q_rope[row].float() @ kv.T * layer.scale
            probs = torch.softmax(torch.cat((logits, layer.attn_sink[:, None]), -1), -1)
            out[row] = probs[:, :-1] @ kv
        out = out.bfloat16().float()
        cos, sin = layer.rotary_emb.cos_sin_cache[positions, None].chunk(2, dim=-1)
        even, odd = out[..., 448::2].clone(), out[..., 449::2].clone()
        out[..., 448::2] = even * cos + odd * sin
        out[..., 449::2] = odd * cos - even * sin
        return out

    return layer, context, positions, reference


@requires_cdna3
@pytest.mark.parametrize("kernel_rotates_q", [False, True])
@torch.inference_mode()
def test_hip_mega_forward_mqa_decode_and_prefill(kernel_rotates_q: bool) -> None:
    """A decode+prefill step through the CDNA3 mega layer's forward_mqa. Q
    arrives rotated, or raw for the prefill kernel and forward_mqa to rotate.
    Prefill rows come back inverse-RoPE'd once: rotating them again, as a layer
    with ``_prefill_rotates_bf16 = False`` would, fails the check."""
    from vllm.forward_context import override_forward_context
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda")
    torch.manual_seed(31)
    init_workspace_manager(device)
    n_decode, n_prefill = 3, 37
    layer, context, positions, reference = _mega_layer(n_decode, n_prefill, device)
    q = torch.randn(n_decode + n_prefill, 16, HEAD_DIM, device=device).bfloat16()
    q_rope = _rope_q(q, positions, layer.rotary_emb.cos_sin_cache)
    expected = reference(q_rope)
    kv = torch.zeros(q.shape[0], HEAD_DIM, device=device, dtype=torch.bfloat16)
    layer._attn_rotates_q_from = 1 if kernel_rotates_q else None

    def run() -> torch.Tensor:
        out = layer._alloc_attn_out(q.shape[0], q)
        with override_forward_context(context):
            layer.forward_mqa(
                (q if kernel_rotates_q else q_rope).clone(), kv, positions, out
            )
        return out.float()

    torch.testing.assert_close(run(), expected, atol=3e-3, rtol=0.02)
    layer._prefill_rotates_bf16 = False
    with pytest.raises(AssertionError):
        torch.testing.assert_close(run(), expected, atol=3e-3, rtol=0.02)


@requires_gfx950
@pytest.mark.parametrize("with_extra", [False, True])
@pytest.mark.parametrize("main_use_fnuz", [False, True])
@pytest.mark.parametrize("with_sink", [False, True])
@torch.inference_mode()
def test_hip_mega_decode_mxfp8_output(
    with_extra: bool, main_use_fnuz: bool, with_sink: bool
) -> None:
    """Paged decode handles holes, empty rows and independently sized KV pages."""
    device = torch.device("cuda")
    torch.manual_seed(22)
    tokens, heads, width = 4, 8, 8 * HEAD_DIM
    q = torch.randn(tokens, heads, HEAD_DIM + 16, device=device).bfloat16()
    q = q[..., :HEAD_DIM]
    main_cache = _pack_fp8_ds_mla_cache(
        torch.randn(12, HEAD_DIM, device=device).bfloat16() * 0.25,
        4,
        False,
    )
    if main_use_fnuz:
        # The gfx950 writer only emits OCP; re-encode its values on the CPU.
        cache_cpu = main_cache.cpu().flatten(1)
        payload = cache_cpu[:, : 4 * 576].reshape(3, 4, 576)[..., :448]
        values = payload.contiguous().view(torch.float8_e4m3fn).float() * 0.5
        payload.copy_(values.to(torch.float8_e4m3fnuz).view(torch.uint8))
        cache_cpu[:, 4 * 576 :].reshape(3, 4, 8)[..., :7] += 1
        main_cache.copy_(cache_cpu.reshape_as(main_cache))
    main_rows = [[], [9, -1, 2, 99, 4, 2], [], [0, 11, 3]]
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)
    extra_cache = extra_indices = extra_indptr = extra_rows = None
    if with_extra:
        extra_cache = _pack_fp8_ds_mla_cache(
            torch.randn(6, HEAD_DIM, device=device).bfloat16() * 0.25, 2, False
        )
        extra_rows = [[], [4, 0], [5, 1, 2], []]
        extra_indices, extra_indptr = _ragged_from_rows(extra_rows, device)
    sink = torch.linspace(-1, 1, heads, device=device) if with_sink else None
    positions = torch.tensor(
        [3, 8, 9, 13], dtype=torch.int32 if with_sink else torch.int64, device=device
    )
    cos_sin = _random_cos_sin_cache(16, device)
    ref = _ref_sparse_decode_ragged(
        q,
        main_cache,
        [[slot for slot in row if 0 <= slot < 12] for row in main_rows],
        HEAD_DIM**-0.5,
        sink,
        4,
        extra_cache,
        extra_rows,
        main_use_fnuz,
        extra_block_size=2,
    ).float()
    cos, sin = cos_sin[positions.long(), None].chunk(2, dim=-1)
    even = ref[..., NOPE_HEAD_DIM::2].clone()
    odd = ref[..., NOPE_HEAD_DIM + 1 :: 2].clone()
    ref[..., NOPE_HEAD_DIM::2] = even * cos + odd * sin
    ref[..., NOPE_HEAD_DIM + 1 :: 2] = odd * cos - even * sin

    # Leave gaps between pages and guards around the output views.
    padded_cache = torch.zeros(6, 4, 584, dtype=torch.uint8, device=device)
    padded_cache[::2].copy_(main_cache)
    data = torch.full(
        (tokens + 2, width + 16), 1.0, dtype=torch.float8_e4m3fn, device=device
    )
    scales = torch.full(
        (tokens + 2, width // 32 + 4), 17, dtype=torch.uint8, device=device
    )
    out_data, out_scale = data[1:-1, :width], scales[1:-1, : width // 32]
    torch.ops._rocm_C.fused_norm_rope_attn_rope_cast_decode(
        q,
        padded_cache[::2],
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        sink,
        HEAD_DIM**-0.5,
        positions,
        cos_sin,
        main_use_fnuz,
        out_data,
        out_scale,
    )
    torch.testing.assert_close(
        _mxfp8_dequant(out_data, out_scale), ref.flatten(1), atol=3e-3, rtol=0.08
    )
    assert torch.count_nonzero(out_data[0].float()) == 0
    assert torch.all(data[[0, -1]].float() == 1)
    assert torch.all(data[:, width:].float() == 1)
    assert torch.all(scales[[0, -1]] == 17)
    assert torch.all(scales[:, width // 32 :] == 17)


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
@pytest.mark.parametrize("block_scales", [False, True])
@torch.inference_mode()
def test_rocm_mxfp8_wo_a_bmm(
    num_tokens: int, n_groups: int, block_scales: bool
) -> None:
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _as_block32_scale,
    )
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
    if block_scales:
        # A 32x32 checkpoint loads as per-row scales that repeat every 32 rows;
        # RocmDotScaledMxfp8LinearKernel compacts them before wo_a sees them.
        w_scale = _as_block32_scale(w_scale[::32].repeat_interleave(32, dim=0))
        assert w_scale is not None
    wo_a = SimpleNamespace(weight=w, weight_scale=w_scale)

    out = rocm_mxfp8_wo_a_bmm(a, a_scale, wo_a, n_groups, o_lora_rank)

    expected = torch.einsum(
        "tgd,grd->tgr",
        _mxfp8_dequant(a, a_scale).view(num_tokens, n_groups, group_dim),
        _mxfp8_dequant(
            w, w_scale.repeat_interleave(w.shape[0] // w_scale.shape[0], 0)
        ).view(n_groups, o_lora_rank, group_dim),
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


@requires_gfx950
@torch.inference_mode()
def test_dsv41_adaptive_mla_swa_metadata_graph_replay(monkeypatch) -> None:
    """Replay V4.1 ROCm decode after device-only query reallocation."""
    from tests.v1.attention.utils import create_vllm_config
    from vllm.models.deepseek_v41.amd.rocm import (
        DeepseekV41ROCMAiterMLASparseMetadataBuilder,
        DeepseekV41ROCMAiterSparseSWAMetadataBuilder,
        compute_global_topk_ragged_indices_and_indptr,
    )
    from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

    device = torch.device("cuda")
    num_reqs = 3
    upper_query_len = 6
    graph_tokens = num_reqs * upper_query_len
    block_size = 128
    compress_ratio = 2
    compressed_block_size = block_size // compress_ratio

    vllm_config = create_vllm_config(
        model_name="facebook/opt-125m",
        max_model_len=1024,
        block_size=block_size,
        max_num_seqs=num_reqs,
        max_num_batched_tokens=graph_tokens,
        hf_config_override={
            "compress_ratios": [0, 1, 2],
            "index_topk": 8,
            "sliding_window": 32,
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
        tokens_per_state=compress_ratio,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.uint8,
        sliding_window=32,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    mla_builder = DeepseekV41ROCMAiterMLASparseMetadataBuilder(
        mla_spec, ["c2a"], vllm_config, device
    )
    swa_builder = DeepseekV41ROCMAiterSparseSWAMetadataBuilder(
        swa_spec, ["c2a"], vllm_config, device
    )
    assert (
        mla_builder.get_cudagraph_support(vllm_config, mla_spec)
        == AttentionCGSupport.ALWAYS
    )
    assert (
        swa_builder.get_cudagraph_support(vllm_config, swa_spec)
        == AttentionCGSupport.ALWAYS
    )

    seq_lens = torch.tensor([100, 110, 120], dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    query_start_loc_cpu = torch.arange(
        0, graph_tokens + 1, upper_query_len, dtype=torch.int32
    )
    block_table = torch.arange(num_reqs, dtype=torch.int32, device=device).view(
        num_reqs, 1
    )

    def build_metadata(query_lens: list[int]):
        query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32, device=device)
        query_start_loc = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                query_lens_tensor.cumsum(0),
            ]
        )
        active_tokens = sum(query_lens)
        slot_mapping = torch.full((graph_tokens,), -1, dtype=torch.int64, device=device)
        slot_mapping[:active_tokens] = torch.arange(
            active_tokens, dtype=torch.int64, device=device
        )
        common = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=graph_tokens,
            max_query_len=upper_query_len,
            max_seq_len=int(seq_lens_cpu.max()),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
        )
        return (
            mla_builder.build_for_cudagraph_capture(common),
            swa_builder.build_for_cudagraph_capture(common),
        )

    mla_metadata, swa_metadata = build_metadata([6, 6, 6])
    metadata_ptrs = (
        mla_metadata.req_id_per_token.data_ptr(),
        mla_metadata.slot_mapping.data_ptr(),
        swa_metadata.token_to_req_indices.data_ptr(),
        swa_metadata.decode_swa_ragged_indices.data_ptr(),
        swa_metadata.decode_swa_ragged_indptr.data_ptr(),
    )

    topk_indices = (
        torch.arange(8, dtype=torch.int32, device=device)
        .expand(graph_tokens, -1)
        .contiguous()
    )
    torch.manual_seed(23)
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
    swa_cache = _pack_fp8_ds_mla_cache(
        torch.randn(
            num_reqs * block_size,
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
            num_reqs * compressed_block_size,
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

    def build_topk_ragged(mla_md, swa_md):
        return compute_global_topk_ragged_indices_and_indptr(
            topk_indices,
            swa_md.token_to_req_indices,
            mla_md.block_table,
            compressed_block_size,
            swa_md.is_valid_token,
        )

    def run_decode(mla_md, swa_md) -> None:
        topk_ragged_indices, topk_ragged_indptr, topk_lens = build_topk_ragged(
            mla_md, swa_md
        )
        mod.rocm_sparse_attn_decode(
            q=q,
            kv_cache=compressed_cache,
            swa_k_cache=swa_cache,
            swa_only=False,
            topk_indices=None,
            topk_lens=topk_lens,
            swa_indices=swa_md.decode_swa_indices,
            swa_lens=swa_md.decode_swa_lens,
            swa_ragged_indices=swa_md.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_md.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=attn_sink,
            scale=HEAD_DIM**-0.5,
            head_dim=HEAD_DIM,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            output=out,
            extra_cache_nan_free=True,
        )

    def reference(mla_md, swa_md) -> torch.Tensor:
        topk_ragged_indices, topk_ragged_indptr, _ = build_topk_ragged(mla_md, swa_md)
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
                topk_ragged_indices,
                topk_ragged_indptr,
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

    reallocated_mla, reallocated_swa = build_metadata([2, 6, 4])
    assert metadata_ptrs == (
        reallocated_mla.req_id_per_token.data_ptr(),
        reallocated_mla.slot_mapping.data_ptr(),
        reallocated_swa.token_to_req_indices.data_ptr(),
        reallocated_swa.decode_swa_ragged_indices.data_ptr(),
        reallocated_swa.decode_swa_ragged_indptr.data_ptr(),
    )
    expected_owners = torch.tensor(
        [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(reallocated_mla.req_id_per_token[:12], expected_owners)
    torch.testing.assert_close(
        reallocated_swa.token_to_req_indices[:12], expected_owners
    )
    assert reallocated_swa.is_valid_token[:12].all()
    assert not reallocated_swa.is_valid_token[12:].any()
    torch.testing.assert_close(
        reallocated_swa.decode_swa_lens[12:],
        torch.zeros(6, dtype=torch.int32, device=device),
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


def _combine_topk_swa_indices_torch(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    """The Torch path this kernel replaces, copied from before the switch.

    Kept verbatim so the reference cannot drift toward the kernel.
    """
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        _SPARSE_PREFILL_TOPK_ALIGNMENT,
    )

    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    # query_start_loc may have a non-zero base for a narrowed mixed batch.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=seq_lens.device), query_lens
    )
    query_starts = query_start_loc[:-1] - query_start_loc[0]
    token_offsets = torch.arange(num_tokens, device=seq_lens.device) - (
        torch.repeat_interleave(query_starts, query_lens)
    )
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets

    logical_topk_width = min(topk, topk_indices.shape[1])
    topk_lens = torch.minimum(
        (positions + 1) // compress_ratio,
        torch.full_like(positions, logical_topk_width),
    ).clamp_min(0)
    topk_offsets = torch.arange(logical_topk_width, device=seq_lens.device)
    topk_mask = topk_offsets[None, :] < topk_lens[:, None]
    topk_values = topk_indices[:, :logical_topk_width].to(torch.int32)
    topk_valid = topk_mask & (topk_values >= 0) & (topk_values < N)
    combined_indices[:, :logical_topk_width] = torch.where(
        topk_valid,
        topk_values + (M * req_ids).to(torch.int32)[:, None],
        -1,
    )

    swa_lens = torch.minimum(
        positions + 1, torch.full_like(positions, window_size)
    ).clamp_min(0)
    swa_offsets = torch.arange(window_size, device=seq_lens.device)
    swa_mask = swa_offsets[None, :] < swa_lens[:, None]
    swa_columns = topk_lens[:, None] + swa_offsets[None, :]
    gather_starts = seq_lens - gather_lens
    swa_values = (
        M * req_ids[:, None]
        + N
        + swa_offsets[None, :]
        + positions[:, None]
        - swa_lens[:, None]
        + 1
        - gather_starts[req_ids, None]
    ).to(torch.int32)
    rows = torch.arange(num_tokens, device=seq_lens.device)[:, None].expand_as(
        swa_columns
    )
    flat_dst = rows[swa_mask] * combined_topk + swa_columns[swa_mask]
    combined_indices.view(-1).index_copy_(0, flat_dst, swa_values[swa_mask])
    combined_lens.copy_((topk_lens + swa_lens).to(torch.int32))
    return combined_indices, combined_lens


def _v41_combine_case(case, window):
    dev = "cuda"
    ratio, topk = 2, 512
    if case == "seq_len_below_query_len":
        qsl = torch.tensor([0, 64], dtype=torch.int32, device=dev)
        sl = torch.tensor([32], dtype=torch.int32, device=dev)
        rows = 64
    elif case == "seq_len_zero":
        # The warmup batch: every pos is negative, so both lengths clamp to 0.
        qsl = torch.tensor([0, 32, 64], dtype=torch.int32, device=dev)
        sl = torch.zeros(2, dtype=torch.int32, device=dev)
        rows = 64
    elif case == "compress_ratio_zero":
        qsl = torch.tensor([0, 4, 8], dtype=torch.int32, device=dev)
        sl = torch.tensor([4, 4], dtype=torch.int32, device=dev)
        rows, ratio, topk = 8, 0, 0
    elif case == "nonzero_base":
        # Prefill rows of a mixed batch: query_start_loc starts past the decodes.
        qsl = torch.tensor([32, 64, 128], dtype=torch.int32, device=dev)
        sl = torch.tensor([48, 80], dtype=torch.int32, device=dev)
        rows = 96
    else:
        qsl = torch.tensor([0, 32, 96], dtype=torch.int32, device=dev)
        sl = torch.tensor([48, 80], dtype=torch.int32, device=dev)
        rows = 96
    # What the SWA builder gathers: the query plus up to window - 1 rows before it.
    gl = torch.minimum(sl, qsl[1:] - qsl[:-1] + window - 1)
    set_random_seed(0)
    ti = torch.randint(0, 4096, (rows, max(topk, 1)), dtype=torch.int32)
    if case == "invalid_topk":
        # Failed candidates stay -1 instead of pointing into another request.
        ti[:, ::7] = -1
    return ti.to(dev), qsl, sl, gl, window, ratio, topk, 100000, 4096


@requires_split_decode_arch
@pytest.mark.parametrize(
    "case,window",
    [
        ("seq_len_below_query_len", 128),
        ("seq_len_zero", 128),
        ("compress_ratio_zero", 128),
        ("nonzero_base", 128),
        ("invalid_topk", 128),
        ("plain", 64),
        ("plain", 128),
        ("plain", 256),
    ],
)
def test_v41_combine_topk_swa_indices_matches_torch(case, window) -> None:
    from vllm.models.deepseek_v41.amd.rocm import combine_topk_swa_indices

    args = _v41_combine_case(case, window)
    got_indices, got_lens = combine_topk_swa_indices(*args)
    ref_indices, ref_lens = _combine_topk_swa_indices_torch(*args)
    assert torch.equal(got_lens, ref_lens), (
        f"lens differ: ref={ref_lens.tolist()} got={got_lens.tolist()}"
    )
    assert torch.equal(got_indices, ref_indices)


def test_v41_rocm_combine_uses_shared_kernel() -> None:
    """ROCm prefill takes the shared combine, so it inherits the replay bound."""
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.models.deepseek_v41.common.ops import cache_utils

    assert rocm.combine_topk_swa_indices is cache_utils.combine_topk_swa_indices
