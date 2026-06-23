# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only used by ROCm"
)


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX950

        return bool(_ON_GFX950)
    except Exception:
        return False


# The flash-decode split-K decode path is only tuned for AMD gfx950; other
# architectures take the fallback decode kernel, so its tests are skipped there.
requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(),
    reason="split-K decode kernel is only tuned for AMD gfx950",
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
    kv: torch.Tensor, block_size: int, is_extra: bool = False
) -> torch.Tensor:
    assert kv.shape[-1] == HEAD_DIM
    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, 584),
        dtype=torch.uint8,
        device=kv.device,
    )
    cache_flat = cache.view(torch.uint8).flatten()
    kv_nope_fp8 = (
        kv[:, :NOPE_HEAD_DIM]
        .to(torch.float8_e4m3fn if is_extra else current_platform.fp8_dtype())
        .view(torch.uint8)
    )
    kv_rope_u8 = kv[:, NOPE_HEAD_DIM:].contiguous().view(torch.uint8)

    for slot in range(num_tokens):
        block_idx = slot // block_size
        pos = slot % block_size
        block_base = block_idx * cache.stride(0)
        token_base = block_base + pos * 576
        scale_base = block_base + block_size * 576 + pos * 8
        cache_flat[token_base : token_base + NOPE_HEAD_DIM].copy_(kv_nope_fp8[slot])
        cache_flat[
            token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
        ].copy_(kv_rope_u8[slot])
        cache_flat[scale_base : scale_base + 7].fill_(127)
    return cache


def _read_fp8_ds_mla_cache(
    cache: torch.Tensor, slot: int, block_size: int, is_extra: bool = False
) -> torch.Tensor:
    cache_flat = cache.view(torch.uint8).flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576

    nope_u8 = cache_flat[token_base : token_base + NOPE_HEAD_DIM]
    nope = nope_u8.view(
        torch.float8_e4m3fn if is_extra else current_platform.fp8_dtype()
    ).to(torch.float32)
    rope_u8 = cache_flat[
        token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
    ]
    rope = rope_u8.view(torch.bfloat16).to(torch.float32)
    return torch.cat([nope, rope])


def _ref_sparse_decode_ragged(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_rows: list[list[int]],
    scale: float,
    attn_sink: torch.Tensor | None,
    block_size: int,
    extra_cache: torch.Tensor | None = None,
    extra_rows: list[list[int]] | None = None,
) -> torch.Tensor:
    q_f32 = q.float()
    out = torch.empty_like(q_f32)

    for query_idx in range(q.shape[0]):
        row_kv = [
            _read_fp8_ds_mla_cache(main_cache, int(slot), block_size)
            for slot in main_rows[query_idx]
        ]
        if extra_cache is not None and extra_rows is not None:
            row_kv.extend(
                _read_fp8_ds_mla_cache(
                    extra_cache, int(slot), block_size, is_extra=True
                )
                for slot in extra_rows[query_idx]
            )

        kv = torch.stack(row_kv).to(q.device)
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


def _ref_combine_topk_swa_ragged(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_ragged = torch.tensor(
        [
            100,
            101,
            7,
            8,
            9,
            110,
            111,
            8,
            9,
            10,
            120,
            121,
            122,
            9,
            10,
            11,
            150,
            27,
            28,
            29,
            160,
            161,
            28,
            29,
            30,
        ],
        dtype=torch.int32,
        device=device,
    )
    expected_lens = torch.tensor([5, 5, 6, 4, 5], dtype=torch.int32, device=device)
    expected_indptr = torch.zeros(6, dtype=torch.int32, device=device)
    torch.cumsum(expected_lens, dim=0, out=expected_indptr[1:])
    return expected_ragged, expected_indptr, expected_lens


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
    q = torch.randn(2, 3, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_kv = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    extra_kv = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)
    main_indices = torch.tensor([0, 2, 4, 1], dtype=torch.int32, device=device)
    main_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    extra_indices = torch.tensor([1, 3, 0], dtype=torch.int32, device=device)
    extra_indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    attn_sink = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
    scale = HEAD_DIM**-0.5

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
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_combine_topk_swa_indices_ragged() -> None:
    from vllm.models.deepseek_v4.amd.rocm import (
        combine_topk_swa_indices_ragged,
    )

    device = torch.device("cuda")
    topk_indices = torch.tensor(
        [
            [100, 101, 102, 103],
            [110, 111, 112, 113],
            [120, 121, 122, 123],
            [130, 131, 132, 133],
            [140, 141, 142, 143],
        ],
        dtype=torch.int32,
        device=device,
    )
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([6, 4], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([4, 3], dtype=torch.int32, device=device)
    window_size = 3
    compress_ratio = 2
    topk = 4
    M = 20
    N = 8

    actual_ragged, actual_indptr, actual_lens = combine_topk_swa_indices_ragged(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )
    expected_ragged, expected_indptr, expected_lens = _ref_combine_topk_swa_ragged(
        device
    )

    torch.testing.assert_close(
        actual_ragged[: expected_ragged.numel()], expected_ragged
    )
    torch.testing.assert_close(actual_indptr, expected_indptr)
    torch.testing.assert_close(actual_lens, expected_lens)


@requires_gfx950
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

    # The chosen count always stays within the searched [1, 16] range, and a
    # zero-length workload never splits (no work to parallelize).
    for num_queries in (1, 4, 24, 224, 1024):
        splits = mod._decode_num_splits(
            num_queries, 1, avg_main_len=512.0, avg_extra_len=128.0
        )
        assert 1 <= splits <= 16
    assert mod._decode_num_splits(2, 1, avg_main_len=0.0, avg_extra_len=0.0) >= 1


@requires_gfx950
@pytest.mark.parametrize("num_splits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("with_extra", [True, False])
@pytest.mark.parametrize("with_sink", [True, False])
@torch.inference_mode()
def test_sparse_attn_decode_split_k_kernel(
    monkeypatch, num_splits: int, with_extra: bool, with_sink: bool
) -> None:
    """Flash-decode split-K decode path (partial + reduce kernels).

    This path is the gfx950 production path (``_ON_GFX950``), so the test only
    runs on gfx950. The split count is pinned so the partial/reduce kernels are
    exercised across split counts. ``num_splits=8`` drives splits past the
    shortest segment length, covering the empty-split edge case handled by the
    reduce kernel.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")
    torch.manual_seed(7)
    block_size = 4
    num_heads = 3

    main_rows = [[0, 2, 4, 6, 1, 3, 7, 5], [4, 1, 6, 0, 2]]
    num_queries = len(main_rows)
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = torch.randn(8, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size)
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)

    extra_rows: list[list[int]] | None = None
    extra_cache: torch.Tensor | None = None
    extra_indices: torch.Tensor | None = None
    extra_indptr: torch.Tensor | None = None
    if with_extra:
        rows = [[1, 3, 0, 5, 2, 4], [3, 0, 6]]
        extra_kv = torch.randn(7, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
        extra_rows = rows
        extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)
        extra_indices, extra_indptr = _ragged_from_rows(rows, device)

    attn_sink = (
        torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    # Pin the split count so each parametrized value is exercised deterministically.
    monkeypatch.setattr(mod, "_decode_num_splits", lambda *args, **kwargs: num_splits)

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
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


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
