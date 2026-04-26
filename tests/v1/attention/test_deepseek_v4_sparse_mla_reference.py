# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the DeepSeek V4 sparse MLA reference path."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers import (
    deepseek_v4_attention as deepseek_v4_attention_module,
)
from vllm.model_executor.layers.deepseek_v4_attention import (
    _deepseek_v4_fp8_einsum_config,
    _sparse_mla_prefill_workspace_bounds,
    deepseek_v4_fp8_einsum,
)
from vllm.utils.deep_gemm import fp8_einsum
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    disable_sparse_mla_reference_cudagraphs_if_enabled,
    sparse_mla_reference_topk_chunk_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk,
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead,
    accumulate_gathered_sparse_mla_attention_chunk,
    accumulate_indexed_sparse_mla_attention_chunk,
    build_combined_sparse_mla_decode_valid_mask,
    finish_gathered_sparse_mla_attention,
    finish_sparse_mla_attention_with_sink,
    finish_two_sparse_mla_attention_states_with_sink,
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    matmul_sparse_mla_attention_with_sink,
    merge_sparse_mla_subset_with_sink,
    merge_two_sparse_mla_subsets_with_sink,
    sparse_mla_decode_head_block_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_reference import (
    accumulate_reference_attention_chunk,
    finish_reference_attention_no_sink,
    merge_reference_attention_with_sink,
    new_reference_attention_state,
    reference_attention_no_sink,
    reference_sparse_mla_prefill,
    sink_aware_reference_attention,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.attention.ops.deepseek_v4_ops import (
    dequantize_and_gather_k_cache,
    dequantize_combined_sparse_mla_decode_kv,
    dequantize_global_slots_k_cache,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fp8_einsum import (
    deepseek_v4_sm12_fp8_einsum,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2


class _FakeWorkspaceManager:

    def get_simultaneous(self, *specs):
        return tuple(torch.empty(shape, dtype=dtype) for shape, dtype in specs)


def test_triton_sparse_mla_default_topk_chunk_size(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", raising=False)

    assert sparse_mla_reference_topk_chunk_size() == 512


def test_sparse_mla_prefill_workspace_bounds_use_active_prefill_lengths() -> None:
    seq_lens_cpu = torch.tensor([15_000, 2_048], dtype=torch.int32)
    gather_lens_cpu = torch.tensor([15_000, 2_048], dtype=torch.int32)

    compressed_region_size, row_stride = _sparse_mla_prefill_workspace_bounds(
        seq_lens_cpu=seq_lens_cpu,
        gather_lens_cpu=gather_lens_cpu,
        compress_ratio=4,
        swa_only=False,
    )

    assert compressed_region_size == 3_750
    assert row_stride == 18_750


def test_sparse_mla_prefill_workspace_bounds_for_swa_only() -> None:
    seq_lens_cpu = torch.tensor([15_000], dtype=torch.int32)
    gather_lens_cpu = torch.tensor([15_000], dtype=torch.int32)

    compressed_region_size, row_stride = _sparse_mla_prefill_workspace_bounds(
        seq_lens_cpu=seq_lens_cpu,
        gather_lens_cpu=gather_lens_cpu,
        compress_ratio=1,
        swa_only=True,
    )

    assert compressed_region_size == 0
    assert row_stride == 15_000


@pytest.mark.parametrize(
    ("num_decode_tokens", "expected_head_block_size"),
    [
        (0, 1),
        (1, 1),
        (4, 1),
        (5, 2),
        (8, 2),
        (15, 2),
        (16, 4),
        (32, 4),
    ],
)
def test_triton_sparse_mla_decode_head_block_size(
    num_decode_tokens: int,
    expected_head_block_size: int,
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE", raising=False)

    assert (
        sparse_mla_decode_head_block_size(num_decode_tokens)
        == expected_head_block_size
    )


@pytest.mark.parametrize("configured_head_block_size", ["1", "2", "4"])
def test_triton_sparse_mla_decode_head_block_size_env_override(
    configured_head_block_size: str,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE",
        configured_head_block_size,
    )

    assert sparse_mla_decode_head_block_size(1) == int(configured_head_block_size)
    assert sparse_mla_decode_head_block_size(32) == int(configured_head_block_size)


@pytest.mark.parametrize("configured_head_block_size", ["0", "3", "invalid"])
def test_triton_sparse_mla_decode_head_block_size_ignores_invalid_env_override(
    configured_head_block_size: str,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE",
        configured_head_block_size,
    )

    assert sparse_mla_decode_head_block_size(8) == 2


def test_swa_mtp_decode_reference_uses_global_swa_slots(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fail_paged_attention_with_sink_multihead(**kwargs) -> None:
        raise AssertionError("MTP SWA decode must use explicit SWA indices")

    def fake_accumulate_global_slots(**kwargs) -> None:
        captured["slot_ids"] = kwargs["slot_ids"]
        captured["lens"] = kwargs["lens"]

    def fake_finish_with_sink(*args, **kwargs) -> None:
        kwargs["output"].zero_()

    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "current_workspace_manager",
        lambda: _FakeWorkspaceManager(),
    )
    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "fp8ds_paged_sparse_mla_attention_with_sink_multihead",
        fail_paged_attention_with_sink_multihead,
    )
    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead",
        fake_accumulate_global_slots,
    )
    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "finish_sparse_mla_attention_with_sink",
        fake_finish_with_sink,
    )

    attention = SimpleNamespace(
        num_heads=2,
        scale=0.1,
        attn_sink=torch.zeros(2, dtype=torch.float32),
    )
    swa_indices = torch.arange(48, dtype=torch.int32).reshape(6, 1, 8)
    swa_lens = torch.tensor([2, 3, 4, 2, 3, 4], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=6,
        decode_swa_lens=swa_lens,
        decode_swa_indices=swa_indices,
        seq_lens=torch.tensor([11, 22], dtype=torch.int32),
        block_table=torch.empty((2, 4), dtype=torch.int32),
        block_size=256,
        token_to_req_indices=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
    )

    deepseek_v4_attention_module.DeepseekV4MLAAttention._forward_sparse_mla_swa_decode_reference(
        attention,
        q=torch.empty((6, 1, 2, 512), dtype=torch.bfloat16),
        swa_k_cache=torch.empty((1, 256, 584), dtype=torch.uint8),
        swa_metadata=metadata,
        output=torch.empty((6, 2, 512), dtype=torch.bfloat16),
    )

    torch.testing.assert_close(captured["slot_ids"], swa_indices)
    torch.testing.assert_close(captured["lens"], swa_lens)


def test_compressed_mtp_decode_reference_uses_global_swa_slots(monkeypatch) -> None:
    captured: list[torch.Tensor] = []

    def fail_matmul_decode(**kwargs) -> None:
        raise AssertionError("MTP compressed decode must not stage paged SWA")

    def fail_direct_global_paged(**kwargs) -> None:
        raise AssertionError("MTP compressed decode must not use paged SWA window")

    def fake_accumulate_global_slots(**kwargs) -> None:
        captured.append(kwargs["slot_ids"])

    def fake_finish_two_states(*args, **kwargs) -> None:
        kwargs["output"].zero_()

    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "current_workspace_manager",
        lambda: _FakeWorkspaceManager(),
    )
    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "dequantize_combined_sparse_mla_decode_kv",
        fail_matmul_decode,
    )
    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "fp8ds_global_paged_sparse_mla_attention_with_sink_multihead",
        fail_direct_global_paged,
    )
    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead",
        fake_accumulate_global_slots,
    )
    monkeypatch.setattr(
        deepseek_v4_attention_module,
        "finish_two_sparse_mla_attention_states_with_sink",
        fake_finish_two_states,
    )

    attention = SimpleNamespace(
        num_heads=2,
        scale=0.1,
        attn_sink=torch.zeros(2, dtype=torch.float32),
        compress_ratio=4,
    )
    swa_indices = torch.arange(48, dtype=torch.int32).reshape(6, 1, 8)
    topk_slot_ids = torch.arange(24, dtype=torch.int32).reshape(6, 1, 4)
    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=6,
        decode_swa_lens=torch.full((6,), 3, dtype=torch.int32),
        decode_swa_indices=swa_indices,
        seq_lens=torch.tensor([11, 22], dtype=torch.int32),
        block_table=torch.empty((2, 4), dtype=torch.int32),
        block_size=256,
        token_to_req_indices=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
    )

    deepseek_v4_attention_module.DeepseekV4MLAAttention._forward_sparse_mla_compressed_decode_reference(
        attention,
        q=torch.empty((6, 1, 2, 512), dtype=torch.bfloat16),
        compressed_k_cache=torch.empty((1, 64, 584), dtype=torch.uint8),
        swa_k_cache=torch.empty((1, 256, 584), dtype=torch.uint8),
        topk_indices=topk_slot_ids,
        topk_lens=torch.full((6,), 4, dtype=torch.int32),
        swa_metadata=swa_metadata,
        attn_metadata=SimpleNamespace(block_size=256),
        output=torch.empty((6, 2, 512), dtype=torch.bfloat16),
    )

    assert len(captured) == 2
    torch.testing.assert_close(captured[0], topk_slot_ids[:, 0])
    torch.testing.assert_close(captured[1], swa_indices)


@pytest.mark.parametrize(
    ("capability_major", "expected_recipe", "expected_tma_aligned"),
    [
        (9, (1, 128, 128), False),
        (10, (1, 1, 128), True),
        (12, (1, 128, 128), False),
    ],
)
def test_deepseek_v4_fp8_einsum_config_for_sm12x(
    capability_major: int,
    expected_recipe: tuple[int, int, int],
    expected_tma_aligned: bool,
) -> None:
    assert _deepseek_v4_fp8_einsum_config(capability_major) == (
        expected_recipe,
        expected_tma_aligned,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("use_e8m0_scale", [False, True])
def test_deepseek_v4_sm12_triton_fp8_einsum_matches_deepgemm_reference(
    use_e8m0_scale: bool,
) -> None:
    if use_e8m0_scale and not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch does not expose float8_e8m0fnu")
    torch.manual_seed(0)
    num_tokens = 17
    num_groups = 4
    hidden_size = 4096
    out_rank = 1024
    recipe = (1, 128, 128)

    a_backing = torch.empty(
        (num_groups, num_tokens, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    a = a_backing.transpose(0, 1)
    a_scale_backing = torch.empty(
        (num_groups, num_tokens, hidden_size // 128),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.01, 0.02)
    a_scale = a_scale_backing.transpose(0, 1)
    b_flat = torch.empty(
        (num_groups * out_rank, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    b = b_flat.view(num_groups, out_rank, hidden_size)
    if use_e8m0_scale:
        scale_choices = torch.tensor(
            [0.00390625, 0.0078125, 0.015625, 0.03125],
            device="cuda",
            dtype=torch.float32,
        )
        scale_indices = torch.randint(
            0,
            len(scale_choices),
            (num_groups * (out_rank // 128), hidden_size // 128),
            device="cuda",
        )
        b_scale_flat = scale_choices[scale_indices].to(torch.float8_e8m0fnu)
        b_scale_ref_flat = b_scale_flat.to(torch.float32)
    else:
        b_scale_flat = torch.empty(
            (num_groups * (out_rank // 128), hidden_size // 128),
            device="cuda",
            dtype=torch.float32,
        ).uniform_(0.01, 0.02)
        b_scale_ref_flat = b_scale_flat
    b_scale_ref = b_scale_ref_flat.view(
        num_groups, out_rank // 128, hidden_size // 128
    )
    expected = torch.empty(
        (num_tokens, num_groups, out_rank),
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual = torch.empty_like(expected)

    fp8_einsum(
        "bhr,hdr->bhd",
        (a, a_scale),
        (b, b_scale_ref),
        expected,
        recipe=recipe,
    )
    deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b_flat,
        b_scale_flat,
        actual,
        "bhr,hdr->bhd",
        list(recipe),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_deepseek_v4_sm12_triton_fp8_einsum_primitive_matches_reference() -> None:
    torch.manual_seed(0)
    num_tokens = 17
    num_groups = 4
    hidden_size = 4096
    out_rank = 1024
    recipe = (1, 128, 128)

    a_backing = torch.empty(
        (num_groups, num_tokens, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    a = a_backing.transpose(0, 1)
    a_scale_backing = torch.empty(
        (num_groups, num_tokens, hidden_size // 128),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.01, 0.02)
    a_scale = a_scale_backing.transpose(0, 1)
    b_flat = torch.empty(
        (num_groups * out_rank, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    b = b_flat.view(num_groups, out_rank, hidden_size)
    b_scale_flat = torch.empty(
        (num_groups * (out_rank // 128), hidden_size // 128),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.01, 0.02)
    b_scale = b_scale_flat.view(num_groups, out_rank // 128, hidden_size // 128)
    expected = torch.empty(
        (num_tokens, num_groups, out_rank),
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual = torch.empty_like(expected)

    fp8_einsum("bhr,hdr->bhd", (a, a_scale), (b, b_scale), expected, recipe=recipe)
    deepseek_v4_sm12_fp8_einsum(a, a_scale, b, b_scale, actual)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("num_groups", [1, 2, 4])
def test_deepseek_v4_sm12_triton_fp8_einsum_supports_tp_local_group_counts(
    num_groups: int,
) -> None:
    torch.manual_seed(18 + num_groups)
    num_tokens = 5
    hidden_size = 4096
    out_rank = 1024
    recipe = (1, 128, 128)

    a_backing = torch.empty(
        (num_groups, num_tokens, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    a = a_backing.transpose(0, 1)
    a_scale_backing = torch.empty(
        (num_groups, num_tokens, hidden_size // 128),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.01, 0.02)
    a_scale = a_scale_backing.transpose(0, 1)
    b_flat = torch.empty(
        (num_groups * out_rank, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    b = b_flat.view(num_groups, out_rank, hidden_size)
    b_scale_flat = torch.empty(
        (num_groups * (out_rank // 128), hidden_size // 128),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.01, 0.02)
    b_scale = b_scale_flat.view(num_groups, out_rank // 128, hidden_size // 128)
    expected = torch.empty(
        (num_tokens, num_groups, out_rank),
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual = torch.empty_like(expected)

    fp8_einsum("bhr,hdr->bhd", (a, a_scale), (b, b_scale), expected, recipe=recipe)
    deepseek_v4_sm12_fp8_einsum(a, a_scale, b, b_scale, actual)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _masked_scores(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q_bhd = q[:, 0].float() if q.dim() == 4 else q.float()
    scores = torch.einsum("bhd,btd->bht", q_bhd, kv.float()) * scale
    return scores.masked_fill(~valid_tokens[:, None, :], float("-inf"))


def _golden_no_sink_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _masked_scores(q, kv, valid_tokens, scale)
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - lse[:, :, None])
    weights = torch.where(
        valid_tokens[:, None, :],
        weights,
        torch.zeros((), dtype=weights.dtype, device=weights.device),
    )
    weights = torch.nan_to_num(weights)
    output = torch.einsum("bht,btd->bhd", weights, kv.float())
    valid = valid_tokens.any(dim=-1)
    output = torch.where(
        valid[:, None, None],
        output,
        torch.zeros((), dtype=output.dtype, device=output.device),
    )
    return output, lse


def _golden_sink_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    scores = _masked_scores(q, kv, valid_tokens, scale)
    sink = attn_sink[None, :].float()
    score_max = scores.amax(dim=-1)
    merge_max = torch.maximum(score_max, sink)

    weights = torch.exp(scores - merge_max[:, :, None])
    weights = torch.where(
        valid_tokens[:, None, :],
        weights,
        torch.zeros((), dtype=weights.dtype, device=weights.device),
    )
    weights = torch.nan_to_num(weights)

    sink_weight = torch.exp(sink - merge_max)
    sink_weight = torch.nan_to_num(sink_weight)
    denom = weights.sum(dim=-1) + sink_weight
    numerator = torch.einsum("bht,btd->bhd", weights, kv.float())
    return numerator / denom[:, :, None]


def _chunked_no_sink_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_bhd, max_score, denom, acc = new_reference_attention_state(q)
    for chunk_start in range(0, kv.shape[1], chunk_size):
        chunk_end = min(chunk_start + chunk_size, kv.shape[1])
        max_score, denom, acc = accumulate_reference_attention_chunk(
            q_bhd=q_bhd,
            kv=kv[:, chunk_start:chunk_end],
            valid_tokens=valid_tokens[:, chunk_start:chunk_end],
            max_score=max_score,
            denom=denom,
            acc=acc,
            scale=scale,
        )
    return finish_reference_attention_no_sink(max_score, denom, acc)


def _write_fp8_ds_mla_token(
    k_cache: torch.Tensor,
    slot: int,
    block_size: int,
) -> torch.Tensor:
    block_idx = slot // block_size
    block_offset = slot % block_size

    values = (
        (torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17)
        - 8
    ) / 16.0
    values = values + float(slot) / 32.0
    scale_exponents = torch.tensor(
        [-2, -1, 0, 1, 2, -2, 1],
        device=k_cache.device,
        dtype=torch.float32,
    )
    scales = torch.exp2(scale_exponents)
    scale_per_dim = scales.repeat_interleave(64)

    fp8_values = (values / scale_per_dim).to(torch.float8_e4m3fn)
    expected_nope = fp8_values.float() * scale_per_dim
    rope = (
        torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device)
        + float(slot) / 16.0
    ).to(torch.bfloat16)

    flat_block = k_cache[block_idx].view(-1)
    token_data_start = block_offset * _TOKEN_DATA_SIZE
    token_scale_start = block_size * _TOKEN_DATA_SIZE + block_offset * _SCALE_DIM
    flat_block[token_data_start : token_data_start + _FP8_DIM] = fp8_values.view(
        torch.uint8
    )
    flat_block[
        token_data_start + _FP8_DIM : token_data_start + _TOKEN_DATA_SIZE
    ] = rope.view(torch.uint8)

    encoded_scales = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat_block[token_scale_start : token_scale_start + encoded_scales.numel()] = (
        encoded_scales
    )
    flat_block[
        token_scale_start + encoded_scales.numel() : token_scale_start + _SCALE_DIM
    ] = 127

    return torch.cat([expected_nope, rope.float()]).to(torch.bfloat16)


def test_reference_attention_no_sink_matches_logsumexp() -> None:
    torch.manual_seed(0)
    scale = 0.25
    q = torch.randn(3, 4, 5)
    kv = torch.randn(3, 6, 5)
    valid_tokens = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, False, False, False, False, False],
            [True, False, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    output, lse = reference_attention_no_sink(q, kv, valid_tokens, scale)
    expected_output, expected_lse = _golden_no_sink_attention(
        q,
        kv,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)



def test_reference_attention_ignores_nan_kv_for_invalid_tokens() -> None:
    torch.manual_seed(24)
    q = torch.randn(2, 1, 3, 8)
    kv = torch.randn(2, 4, 8)
    kv[:, 2:] = float("nan")
    valid_tokens = torch.tensor(
        [[True, True, False, False], [True, False, False, False]],
        dtype=torch.bool,
    )

    output, lse = reference_attention_no_sink(
        q=q,
        kv=kv,
        valid_tokens=valid_tokens,
        scale=0.125,
    )

    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()


def test_sink_aware_reference_attention_matches_dense_golden() -> None:
    torch.manual_seed(1)
    scale = 0.125
    q = torch.randn(3, 1, 4, 5)
    kv = torch.randn(3, 6, 5)
    valid_tokens = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, False, False, False, False, False],
            [False, True, True, False, True, True],
        ],
        dtype=torch.bool,
    )
    sink = torch.tensor([-1.0, 0.25, 1.5, -0.5])
    output = torch.empty(3, 4, 5)
    sink_aware_reference_attention(q, kv, valid_tokens, scale, sink, output)
    expected = _golden_sink_attention(q, kv, valid_tokens, scale, sink)

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_lse_merge_with_sink_matches_concatenated_attention() -> None:
    torch.manual_seed(2)
    scale = 0.2
    q = torch.randn(4, 3, 7)
    compressed_kv = torch.randn(4, 5, 7)
    swa_kv = torch.randn(4, 3, 7)
    compressed_kv[:, 1] = compressed_kv[:, 0]
    swa_kv[:, 2] = compressed_kv[:, 0]
    compressed_valid = torch.tensor(
        [
            [True, True, False, True, False],
            [False, False, False, False, False],
            [True, False, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    swa_valid = torch.tensor(
        [
            [True, False, True],
            [True, True, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    sink = torch.tensor([-0.25, 0.75, 1.25])
    output = torch.empty(4, 3, 7)
    comp_output, comp_lse = reference_attention_no_sink(
        q,
        compressed_kv,
        compressed_valid,
        scale,
    )
    swa_output, swa_lse = reference_attention_no_sink(q, swa_kv, swa_valid, scale)
    merge_reference_attention_with_sink(
        subset_outputs=[comp_output, swa_output],
        subset_lses=[comp_lse, swa_lse],
        attn_sink=sink,
        output=output,
    )

    expected = _golden_sink_attention(
        q,
        torch.cat([compressed_kv, swa_kv], dim=1),
        torch.cat([compressed_valid, swa_valid], dim=1),
        scale,
        sink,
    )
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(output[3], torch.zeros_like(output[3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_lse_merge_with_sink_matches_reference() -> None:
    torch.manual_seed(5)
    comp_output = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    swa_output = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    comp_lse = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    swa_lse = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    comp_lse[1, 2] = float("-inf")
    swa_lse[2, 1] = float("-inf")
    sink = torch.tensor([-0.5, 0.25, 1.0, -1.5], device="cuda")

    output = torch.empty(3, 4, 9, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    merge_two_sparse_mla_subsets_with_sink(
        subset0_output=comp_output,
        subset0_lse=comp_lse,
        subset1_output=swa_output,
        subset1_lse=swa_lse,
        attn_sink=sink,
        output=output,
    )
    merge_reference_attention_with_sink(
        subset_outputs=[comp_output, swa_output],
        subset_lses=[comp_lse, swa_lse],
        attn_sink=sink,
        output=expected,
    )

    torch.testing.assert_close(output.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_single_lse_merge_with_sink_matches_reference() -> None:
    torch.manual_seed(14)
    subset_output = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    subset_lse = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    subset_lse[1, 2] = float("-inf")
    sink = torch.tensor([-0.5, 0.25, 1.0, -1.5], device="cuda")

    output = torch.empty(3, 4, 9, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    merge_sparse_mla_subset_with_sink(
        subset_output=subset_output,
        subset_lse=subset_lse,
        attn_sink=sink,
        output=output,
    )
    merge_reference_attention_with_sink(
        subset_outputs=[subset_output],
        subset_lses=[subset_lse],
        attn_sink=sink,
        output=expected,
    )

    torch.testing.assert_close(output.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_finish_with_sink_matches_finish_then_merge_reference() -> None:
    torch.manual_seed(18)
    max_score = torch.randn(4, 3, device="cuda", dtype=torch.float32)
    denom = torch.rand(4, 3, device="cuda", dtype=torch.float32) + 0.1
    denom[1, 2] = 0.0
    max_score[1, 2] = float("-inf")
    acc = torch.randn(4, 3, 17, device="cuda", dtype=torch.float32)
    sink = torch.tensor(
        [-0.5, 0.25, 1.0, -float("inf"), -float("inf")],
        device="cuda",
        dtype=torch.float32,
    )

    output = torch.full((4, 5, 17), -7.0, device="cuda", dtype=torch.bfloat16)
    finish_sparse_mla_attention_with_sink(max_score, denom, acc, sink, output)

    subset_output = torch.empty_like(acc)
    subset_lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=subset_output,
        lse=subset_lse,
    )
    expected = torch.empty(4, 3, 17, device="cuda", dtype=torch.bfloat16)
    merge_reference_attention_with_sink(
        subset_outputs=[subset_output],
        subset_lses=[subset_lse],
        attn_sink=sink[:3],
        output=expected,
    )

    torch.testing.assert_close(
        output[:, :3].float(), expected.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        output[:, 3:].float(),
        torch.full_like(output[:, 3:].float(), -7.0),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_finish_with_sink_returns_zero_when_no_tokens_or_sink() -> None:
    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.full((2, 3, 17), float("nan"), device="cuda")
    sink = torch.full((3,), float("-inf"), device="cuda")

    single_output = torch.full(
        (2, 3, 17), 7.0, device="cuda", dtype=torch.bfloat16
    )
    finish_sparse_mla_attention_with_sink(
        max_score,
        denom,
        acc,
        sink,
        output=single_output,
    )
    torch.testing.assert_close(
        single_output.float(),
        torch.zeros_like(single_output.float()),
        rtol=0,
        atol=0,
    )

    two_output = torch.full((2, 3, 17), 7.0, device="cuda", dtype=torch.bfloat16)
    finish_two_sparse_mla_attention_states_with_sink(
        max_score,
        denom,
        acc,
        max_score,
        denom,
        acc,
        sink,
        output=two_output,
    )
    torch.testing.assert_close(
        two_output.float(),
        torch.zeros_like(two_output.float()),
        rtol=0,
        atol=0,
    )


def test_triton_finish_two_states_with_sink_matches_finish_then_merge() -> None:
    torch.manual_seed(22)
    comp_max = torch.randn(4, 3, device="cuda", dtype=torch.float32)
    comp_denom = torch.rand(4, 3, device="cuda", dtype=torch.float32) + 0.1
    comp_acc = torch.randn(4, 3, 17, device="cuda", dtype=torch.float32)
    swa_max = torch.randn(4, 3, device="cuda", dtype=torch.float32)
    swa_denom = torch.rand(4, 3, device="cuda", dtype=torch.float32) + 0.1
    swa_acc = torch.randn(4, 3, 17, device="cuda", dtype=torch.float32)
    sink = torch.tensor(
        [-0.5, 0.25, 1.0, -float("inf"), -float("inf")],
        device="cuda",
        dtype=torch.float32,
    )

    comp_denom[0, 1] = 0.0
    comp_max[0, 1] = float("-inf")
    swa_denom[2, 0] = 0.0
    swa_max[2, 0] = float("-inf")
    comp_denom[3, 2] = 0.0
    comp_max[3, 2] = float("-inf")
    swa_denom[3, 2] = 0.0
    swa_max[3, 2] = float("-inf")

    output = torch.full((4, 5, 17), -7.0, device="cuda", dtype=torch.bfloat16)
    finish_two_sparse_mla_attention_states_with_sink(
        comp_max,
        comp_denom,
        comp_acc,
        swa_max,
        swa_denom,
        swa_acc,
        sink,
        output,
    )

    comp_output = torch.empty_like(comp_acc)
    comp_lse = torch.empty_like(comp_max)
    swa_output = torch.empty_like(swa_acc)
    swa_lse = torch.empty_like(swa_max)
    finish_gathered_sparse_mla_attention(
        comp_max,
        comp_denom,
        comp_acc,
        comp_output,
        comp_lse,
    )
    finish_gathered_sparse_mla_attention(
        swa_max,
        swa_denom,
        swa_acc,
        swa_output,
        swa_lse,
    )
    expected = torch.empty(4, 3, 17, device="cuda", dtype=torch.bfloat16)
    merge_two_sparse_mla_subsets_with_sink(
        subset0_output=comp_output,
        subset0_lse=comp_lse,
        subset1_output=swa_output,
        subset1_lse=swa_lse,
        attn_sink=sink[:3],
        output=expected,
    )

    torch.testing.assert_close(
        output[:, :3].float(), expected.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        output[:, 3:].float(),
        torch.full_like(output[:, 3:].float(), -7.0),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("head_dim", [16, 512])
def test_triton_gathered_attention_chunk_matches_reference(head_dim: int) -> None:
    torch.manual_seed(6)
    scale = 0.125
    q = torch.randn(2, 1, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    q_active = q[:, :, :3]
    kv = torch.randn(2, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    slot_ids = torch.tensor(
        [
            [0, 1, -1, 3, 4],
            [5, -1, 7, 8, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, head_dim), device="cuda")

    accumulate_gathered_sparse_mla_attention_chunk(
        q=q,
        kv=kv[:, :2],
        slot_ids=slot_ids[:, :2],
        lens=lens,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    accumulate_gathered_sparse_mla_attention_chunk(
        q=q,
        kv=kv[:, 2:],
        slot_ids=slot_ids[:, 2:],
        lens=lens,
        candidate_offset=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid_tokens = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = reference_attention_no_sink(
        q_active,
        kv,
        valid_tokens,
        scale,
    )
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_gathered_attention_chunk_matches_reference_without_slot_ids() -> None:
    torch.manual_seed(8)
    scale = 0.2
    q = torch.randn(3, 1, 2, 32, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(3, 6, 32, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([6, 3, 0], dtype=torch.int32, device="cuda")
    max_score = torch.full((3, 2), float("-inf"), device="cuda")
    denom = torch.zeros((3, 2), device="cuda")
    acc = torch.zeros((3, 2, 32), device="cuda")

    accumulate_gathered_sparse_mla_attention_chunk(
        q=q,
        kv=kv,
        slot_ids=None,
        lens=lens,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    offsets = torch.arange(kv.shape[1], device="cuda")
    valid_tokens = offsets[None, :] < lens[:, None]
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        kv,
        valid_tokens,
        scale,
    )
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_dequantize_global_slots_k_cache_fp8_ds_mla_layout() -> None:
    block_size = 4
    num_blocks = 2
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(k_cache, slot, block_size)
        for slot in (0, 3, 4)
    }
    slot_ids = torch.tensor(
        [
            [0, 3, -1, 4],
            [4, 0, 3, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    output = torch.empty(2, 4, 512, dtype=torch.bfloat16, device="cuda")
    dequantize_global_slots_k_cache(output, k_cache, slot_ids, block_size)

    expected = torch.zeros_like(output)
    for token_idx in range(slot_ids.shape[0]):
        for topk_idx in range(slot_ids.shape[1]):
            slot = int(slot_ids[token_idx, topk_idx].item())
            if slot >= 0:
                expected[token_idx, topk_idx] = expected_by_slot[slot]

    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)

    output_from_3d_indices = torch.empty_like(output)
    dequantize_global_slots_k_cache(
        output_from_3d_indices,
        k_cache,
        slot_ids.unsqueeze(1),
        block_size,
    )
    torch.testing.assert_close(
        output_from_3d_indices.float(),
        expected.float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_dequantize_combined_sparse_mla_decode_kv_writes_direct_views() -> None:
    compressed_block_size = 4
    swa_block_size = 4
    compressed_cache = torch.zeros(
        2,
        compressed_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    swa_cache = torch.zeros(
        3,
        swa_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    for slot in (0, 3, 4):
        _write_fp8_ds_mla_token(compressed_cache, slot, compressed_block_size)
    for slot in (0, 1, 2, 3, 4):
        _write_fp8_ds_mla_token(swa_cache, slot, swa_block_size)

    compressed_slot_ids = torch.tensor(
        [[0, 3, -1], [4, 0, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    swa_lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [[0, 1, 2], [2, 0, 1]],
        dtype=torch.int32,
        device="cuda",
    )

    combined = torch.full(
        (2, 6, 512),
        -7,
        dtype=torch.bfloat16,
        device="cuda",
    )
    dequantize_combined_sparse_mla_decode_kv(
        combined,
        compressed_cache,
        compressed_slot_ids,
        compressed_block_size,
        swa_cache,
        seq_lens,
        swa_lens,
        block_table,
        swa_block_size,
    )

    expected_comp = torch.empty(2, 3, 512, dtype=torch.bfloat16, device="cuda")
    expected_swa = torch.full(
        (2, 3, 512),
        -7,
        dtype=torch.bfloat16,
        device="cuda",
    )
    dequantize_global_slots_k_cache(
        expected_comp,
        compressed_cache,
        compressed_slot_ids,
        compressed_block_size,
    )
    dequantize_and_gather_k_cache(
        expected_swa,
        swa_cache,
        seq_lens=seq_lens,
        gather_lens=swa_lens,
        block_table=block_table,
        block_size=swa_block_size,
        offset=0,
    )
    expected = torch.full_like(combined, -7)
    expected[:, :3].copy_(expected_comp)
    expected[:, 3:].copy_(expected_swa)

    torch.testing.assert_close(combined.float(), expected.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_fp8ds_global_slots_attention_chunk_matches_reference() -> None:
    torch.manual_seed(10)
    block_size = 4
    num_blocks = 3
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(k_cache, slot, block_size)
        for slot in (0, 1, 3, 4, 7, 8)
    }
    slot_ids = torch.tensor(
        [
            [0, 3, -1, 8, 1],
            [7, -1, 4, 0, 8],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 3, 512, device="cuda", dtype=torch.bfloat16)
    scale = 0.0625

    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, 512), device="cuda")
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, :2],
        lens=lens,
        block_size=block_size,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, 2:],
        lens=lens,
        block_size=block_size,
        candidate_offset=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    gathered = torch.zeros(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    for token_idx in range(slot_ids.shape[0]):
        for topk_idx in range(slot_ids.shape[1]):
            slot = int(slot_ids[token_idx, topk_idx].item())
            if slot >= 0:
                gathered[token_idx, topk_idx] = expected_by_slot[slot]
    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid_tokens = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        gathered,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("head_block_size", [1, 2, 4])
def test_triton_fp8ds_global_slots_multihead_attention_matches_reference(
    head_block_size: int,
) -> None:
    torch.manual_seed(19)
    block_size = 4
    num_blocks = 3
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(k_cache, slot, block_size)
        for slot in (0, 1, 3, 4, 7, 8)
    }
    slot_ids = torch.tensor(
        [
            [0, 3, -1, 8, 1],
            [7, -1, 4, 0, 8],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    q_active = q[:, :, :5]
    scale = 0.0625

    max_score = torch.full((2, 5), float("-inf"), device="cuda")
    denom = torch.zeros((2, 5), device="cuda")
    acc = torch.zeros((2, 5, 512), device="cuda")
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, :2],
        lens=lens,
        block_size=block_size,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
        head_block_size=head_block_size,
    )
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, 2:],
        lens=lens,
        block_size=block_size,
        candidate_offset=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
        head_block_size=head_block_size,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    gathered = torch.zeros(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    for token_idx in range(slot_ids.shape[0]):
        for topk_idx in range(slot_ids.shape[1]):
            slot = int(slot_ids[token_idx, topk_idx].item())
            if slot >= 0:
                gathered[token_idx, topk_idx] = expected_by_slot[slot]
    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid_tokens = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = reference_attention_no_sink(
        q_active,
        gathered,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_fp8ds_paged_attention_chunk_matches_reference() -> None:
    torch.manual_seed(12)
    block_size = 4
    k_cache = torch.zeros(
        3,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [
            [1, 0, 2],
            [2, 1, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([6, 9], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 3, 512, device="cuda", dtype=torch.bfloat16)
    scale = 0.0625

    gathered = torch.zeros(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    expected_by_slot: dict[int, torch.Tensor] = {}
    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = int(block_table[token_idx, block_idx].item())
            slot = physical_block * block_size + block_offset
            expected_by_slot.setdefault(
                slot,
                _write_fp8_ds_mla_token(k_cache, slot, block_size),
            )
            gathered[token_idx, gather_idx] = expected_by_slot[slot]

    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, 512), device="cuda")
    accumulate_fp8ds_paged_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    accumulate_fp8ds_paged_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=2,
        num_candidates=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    offsets = torch.arange(gathered.shape[1], device="cuda")
    valid_tokens = offsets[None, :] < gather_lens[:, None]
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        gathered,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("head_block_size", [1, 2, 4])
def test_triton_fp8ds_paged_multihead_attention_matches_singlehead_and_reference(
    head_block_size: int,
) -> None:
    torch.manual_seed(23)
    block_size = 4
    k_cache = torch.zeros(
        4,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [
            [1, 0, 2, 3],
            [2, 3, 1, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    q_active = q[:, :, :5]
    scale = 0.0625

    gathered = torch.zeros(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    expected_by_slot: dict[int, torch.Tensor] = {}
    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = int(block_table[token_idx, block_idx].item())
            slot = physical_block * block_size + block_offset
            expected_by_slot.setdefault(
                slot,
                _write_fp8_ds_mla_token(k_cache, slot, block_size),
            )
            gathered[token_idx, gather_idx] = expected_by_slot[slot]

    single_max = torch.full((2, 5), float("-inf"), device="cuda")
    single_denom = torch.zeros((2, 5), device="cuda")
    single_acc = torch.zeros((2, 5, 512), device="cuda")
    multi_max = torch.full_like(single_max, float("-inf"))
    multi_denom = torch.zeros_like(single_denom)
    multi_acc = torch.zeros_like(single_acc)

    for candidate_offset, num_candidates in ((0, 2), (2, 3)):
        accumulate_fp8ds_paged_sparse_mla_attention_chunk(
            q=q,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            candidate_offset=candidate_offset,
            num_candidates=num_candidates,
            scale=scale,
            max_score=single_max,
            denom=single_denom,
            acc=single_acc,
        )
        accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            candidate_offset=candidate_offset,
            num_candidates=num_candidates,
            scale=scale,
            max_score=multi_max,
            denom=multi_denom,
            acc=multi_acc,
            head_block_size=head_block_size,
        )

    torch.testing.assert_close(multi_max, single_max, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(multi_denom, single_denom, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(multi_acc, single_acc, rtol=2e-2, atol=2e-2)

    output = torch.empty_like(multi_acc)
    lse = torch.empty_like(multi_max)
    finish_gathered_sparse_mla_attention(
        max_score=multi_max,
        denom=multi_denom,
        acc=multi_acc,
        output=output,
        lse=lse,
    )
    offsets = torch.arange(gathered.shape[1], device="cuda")
    valid_tokens = offsets[None, :] < gather_lens[:, None]
    expected_output, expected_lse = reference_attention_no_sink(
        q_active,
        gathered,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_fp8ds_paged_attention_with_sink_matches_reference() -> None:
    torch.manual_seed(15)
    block_size = 4
    k_cache = torch.zeros(
        3,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor([[1, 0, 2]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
    q = torch.randn(1, 1, 3, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.tensor([-0.25, 0.5, 1.25], device="cuda")
    scale = 0.0625

    gathered = torch.zeros(1, 4, 512, device="cuda", dtype=torch.bfloat16)
    expected_by_slot: dict[int, torch.Tensor] = {}
    start_pos = int(seq_lens[0].item() - gather_lens[0].item())
    for gather_idx in range(int(gather_lens[0].item())):
        pos = start_pos + gather_idx
        physical_block = int(block_table[0, pos // block_size].item())
        slot = physical_block * block_size + pos % block_size
        expected_by_slot.setdefault(
            slot,
            _write_fp8_ds_mla_token(k_cache, slot, block_size),
        )
        gathered[0, gather_idx] = expected_by_slot[slot]

    max_score = torch.full((1, 3), float("-inf"), device="cuda")
    denom = torch.zeros((1, 3), device="cuda")
    acc = torch.zeros((1, 3, 512), device="cuda")
    accumulate_fp8ds_paged_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=4,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    subset_output = torch.empty_like(acc)
    subset_lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=subset_output,
        lse=subset_lse,
    )

    output = torch.empty(1, 3, 512, device="cuda", dtype=torch.bfloat16)
    merge_sparse_mla_subset_with_sink(
        subset_output=subset_output,
        subset_lse=subset_lse,
        attn_sink=sink,
        output=output,
    )
    valid_tokens = torch.ones(1, 4, device="cuda", dtype=torch.bool)
    expected = _golden_sink_attention(q, gathered, valid_tokens, scale, sink)

    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("head_block_size", [1, 2, 4])
def test_triton_fp8ds_paged_attention_with_sink_direct_matches_state_path(
    head_block_size: int,
) -> None:
    torch.manual_seed(29)
    block_size = 4
    k_cache = torch.zeros(
        4,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [[1, 0, 2, 3], [2, 3, 1, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-0.5, 0.5, 5, device="cuda")
    scale = 0.0625

    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            physical_block = int(block_table[token_idx, pos // block_size].item())
            slot = physical_block * block_size + pos % block_size
            _write_fp8_ds_mla_token(k_cache, slot, block_size)

    max_score = torch.full((2, 5), float("-inf"), device="cuda")
    denom = torch.zeros((2, 5), device="cuda")
    acc = torch.zeros((2, 5, 512), device="cuda")
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=5,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
        head_block_size=1,
    )
    expected = torch.empty(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    finish_sparse_mla_attention_with_sink(max_score, denom, acc, sink, expected)

    actual = torch.empty_like(expected)
    fp8ds_paged_sparse_mla_attention_with_sink_multihead(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=5,
        scale=scale,
        attn_sink=sink,
        output=actual,
        head_block_size=head_block_size,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("head_block_size", [1, 2, 4])
def test_triton_fp8ds_global_paged_attention_with_sink_direct_matches_state_path(
    head_block_size: int,
) -> None:
    torch.manual_seed(31)
    compressed_block_size = 4
    swa_block_size = 4
    compressed_cache = torch.zeros(
        4,
        compressed_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    swa_cache = torch.zeros(
        4,
        swa_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    slot_ids = torch.tensor(
        [[0, 3, -1, 8, 1], [7, -1, 4, 0, 8]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [[1, 0, 2, 3], [2, 3, 1, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-1.0, 1.0, 5, device="cuda")
    scale = 0.0625

    for slot in (0, 1, 3, 4, 7, 8):
        _write_fp8_ds_mla_token(compressed_cache, slot, compressed_block_size)
    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            physical_block = int(block_table[token_idx, pos // swa_block_size].item())
            slot = physical_block * swa_block_size + pos % swa_block_size
            _write_fp8_ds_mla_token(swa_cache, slot, swa_block_size)

    comp_max = torch.full((2, 5), float("-inf"), device="cuda")
    comp_denom = torch.zeros((2, 5), device="cuda")
    comp_acc = torch.zeros((2, 5, 512), device="cuda")
    swa_max = torch.full((2, 5), float("-inf"), device="cuda")
    swa_denom = torch.zeros((2, 5), device="cuda")
    swa_acc = torch.zeros((2, 5, 512), device="cuda")
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=compressed_cache,
        slot_ids=slot_ids,
        lens=topk_lens,
        block_size=compressed_block_size,
        candidate_offset=0,
        scale=scale,
        max_score=comp_max,
        denom=comp_denom,
        acc=comp_acc,
        head_block_size=1,
    )
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=swa_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=swa_block_size,
        candidate_offset=0,
        num_candidates=5,
        scale=scale,
        max_score=swa_max,
        denom=swa_denom,
        acc=swa_acc,
        head_block_size=1,
    )
    expected = torch.empty(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    finish_two_sparse_mla_attention_states_with_sink(
        comp_max,
        comp_denom,
        comp_acc,
        swa_max,
        swa_denom,
        swa_acc,
        sink,
        expected,
    )

    actual = torch.empty_like(expected)
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
        q=q,
        compressed_k_cache=compressed_cache,
        slot_ids=slot_ids,
        topk_lens=topk_lens,
        compressed_block_size=compressed_block_size,
        swa_k_cache=swa_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        swa_block_size=swa_block_size,
        num_compressed_candidates=5,
        num_swa_candidates=5,
        scale=scale,
        attn_sink=sink,
        output=actual,
        head_block_size=head_block_size,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_matmul_sparse_mla_attention_with_sink_matches_reference() -> None:
    torch.manual_seed(41)
    q = torch.randn(2, 1, 5, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 7, 512, device="cuda", dtype=torch.bfloat16)
    valid_tokens = torch.tensor(
        [
            [True, True, False, True, False, True, True],
            [False, True, True, False, True, False, False],
        ],
        dtype=torch.bool,
        device="cuda",
    )
    sink = torch.linspace(-0.25, 0.25, 5, device="cuda")
    scale = 0.0625

    expected = torch.empty(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    sink_aware_reference_attention(
        q,
        kv,
        valid_tokens,
        scale,
        sink,
        expected,
    )

    actual = torch.empty_like(expected)
    matmul_sparse_mla_attention_with_sink(
        q,
        kv,
        valid_tokens,
        scale,
        sink,
        actual,
        num_heads=5,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_build_combined_sparse_mla_decode_valid_mask_matches_torch() -> None:
    compressed_slot_ids = torch.tensor(
        [
            [7, 4, -1, 9, 11],
            [2, -1, 3, 8, 10],
            [-1, -1, -1, -1, -1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    topk_lens = torch.tensor([4, 3, 0], device="cuda", dtype=torch.int32)
    swa_lens = torch.tensor([3, 1, 0], device="cuda", dtype=torch.int32)
    valid_tokens = torch.empty(3, 9, device="cuda", dtype=torch.bool)

    build_combined_sparse_mla_decode_valid_mask(
        valid_tokens,
        compressed_slot_ids,
        topk_lens,
        swa_lens,
    )

    comp_offsets = torch.arange(5, device="cuda", dtype=torch.int32)
    swa_offsets = torch.arange(4, device="cuda", dtype=torch.int32)
    expected = torch.empty_like(valid_tokens)
    expected[:, :5] = (comp_offsets[None, :] < topk_lens[:, None]) & (
        compressed_slot_ids >= 0
    )
    expected[:, 5:] = swa_offsets[None, :] < swa_lens[:, None]

    torch.testing.assert_close(valid_tokens, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
def test_triton_fp8ds_paged_attention_with_sink_supports_tp_local_heads(
    num_heads: int,
) -> None:
    torch.manual_seed(37 + num_heads)
    block_size = 4
    k_cache = torch.zeros(
        4,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [[1, 0, 2, 3], [2, 3, 1, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-0.5, 0.5, num_heads, device="cuda")
    scale = 0.0625

    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            physical_block = int(block_table[token_idx, pos // block_size].item())
            slot = physical_block * block_size + pos % block_size
            _write_fp8_ds_mla_token(k_cache, slot, block_size)

    max_score = torch.full((2, num_heads), float("-inf"), device="cuda")
    denom = torch.zeros((2, num_heads), device="cuda")
    acc = torch.zeros((2, num_heads, 512), device="cuda")
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=5,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
        head_block_size=1,
    )
    expected = torch.empty(2, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    finish_sparse_mla_attention_with_sink(max_score, denom, acc, sink, expected)

    actual = torch.empty_like(expected)
    fp8ds_paged_sparse_mla_attention_with_sink_multihead(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=5,
        scale=scale,
        attn_sink=sink,
        output=actual,
        head_block_size=4,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
def test_triton_fp8ds_global_paged_attention_with_sink_supports_tp_local_heads(
    num_heads: int,
) -> None:
    torch.manual_seed(41 + num_heads)
    compressed_block_size = 4
    swa_block_size = 4
    compressed_cache = torch.zeros(
        4,
        compressed_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    swa_cache = torch.zeros(
        4,
        swa_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    slot_ids = torch.tensor(
        [[0, 3, -1, 8, 1], [7, -1, 4, 0, 8]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [[1, 0, 2, 3], [2, 3, 1, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-1.0, 1.0, num_heads, device="cuda")
    scale = 0.0625

    for slot in (0, 1, 3, 4, 7, 8):
        _write_fp8_ds_mla_token(compressed_cache, slot, compressed_block_size)
    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            physical_block = int(block_table[token_idx, pos // swa_block_size].item())
            slot = physical_block * swa_block_size + pos % swa_block_size
            _write_fp8_ds_mla_token(swa_cache, slot, swa_block_size)

    comp_max = torch.full((2, num_heads), float("-inf"), device="cuda")
    comp_denom = torch.zeros((2, num_heads), device="cuda")
    comp_acc = torch.zeros((2, num_heads, 512), device="cuda")
    swa_max = torch.full((2, num_heads), float("-inf"), device="cuda")
    swa_denom = torch.zeros((2, num_heads), device="cuda")
    swa_acc = torch.zeros((2, num_heads, 512), device="cuda")
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=compressed_cache,
        slot_ids=slot_ids,
        lens=topk_lens,
        block_size=compressed_block_size,
        candidate_offset=0,
        scale=scale,
        max_score=comp_max,
        denom=comp_denom,
        acc=comp_acc,
        head_block_size=1,
    )
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=swa_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=swa_block_size,
        candidate_offset=0,
        num_candidates=5,
        scale=scale,
        max_score=swa_max,
        denom=swa_denom,
        acc=swa_acc,
        head_block_size=1,
    )
    expected = torch.empty(2, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    finish_two_sparse_mla_attention_states_with_sink(
        comp_max,
        comp_denom,
        comp_acc,
        swa_max,
        swa_denom,
        swa_acc,
        sink,
        expected,
    )

    actual = torch.empty_like(expected)
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
        q=q,
        compressed_k_cache=compressed_cache,
        slot_ids=slot_ids,
        topk_lens=topk_lens,
        compressed_block_size=compressed_block_size,
        swa_k_cache=swa_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        swa_block_size=swa_block_size,
        num_compressed_candidates=5,
        num_swa_candidates=5,
        scale=scale,
        attn_sink=sink,
        output=actual,
        head_block_size=4,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_indexed_bf16_prefill_chunks_match_reference() -> None:
    torch.manual_seed(17)
    q = torch.randn(5, 5, 16, device="cuda", dtype=torch.bfloat16)
    q_active = q[:, :3]
    kv = torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16)
    kv_flat = kv.reshape(-1, q.shape[-1])
    combined_indices = torch.tensor(
        [
            [0, 3, -1, 5, 3, 1],
            [4, -1, 2, 2, 1, 8],
            [-1, -1, -1, -1, -1, -1],
            [8, 0, 9, -1, 7, 4],
            [13, 12, 0, 12, -1, 3],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    combined_lens = torch.tensor([5, 4, 0, 6, 5], dtype=torch.int32, device="cuda")
    sink = torch.tensor([-0.5, 1.0, 0.25], dtype=torch.float32, device="cuda")
    scale = 0.375
    output = torch.empty_like(q_active)

    for token_start in (0, 2, 4):
        token_end = min(token_start + 2, q.shape[0])
        q_chunk = q[token_start:token_end]
        indices_chunk = combined_indices[token_start:token_end]
        lens_chunk = combined_lens[token_start:token_end]
        max_score = torch.full(
            (q_chunk.shape[0], q_active.shape[1]),
            float("-inf"),
            device="cuda",
        )
        denom = torch.zeros_like(max_score)
        acc = torch.zeros(
            q_chunk.shape[0],
            q_active.shape[1],
            q_chunk.shape[-1],
            device="cuda",
            dtype=torch.float32,
        )
        for index_start in (0, 3):
            index_end = min(index_start + 3, combined_indices.shape[-1])
            accumulate_indexed_sparse_mla_attention_chunk(
                q=q_chunk,
                kv_flat=kv_flat,
                indices=indices_chunk[:, index_start:index_end],
                lens=lens_chunk,
                candidate_offset=index_start,
                scale=scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
            )
        subset_output = torch.empty_like(acc)
        subset_lse = torch.empty_like(max_score)
        finish_gathered_sparse_mla_attention(
            max_score=max_score,
            denom=denom,
            acc=acc,
            output=subset_output,
            lse=subset_lse,
        )
        merge_sparse_mla_subset_with_sink(
            subset_output=subset_output,
            subset_lse=subset_lse,
            attn_sink=sink,
            output=output[token_start:token_end],
        )

    expected = torch.empty_like(q_active)
    reference_sparse_mla_prefill(
        q=q_active,
        kv=kv,
        combined_indices=combined_indices,
        combined_lens=combined_lens,
        scale=scale,
        attn_sink=sink,
        output=expected,
        topk_chunk_size=3,
        query_chunk_size=2,
    )

    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("topk_chunk_size", "query_chunk_size"),
    [(1, 1), (2, 3), (5, 2)],
)
def test_reference_sparse_mla_prefill_matches_dense_golden(
    topk_chunk_size: int,
    query_chunk_size: int,
) -> None:
    torch.manual_seed(4)
    scale = 0.375
    q = torch.randn(4, 2, 3)
    kv = torch.randn(2, 5, 3)
    combined_indices = torch.tensor(
        [
            [0, 3, -1, 5, 3],
            [4, -1, 2, 2, 1],
            [-1, -1, -1, -1, -1],
            [8, 0, 9, -1, 7],
        ],
        dtype=torch.int64,
    )
    combined_lens = torch.tensor([4, 3, 0, 5], dtype=torch.int32)
    sink = torch.tensor([-0.5, 1.0])
    output = torch.empty_like(q)

    reference_sparse_mla_prefill(
        q=q,
        kv=kv,
        combined_indices=combined_indices,
        combined_lens=combined_lens,
        scale=scale,
        attn_sink=sink,
        output=output,
        topk_chunk_size=topk_chunk_size,
        query_chunk_size=query_chunk_size,
    )

    kv_flat = kv.reshape(-1, q.shape[-1])
    offsets = torch.arange(combined_indices.shape[-1])
    valid_tokens = (offsets[None, :] < combined_lens[:, None]) & (
        combined_indices >= 0
    )
    safe_indices = torch.where(
        valid_tokens,
        combined_indices,
        torch.zeros((), dtype=combined_indices.dtype),
    ).long()
    gathered_kv = kv_flat[safe_indices]
    expected = _golden_sink_attention(q, gathered_kv, valid_tokens, scale, sink)

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_chunked_reference_accumulation_matches_one_shot(chunk_size: int) -> None:
    torch.manual_seed(3)
    scale = 0.3
    q = torch.randn(3, 2, 4)
    kv = torch.randn(3, 9, 4)
    valid_tokens = torch.tensor(
        [
            [True, False, True, True, False, False, True, False, True],
            [False, False, False, False, False, False, False, False, False],
            [True, True, True, False, True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    output, lse = _chunked_no_sink_attention(
        q,
        kv,
        valid_tokens,
        scale,
        chunk_size,
    )
    expected_output, expected_lse = _golden_no_sink_attention(
        q,
        kv,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)

def test_triton_sparse_mla_fallback_allows_cudagraph_support_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE", "1")
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", raising=False)

    mla_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        model_version="deepseek_v4",
    )

    assert FlashMLASparseMetadataBuilder.get_cudagraph_support(None, mla_spec) is (
        AttentionCGSupport.UNIFORM_BATCH
    )
    assert DeepseekSparseSWAMetadataBuilder.get_cudagraph_support(None, swa_spec) is (
        AttentionCGSupport.UNIFORM_BATCH
    )

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            compile_sizes=[1, 2],
            compile_ranges_endpoints=[8192],
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4],
            max_cudagraph_capture_size=4,
        )
    )
    disable_sparse_mla_reference_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert vllm_config.compilation_config.compile_sizes == [1, 2]
    assert vllm_config.compilation_config.compile_ranges_endpoints == [8192]
    assert (
        vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2, 4]
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 4



def test_triton_sparse_mla_fallback_can_disable_cudagraphs(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE", "1")
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "0")

    mla_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        model_version="deepseek_v4",
    )

    assert FlashMLASparseMetadataBuilder.get_cudagraph_support(None, mla_spec) is (
        AttentionCGSupport.NEVER
    )
    assert DeepseekSparseSWAMetadataBuilder.get_cudagraph_support(None, swa_spec) is (
        AttentionCGSupport.NEVER
    )

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            compile_sizes=[1, 2],
            compile_ranges_endpoints=[8192],
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4],
            max_cudagraph_capture_size=4,
        )
    )
    disable_sparse_mla_reference_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert vllm_config.compilation_config.compile_sizes == []
    assert vllm_config.compilation_config.compile_ranges_endpoints == []
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert vllm_config.compilation_config.cudagraph_capture_sizes == []
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 0


def test_triton_sparse_mla_fallback_disables_cudagraphs_for_mtp(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE", "1")
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", raising=False)

    mla_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        model_version="deepseek_v4",
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=2,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            compile_sizes=[1, 2],
            compile_ranges_endpoints=[8192],
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4],
            max_cudagraph_capture_size=4,
        ),
    )

    assert FlashMLASparseMetadataBuilder.get_cudagraph_support(
        vllm_config,
        mla_spec,
    ) is AttentionCGSupport.NEVER
    assert DeepseekSparseSWAMetadataBuilder.get_cudagraph_support(
        vllm_config,
        swa_spec,
    ) is AttentionCGSupport.NEVER

    disable_sparse_mla_reference_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert vllm_config.compilation_config.compile_sizes == []
    assert vllm_config.compilation_config.compile_ranges_endpoints == []
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert vllm_config.compilation_config.cudagraph_capture_sizes == []
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 0
