# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.offloader as offloader
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
)


class _WeightOnlyLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class _Compressor(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.fused_wkv_wgate = _WeightOnlyLinear(weight)


class _IndexerWeightsProjection(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states.sum(dim=-1, keepdim=True), None


class _Indexer(torch.nn.Module):
    def __init__(self, compressor_weight: torch.Tensor) -> None:
        super().__init__()
        self.compressor = _Compressor(compressor_weight)
        self.weights_proj = _IndexerWeightsProjection()


def _make_attention(
    main_weight: torch.Tensor,
    indexer_weight: torch.Tensor | None,
    use_aiter: bool = False,
) -> DeepseekV4ROCMAiterMLAAttention:
    attention = DeepseekV4ROCMAiterMLAAttention.__new__(DeepseekV4ROCMAiterMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.compressor = _Compressor(main_weight)
    attention.indexer = None if indexer_weight is None else _Indexer(indexer_weight)
    attention.register_buffer("_fused_compressor_weight", None, persistent=False)
    attention._fused_compressor_split_sizes = None
    attention._use_aiter_fp32_score_gemm = use_aiter
    return attention


def _record_score_gemms(monkeypatch, use_aiter: bool) -> list:
    """Replace the FP32 score GEMM the layer picked with a CPU recorder.

    Each call is stored as (input, [N, K] weight, output dtype).
    """
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.dtype | None]] = []
    if use_aiter:
        tgemm = pytest.importorskip("aiter.tuned_gemm").tgemm

        def fake_tgemm_mm(left, weight, *, otype=None):
            calls.append((left, weight, otype))
            return torch.matmul(left, weight.t()).to(otype)

        monkeypatch.setattr(tgemm, "mm", fake_tgemm_mm)
    else:
        real_mm = torch.mm

        def fake_torch_mm(left, right, *, out_dtype=None):
            calls.append((left, right.t(), out_dtype))
            return real_mm(left, right).to(out_dtype)

        monkeypatch.setattr(torch, "mm", fake_torch_mm)
    return calls


def test_prepare_compressor_gemm_fusion_aliases_original_weights():
    main_value = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    indexer_value = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 100
    attention = _make_attention(main_value, indexer_value)
    main_weight = attention.compressor.fused_wkv_wgate.weight
    indexer_weight = attention.indexer.compressor.fused_wkv_wgate.weight

    assert attention.prepare_compressor_gemm_fusion() is True

    fused_weight = attention._fused_compressor_weight
    assert fused_weight is not None
    assert attention._fused_compressor_split_sizes == (4, 2)
    torch.testing.assert_close(fused_weight[:4], main_value)
    torch.testing.assert_close(fused_weight[4:], indexer_value)

    fused_storage = fused_weight.untyped_storage().data_ptr()
    assert main_weight.untyped_storage().data_ptr() == fused_storage
    assert indexer_weight.untyped_storage().data_ptr() == fused_storage
    assert main_weight.storage_offset() == 0
    assert indexer_weight.storage_offset() == main_weight.numel()

    state_dict = attention.state_dict()
    assert "_fused_compressor_weight" not in state_dict
    torch.testing.assert_close(
        state_dict["compressor.fused_wkv_wgate.weight"], main_value
    )
    torch.testing.assert_close(
        state_dict["indexer.compressor.fused_wkv_wgate.weight"],
        indexer_value,
    )

    fused_data_ptr = fused_weight.data_ptr()
    assert attention.prepare_compressor_gemm_fusion() is False
    assert attention._fused_compressor_weight.data_ptr() == fused_data_ptr


def test_prepare_compressor_gemm_fusion_skips_weight_offloading(monkeypatch):
    attention = _make_attention(torch.ones(4, 3), torch.ones(2, 3))
    main_weight = attention.compressor.fused_wkv_wgate.weight
    indexer_weight = attention.indexer.compressor.fused_wkv_wgate.weight

    monkeypatch.setattr(offloader, "get_offloader", object)

    assert attention.prepare_compressor_gemm_fusion() is False
    assert attention._fused_compressor_weight is None
    assert main_weight.untyped_storage().data_ptr() != (
        indexer_weight.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("use_aiter", [False, True])
def test_fused_input_projection_uses_one_mm_and_returns_expected_tuple(
    monkeypatch, use_aiter
):
    main_weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    indexer_weight = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    attention = _make_attention(main_weight, indexer_weight, use_aiter)
    assert attention.prepare_compressor_gemm_fusion() is True

    hidden_states = torch.tensor(
        [[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]], dtype=torch.float32
    )
    qr_kv = torch.full((2, 4), 17.0)
    monkeypatch.setattr(
        DeepseekV4ROCMAiterMLAAttention,
        "_fused_wqa_wkv_gemm",
        lambda self, hidden: qr_kv,
    )

    mm_calls = _record_score_gemms(monkeypatch, use_aiter)

    qr_out, kv_score, indexer_kv_score, indexer_weights = (
        attention._run_parallel_input_projections(hidden_states)
    )

    assert qr_out is qr_kv
    torch.testing.assert_close(kv_score, hidden_states[:, :2])
    torch.testing.assert_close(indexer_kv_score, hidden_states[:, 2:])
    torch.testing.assert_close(indexer_weights, hidden_states.sum(dim=-1, keepdim=True))
    assert len(mm_calls) == 1
    assert mm_calls[0][0] is hidden_states
    assert mm_calls[0][2] is torch.float32
    assert mm_calls[0][1].shape == (3, 3)


def test_hca_input_projection_uses_aiter_score_gemm(monkeypatch):
    """HCA layers (compressor, no indexer) take the score GEMM the layer chose
    and return no indexer outputs."""
    main_weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    attention = _make_attention(main_weight, None, use_aiter=True)
    hidden_states = torch.tensor(
        [[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]], dtype=torch.float32
    )
    qr_kv = torch.full((2, 4), 17.0)
    monkeypatch.setattr(
        DeepseekV4ROCMAiterMLAAttention,
        "_fused_wqa_wkv_gemm",
        lambda self, hidden: qr_kv,
    )
    mm_calls = _record_score_gemms(monkeypatch, use_aiter=True)

    qr_out, kv_score, indexer_kv_score, indexer_weights = (
        attention._run_parallel_input_projections(hidden_states)
    )

    assert qr_out is qr_kv
    torch.testing.assert_close(kv_score, hidden_states[:, :2])
    assert indexer_kv_score is None
    assert indexer_weights is None
    assert len(mm_calls) == 1
    assert mm_calls[0][1] is attention.compressor.fused_wkv_wgate.weight
    assert mm_calls[0][2] is torch.float32


def test_hca_input_projection_without_aiter_uses_base_path(monkeypatch):
    """With the aiter score GEMM off, HCA layers keep the base implementation."""
    attention = _make_attention(torch.ones(2, 3), None, use_aiter=False)
    sentinel = (torch.zeros(1), torch.zeros(1), None, None)
    monkeypatch.setattr(
        DeepseekV4ROCMAiterMLAAttention.__mro__[1],
        "_run_parallel_input_projections",
        lambda self, hidden: sentinel,
    )

    assert attention._run_parallel_input_projections(torch.ones(2, 3)) is sentinel


@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_rocm_inv_rope_einsum_bmm_matches_einsum(monkeypatch, num_tokens):
    """The aiter BF16 BMM path over transposed strided views matches the einsum
    it replaces. Runs only where that path is enabled (CDNA4+ with aiter)."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    monkeypatch.setattr(rocm_aiter_ops, "_LINEAR_ENABLED", True)
    if not torch.cuda.is_available() or not rocm_aiter_ops.is_triton_bf16_bmm_enabled():
        pytest.skip("needs aiter's Triton BF16 BMM (CDNA4+ with aiter)")
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum

    groups, rank, dim = 8, 1024, 512
    torch.manual_seed(num_tokens)
    o = torch.randn(num_tokens, groups * dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(groups, rank, dim, device="cuda", dtype=torch.bfloat16)
    wo_a = torch.nn.Module()
    wo_a._dsv4_wo_a_bf16 = weight  # the cached bf16 weight the op reads

    out = rocm_inv_rope_einsum(
        None, o, None, 64, groups, rank, wo_a, inverse_rope=False
    )
    ref = torch.einsum(
        "tgd,grd->tgr", o.view(num_tokens, groups, dim).float(), weight.float()
    )
    torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=2e-2)
