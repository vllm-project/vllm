# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    indexer_weight: torch.Tensor,
) -> DeepseekV4ROCMAiterMLAAttention:
    attention = DeepseekV4ROCMAiterMLAAttention.__new__(DeepseekV4ROCMAiterMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.compressor = _Compressor(main_weight)
    attention.indexer = _Indexer(indexer_weight)
    attention.register_buffer("_fused_compressor_weight", None, persistent=False)
    attention._fused_compressor_split_sizes = None
    return attention


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


def test_fused_input_projection_uses_one_mm_and_returns_expected_tuple(
    monkeypatch,
):
    main_weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    indexer_weight = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    attention = _make_attention(main_weight, indexer_weight)
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

    mm_calls: list[tuple[torch.Tensor, torch.Tensor, torch.dtype | None]] = []

    def cpu_mm_with_out_dtype(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        mm_calls.append((left, right, out_dtype))
        return torch.matmul(left, right).to(out_dtype)

    monkeypatch.setattr(torch, "mm", cpu_mm_with_out_dtype)

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
