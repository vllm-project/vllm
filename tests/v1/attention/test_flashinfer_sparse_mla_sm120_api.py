# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(
    model_type: str,
    *,
    index_topk: int | None = 2048,
) -> SimpleNamespace:
    hf_text_config = SimpleNamespace(model_type=model_type)
    if index_topk is not None:
        hf_text_config.index_topk = index_topk
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=hf_text_config,
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_sparse_backend_rejects_model_without_index_topk(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe", index_topk=None)):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == [
        "FLASHINFER_MLA_SPARSE_SM120 requires a model with index_topk config"
    ]


def test_sm120_sparse_impl_accepts_shared_topk_buffer_without_indexer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    topk_indices_buffer = torch.empty(4, 2048, dtype=torch.int32)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=16,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            indexer=None,
            kv_lora_rank=512,
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
            topk_indices_buffer=topk_indices_buffer,
        )

    assert impl.topk_indices_buffer is topk_indices_buffer


def test_sm120_sparse_impl_rejects_missing_topk_indices() -> None:
    with (
        set_current_vllm_config(_fake_vllm_config("glm4_moe")),
        pytest.raises(ValueError, match="requires sparse-MLA top-k indices"),
    ):
        FlashInferMLASparseSM120Impl(
            num_heads=16,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            indexer=None,
            kv_lora_rank=512,
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
        )
