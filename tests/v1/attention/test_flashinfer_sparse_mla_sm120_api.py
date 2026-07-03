# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import (
    _canonicalize_sparse_mla_kv_cache_dtype,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
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
    monkeypatch.setattr(
        fi_utils,
        "flashinfer_mla_decode_supports_kv_scale_format",
        lambda: True,
    )

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


def test_sm120_standard_fp8_cache_uses_packed_flashinfer_shape() -> None:
    assert (
        _canonicalize_sparse_mla_kv_cache_dtype(FlashInferMLASparseSM120Backend, "auto")
        == "fp8_ds_mla"
    )
    assert (
        _canonicalize_sparse_mla_kv_cache_dtype(
            FlashInferMLASparseSM120Backend, "fp8_e4m3"
        )
        == "fp8_ds_mla"
    )
    assert FlashInferMLASparseSM120Backend.get_kv_cache_shape(
        num_blocks=2,
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str="fp8_e4m3",
    ) == (2, 64, 576)
    assert FlashInferMLASparseSM120Backend.get_kv_cache_shape(
        num_blocks=2,
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str="fp8_ds_mla",
    ) == (2, 64, 656)


def test_sm120_rejects_fp8_ds_mla_without_flashinfer_packed_support(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils,
        "flashinfer_mla_decode_supports_kv_scale_format",
        lambda: False,
    )

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert any("fp8_ds_mla" in reason for reason in invalid_reasons)


def test_sm120_impl_passes_packed_sparse_mla_args(
    monkeypatch,
) -> None:
    captured_kwargs = {}

    def fake_decode(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(
        fi_utils,
        "flashinfer_mla_decode_supports_kv_scale_format",
        lambda: True,
    )
    monkeypatch.setattr(
        fi_utils,
        "flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_decode,
    )
    monkeypatch.setattr(
        flashinfer_mla_sparse_sm120,
        "_get_workspace_buffer",
        lambda device: torch.empty((1,), device=device),
    )
    monkeypatch.setattr(
        flashinfer_mla_sparse_sm120,
        "triton_convert_req_index_to_global_index",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 1), dtype=torch.int32),
            torch.ones((1,), dtype=torch.int32),
        ),
    )

    impl = object.__new__(FlashInferMLASparseSM120Impl)
    impl.num_heads = 1
    impl.scale = 1.0
    impl.kv_lora_rank = 1
    impl.qk_nope_head_dim = 1
    impl.qk_rope_head_dim = 0
    impl.kv_scale_format = "arbitrary_fp32"
    impl.topk_indices_buffer = torch.zeros((1, 1), dtype=torch.int32)
    impl._workspace_buffer = None
    impl.kv_cache_dtype = "fp8_ds_mla"

    q = torch.empty((1, 1), dtype=torch.bfloat16)
    kv_cache = torch.zeros((1, 1), dtype=torch.uint8)
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros((1,), dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=1,
        topk_tokens=1,
    )
    layer = SimpleNamespace(_q_scale_float=2.0, _k_scale_float=3.0)

    output, _ = impl.forward_mqa(q, kv_cache, metadata, layer=layer)

    assert output.shape == (1, 1, 1)
    assert output.dtype == torch.bfloat16
    assert captured_kwargs["query"].dtype == torch.bfloat16
    assert captured_kwargs["kv_cache"].dtype == torch.uint8
    assert captured_kwargs["bmm1_scale"] == 1.0
    assert captured_kwargs["bmm2_scale"] == 1.0
    assert captured_kwargs["seq_lens"].shape == (1,)
    assert captured_kwargs["seq_lens"].dtype == torch.int32
    assert captured_kwargs["kv_scale_format"] == "arbitrary_fp32"
    assert "backend" not in captured_kwargs
