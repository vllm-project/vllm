# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
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


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


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


def _make_sm120_impl(
    monkeypatch,
    *,
    model_type: str = "deepseek_v32",
    num_heads: int = 8,
    sinks: torch.Tensor | None = None,
) -> FlashInferMLASparseSM120Impl:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    topk = torch.zeros((4, 2048), dtype=torch.int32)
    with set_current_vllm_config(_fake_vllm_config(model_type)):
        return FlashInferMLASparseSM120Impl(
            num_heads=num_heads,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            indexer=None,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            topk_indices_buffer=topk,
            sinks=sinks,
        )


def test_sm120_backend_supports_sink(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    assert FlashInferMLASparseSM120Backend.supports_sink()

    with set_current_vllm_config(_fake_vllm_config("hy_v4")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=True,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_impl_stores_sinks(monkeypatch) -> None:
    sinks = torch.zeros(8, dtype=torch.float32)
    impl = _make_sm120_impl(monkeypatch, sinks=sinks)
    assert impl.sinks is sinks


def test_sm120_impl_rejects_bad_sink_shape(monkeypatch) -> None:
    with pytest.raises(ValueError, match="sinks"):
        _make_sm120_impl(monkeypatch, sinks=torch.zeros(4, dtype=torch.float32))


def _patch_forward_deps(monkeypatch, captured: dict) -> None:
    def fake_kernel(**kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120."
        "triton_convert_req_index_to_global_index",
        lambda *args, **kwargs: args[2],
    )
    monkeypatch.setattr(
        "vllm.utils.flashinfer.flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_kernel,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120."
        "_get_workspace_buffer",
        lambda device: torch.zeros(8, dtype=torch.int8),
    )


def _sparse_metadata(num_toks: int) -> FlashInferMLASparseMetadata:
    return FlashInferMLASparseMetadata(
        num_reqs=1,
        max_query_len=num_toks,
        max_seq_len=num_toks,
        num_actual_tokens=num_toks,
        query_start_loc=torch.tensor([0, num_toks], dtype=torch.int32),
        slot_mapping=torch.zeros(num_toks, dtype=torch.int64),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        req_id_per_token=torch.zeros(num_toks, dtype=torch.int32),
        seq_lens=torch.tensor([num_toks], dtype=torch.int32),
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=num_toks,
        block_size=64,
        topk_tokens=2048,
    )


def test_sm120_impl_forwards_sinks_to_kernel(monkeypatch) -> None:
    captured: dict = {}
    _patch_forward_deps(monkeypatch, captured)

    sinks = torch.linspace(0.1, 0.7, 8, dtype=torch.float32)
    impl = _make_sm120_impl(monkeypatch, sinks=sinks)
    num_toks = 2
    q = torch.zeros((num_toks, 8, 576), dtype=torch.bfloat16)
    kv = torch.zeros((4, 64, 656), dtype=torch.uint8)
    metadata = _sparse_metadata(num_toks)
    impl.forward_mqa(q, kv, metadata, layer=None)  # type: ignore[arg-type]
    assert captured["sinks"] is sinks


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192
