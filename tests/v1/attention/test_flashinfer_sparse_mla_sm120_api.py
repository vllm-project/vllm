# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    required_sm120_sparse_topk,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
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


def test_sm120_dense_prefill_requires_rope_tail(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    vllm_config = _fake_vllm_config("deepseek_v32")
    vllm_config.parallel_config = SimpleNamespace(decode_context_parallel_size=1)

    def build_impl(qk_rope_head_dim: int):
        def fake_common_init(self, *args, **kwargs) -> None:
            self.qk_rope_head_dim = qk_rope_head_dim
            self.topk_indices_buffer = kwargs["topk_indices_buffer"]

        monkeypatch.setattr(
            "vllm.model_executor.layers.attention.sparse_mla_attention."
            "SparseMLACommonImpl.__init__",
            fake_common_init,
        )
        with set_current_vllm_config(vllm_config):
            return FlashInferMLASparseSM120Backend.get_impl_cls()(
                num_heads=64,
                head_size=512 + qk_rope_head_dim,
                scale=1.0,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="fp8_ds_mla",
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
                topk_indices_buffer=torch.empty(1, dtype=torch.int32),
            )

    assert build_impl(qk_rope_head_dim=64).supports_dense_mha_prefill
    assert not build_impl(qk_rope_head_dim=0).supports_dense_mha_prefill


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

    assert required_sm120_sparse_topk(causal, 128) == 128
    assert required_sm120_sparse_topk(dspark, 128) == 192
