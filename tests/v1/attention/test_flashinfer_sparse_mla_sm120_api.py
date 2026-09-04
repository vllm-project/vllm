# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection and calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
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


def test_sm120_nope_forward_preserves_native_sparse_mla_contract(monkeypatch) -> None:
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16, device="cpu")
    kv_cache = torch.zeros(1, 64, 656, dtype=torch.uint8, device="cpu")
    topk = torch.zeros(2, 2176, dtype=torch.int32, device="cpu")
    physical_topk = torch.full((2, 2176), -1, dtype=torch.int32, device="cpu")
    convert = MagicMock(return_value=physical_topk)
    monkeypatch.setattr(sm120, "triton_convert_req_index_to_global_index", convert)
    decode = MagicMock(side_effect=lambda **kwargs: kwargs["out"].fill_(1))
    monkeypatch.setattr(
        fi_utils, "flashinfer_trtllm_batch_decode_with_kv_cache_mla", decode
    )

    impl = object.__new__(sm120.FlashInferMLASparseSM120Impl)
    impl.num_heads = 16
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.scale = 0.125
    impl.kv_scale_format = sm120._kv_scale_format_for_model("glm4_moe")
    impl.topk_indices_buffer = topk
    impl.index_group = None
    impl._workspace_buffer = torch.empty(1, dtype=torch.uint8, device="cpu")
    metadata = SimpleNamespace(
        topk_tokens=2048,
        block_size=64,
        req_id_per_token=torch.zeros(2, dtype=torch.int32, device="cpu"),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device="cpu"),
    )

    output, lse = impl.forward_mqa(q, kv_cache, metadata, SimpleNamespace())

    convert.assert_called_once()
    decode.assert_called_once()
    kwargs = decode.call_args.kwargs
    torch.testing.assert_close(kwargs["query"], q.unsqueeze(1))
    assert kwargs["query"].data_ptr() == q.data_ptr()
    assert kwargs["qk_nope_head_dim"] == 256
    assert kwargs["qk_rope_head_dim"] == 0
    assert kwargs["kv_lora_rank"] == 512
    assert kwargs["kv_cache"].shape == (1, 1, 64, 656)
    assert kwargs["kv_cache"].dtype == torch.uint8
    assert kwargs["kv_cache"].data_ptr() == kv_cache.data_ptr()
    torch.testing.assert_close(kwargs["block_tables"], physical_topk.unsqueeze(1))
    assert kwargs["max_seq_len"] == 2176
    assert kwargs["sparse_mla_top_k"] == 2176
    assert kwargs["seq_lens"] is None
    assert "sparse_mla_top_k_lens" not in kwargs
    assert kwargs["kv_scale_format"] == "arbitrary_fp32"
    torch.testing.assert_close(output, torch.ones_like(q))
    assert lse is None
