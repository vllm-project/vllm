# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120_module
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


def test_sm120_sparse_mla_dcp_plumbs_lse_and_valid_counts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        sm120_module,
        "_get_workspace_buffer",
        lambda device: torch.empty(1, dtype=torch.uint8, device=device),
    )

    num_tokens = 3
    topk_tokens = 4
    num_query_heads = 32
    kv_lora_rank = 512
    topk_indices = torch.arange(num_tokens * topk_tokens, dtype=torch.int32).reshape(
        num_tokens, topk_tokens
    )
    topk_indices_physical = torch.tensor(
        [[0, 1, -1, -1], [-1, -1, -1, -1], [4, 5, 6, 7]],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([2, 0, 4], dtype=torch.int32)
    captured: dict[str, dict[str, object]] = {}

    def fake_filter_and_convert_dcp_index(*args: object, **kwargs: object):
        captured["dcp_kwargs"] = kwargs
        return topk_indices_physical, seq_lens

    def fake_flashinfer_decode(**kwargs: object):
        out = kwargs["out"]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (num_tokens, 1, num_query_heads, kv_lora_rank)
        assert kwargs["seq_lens"] is seq_lens
        assert kwargs["return_lse"] is True
        assert isinstance(kwargs["block_tables"], torch.Tensor)
        assert kwargs["block_tables"].shape == (num_tokens, 1, topk_tokens)
        out.fill_(1.0)
        lse = torch.ones(num_tokens, 1, num_query_heads, dtype=torch.float32)
        return out, lse

    monkeypatch.setattr(
        sm120_module,
        "triton_filter_and_convert_dcp_index",
        fake_filter_and_convert_dcp_index,
    )
    monkeypatch.setattr(
        fi_utils,
        "flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_flashinfer_decode,
    )

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=8,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            indexer=SimpleNamespace(topk_indices_buffer=topk_indices),
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
        )

    impl.dcp_world_size = 4
    impl.dcp_rank = 2
    impl.need_to_return_lse_for_decode = True

    q = torch.zeros(num_tokens, num_query_heads, 576, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, 64, 656, dtype=torch.uint8)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 0, 1], dtype=torch.int32),
        block_table=torch.arange(8, dtype=torch.int32).reshape(2, 4),
        block_size=64,
        cp_kv_cache_interleave_size=1,
        topk_tokens=topk_tokens,
    )

    out, lse = impl.forward_mqa(q, kv_cache, metadata, SimpleNamespace())

    assert out.shape == (num_tokens, num_query_heads, kv_lora_rank)
    assert lse is not None
    assert lse.shape == (num_tokens, num_query_heads)
    assert torch.all(out[0] == 1)
    assert torch.all(out[1] == 0)
    assert torch.isneginf(lse[1]).all()
    assert captured["dcp_kwargs"]["dcp_size"] == 4
    assert captured["dcp_kwargs"]["dcp_rank"] == 2
    assert captured["dcp_kwargs"]["return_valid_counts"] is True
