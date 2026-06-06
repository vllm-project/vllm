# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.config as vllm_config
import vllm.utils.flashinfer as flashinfer_utils
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as dsv4_flashinfer
from vllm.models.deepseek_v4.nvidia import model as dsv4_model
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig


def _vllm_config(backend):
    return SimpleNamespace(attention_config=SimpleNamespace(backend=backend))


def _set_capability(monkeypatch, major: int, minor: int = 0) -> None:
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(major, minor),
    )


def test_dsv4_default_uses_flashinfer_sparse_on_sm120(monkeypatch):
    _set_capability(monkeypatch, 12, 0)

    attn_cls = dsv4_model._select_dsv4_attn_cls(_vllm_config(None))

    assert attn_cls is dsv4_model.DeepseekV4FlashInferSM120Attention


def test_dsv4_default_keeps_flashmla_on_sm100(monkeypatch):
    _set_capability(monkeypatch, 10, 0)

    attn_cls = dsv4_model._select_dsv4_attn_cls(_vllm_config(None))

    assert attn_cls is dsv4_model.DeepseekV4FlashMLAAttention


def test_dsv4_explicit_flashinfer_sparse_backend_is_mapped(monkeypatch):
    _set_capability(monkeypatch, 10, 0)

    attn_cls = dsv4_model._select_dsv4_attn_cls(
        _vllm_config(AttentionBackendEnum.FLASHINFER_MLA_SPARSE)
    )

    assert attn_cls is dsv4_model.DeepseekV4FlashInferTRTLLMAttention


def test_dsv4_explicit_flashmla_sparse_backend_is_respected(monkeypatch):
    _set_capability(monkeypatch, 12, 0)

    attn_cls = dsv4_model._select_dsv4_attn_cls(
        _vllm_config(AttentionBackendEnum.FLASHMLA_SPARSE_DSV4)
    )

    assert attn_cls is dsv4_model.DeepseekV4FlashMLAAttention


def test_dsv4_flashinfer_sparse_head_padding_is_arch_specific():
    sm120_cls = dsv4_model.DeepseekV4FlashInferSM120Attention
    assert sm120_cls.get_padded_num_q_heads(16) == 16
    assert sm120_cls.get_padded_num_q_heads(17) == 32
    assert sm120_cls.get_padded_num_q_heads(65) == 128

    trtllm_cls = dsv4_model.DeepseekV4FlashInferTRTLLMAttention
    assert trtllm_cls.get_padded_num_q_heads(16) == 64
    assert trtllm_cls.get_padded_num_q_heads(65) == 128


def test_dsv4_flashinfer_kv_layout_is_arch_specific():
    sm120_attn = dsv4_flashinfer.DeepseekV4FlashInferSM120Attention.__new__(
        dsv4_flashinfer.DeepseekV4FlashInferSM120Attention
    )

    cache_config = SimpleNamespace(cache_dtype="fp8")
    kv_cache_dtype, torch_dtype = dsv4_attention._resolve_dsv4_kv_cache_dtype(
        sm120_attn._uses_fp8_ds_mla_layout(),
        cache_config.cache_dtype,
        cache_config,
    )
    assert kv_cache_dtype == "fp8_ds_mla"
    assert cache_config.cache_dtype == "fp8_ds_mla"
    assert torch_dtype == torch.uint8

    trtllm_attn = dsv4_flashinfer.DeepseekV4FlashInferTRTLLMAttention.__new__(
        dsv4_flashinfer.DeepseekV4FlashInferTRTLLMAttention
    )
    cache_config = SimpleNamespace(cache_dtype="fp8")
    kv_cache_dtype, torch_dtype = dsv4_attention._resolve_dsv4_kv_cache_dtype(
        trtllm_attn._uses_fp8_ds_mla_layout(),
        cache_config.cache_dtype,
        cache_config,
    )
    assert kv_cache_dtype == "fp8"
    assert cache_config.cache_dtype == "fp8"
    assert torch_dtype == torch.float8_e4m3fn


def test_flashinfer_sparse_sm120_accepts_auto_for_packed_kv_layout(monkeypatch):
    monkeypatch.setattr(
        flashinfer_utils, "has_flashinfer_sparse_mla_sm120", lambda: True
    )
    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_topk=2048)
            )
        ),
    )

    backend_cls = flashinfer_mla_sparse.FlashInferMLASparseSM120Backend

    for kv_cache_dtype in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
        assert (
            backend_cls.supports_combination(
                head_size=576,
                dtype=torch.bfloat16,
                kv_cache_dtype=kv_cache_dtype,
                block_size=64,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
                device_capability=DeviceCapability(12, 0),
            )
            is None
        )
        assert backend_cls.normalize_kv_cache_dtype(kv_cache_dtype) == "fp8_ds_mla"

    assert (
        backend_cls.supports_combination(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="bfloat16",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            device_capability=DeviceCapability(12, 0),
        )
        == "kv_cache_dtype not supported"
    )


def test_cuda_selector_returns_flashinfer_sparse_sm120_backend(monkeypatch):
    try:
        from vllm.platforms.cuda import CudaPlatform
    except (ImportError, ModuleNotFoundError):
        pytest.skip("CudaPlatform not available")

    monkeypatch.setattr(
        CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(12, 0)),
    )
    monkeypatch.setattr(
        flashinfer_utils, "has_flashinfer_sparse_mla_sm120", lambda: True
    )
    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_topk=2048)
            )
        ),
    )

    backend_path = CudaPlatform.get_attn_backend_cls(
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionSelectorConfig(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            use_sparse=True,
        ),
    )

    assert backend_path.endswith(".FlashInferMLASparseSM120Backend")


def test_dsv4_flashinfer_sm120_cache_view_for_fp8():
    fp8_cache = torch.empty((2, 3, 512), dtype=torch.float8_e4m3fn)
    sm120_cache = dsv4_flashinfer._as_sparse_sm120_cache(fp8_cache)

    assert sm120_cache.dtype == torch.uint8
    assert sm120_cache.shape == (2, 3, 1, 512)

    bf16_cache = torch.empty((2, 3, 512), dtype=torch.bfloat16)
    sm120_bf16_cache = dsv4_flashinfer._as_sparse_sm120_cache(bf16_cache)

    assert sm120_bf16_cache.dtype == torch.bfloat16
    assert sm120_bf16_cache.shape == (2, 3, 1, 512)


def test_dsv4_flashinfer_sm120_decode_keeps_singleton_query_axis(monkeypatch):
    class FakeSM120Wrapper:
        def __init__(self):
            self.call = None

        def run_sparse_mla(self, **kwargs):
            self.call = kwargs

    def fake_decode_scratch(num_tokens, num_heads, head_dim, topk, extra_topk=0):
        num_splits = 1 + int(extra_topk > 0)
        return (
            torch.empty((num_tokens, num_heads, num_splits, head_dim)),
            torch.empty((num_tokens, num_heads, num_splits)),
        )

    monkeypatch.setattr(dsv4_flashinfer, "_get_decode_scratch", fake_decode_scratch)

    wrapper = FakeSM120Wrapper()
    attn = dsv4_flashinfer.DeepseekV4FlashInferSM120Attention.__new__(
        dsv4_flashinfer.DeepseekV4FlashInferSM120Attention
    )
    object.__setattr__(attn, "kv_cache_torch_dtype", torch.bfloat16)
    object.__setattr__(attn, "scale", 1.0)
    object.__setattr__(attn, "attn_sink", torch.empty((4,), dtype=torch.float32))
    object.__setattr__(
        attn,
        "swa_cache_layer",
        SimpleNamespace(
            kv_cache=torch.empty((2, 4, 512), dtype=torch.bfloat16),
        ),
    )
    object.__setattr__(attn, "_sm120_wrapper", wrapper)

    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        decode_swa_indices=torch.zeros((2, 1, 5), dtype=torch.int32),
        decode_swa_lens=torch.ones((2,), dtype=torch.int32),
    )
    q = torch.empty((2, 3, 512), dtype=torch.bfloat16)
    output = torch.empty((2, 4, 512), dtype=torch.bfloat16)

    dsv4_flashinfer.DeepseekV4FlashInferSM120Attention._forward_sm120_decode(
        attn,
        q=q,
        kv_cache=None,
        swa_metadata=swa_metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )

    assert wrapper.call is not None
    assert wrapper.call["q"].shape == (2, 1, 4, 512)
    assert wrapper.call["kv_cache"].shape == (2, 4, 1, 512)
    assert wrapper.call["sparse_indices"].shape == (2, 1, 5)
    assert wrapper.call["out"].shape == (2, 4, 512)
