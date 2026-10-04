# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU-specific tests for the FlashAttnMLA backend."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import try_get_attention_backend
from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig


class _ForceXPUPlatform:
    """``current_platform`` stand-in that enables the XPU paths off-device."""

    @staticmethod
    def is_xpu() -> bool:
        return True


def _import_flashattn_mla():
    """Return the FlashAttnMLA module, skipping if flash-attn is unavailable."""
    _, impl_cls = try_get_attention_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
    return sys.modules[impl_cls.__module__]


def test_flashattn_mla_xpu_gating_matches_kernel_support(monkeypatch):
    """XPU only compiles the 576-wide MLA head at block_size=64."""
    flashattn_mla_module = _import_flashattn_mla()
    monkeypatch.setattr(flashattn_mla_module, "current_platform", _ForceXPUPlatform())

    backend = flashattn_mla_module.FlashAttnMLABackend
    assert [size.base for size in backend.get_supported_kernel_block_sizes()] == [64]
    assert backend.get_supported_head_sizes() == [576]
    assert backend.supports_block_size(64)
    assert not backend.supports_block_size(32)
    assert backend.supports_compute_capability(DeviceCapability(1, 0))


@pytest.mark.parametrize("q_is_tuple", [True, False])
def test_flashattn_mla_xpu_decode_packs_query_and_narrows_value(
    monkeypatch, q_is_tuple
):
    """XPU decode sends one 576-wide query per request against a 512-wide value."""
    flashattn_mla_module = _import_flashattn_mla()
    monkeypatch.setattr(flashattn_mla_module, "current_platform", _ForceXPUPlatform())

    captured: dict = {}
    expected_out = torch.empty(2, 4, 512)

    def fake_flash_attn_varlen_func(q, k, v, **kwargs):
        captured.update(kwargs, q=q, k=k, v=v)
        return expected_out

    monkeypatch.setattr(
        flashattn_mla_module, "flash_attn_varlen_func", fake_flash_attn_varlen_func
    )

    query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    seq_lens = torch.tensor([9, 17], dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_decodes=2,
        query_start_loc=query_start_loc,
        decode=SimpleNamespace(
            max_seq_len=17,
            seq_lens=seq_lens,
            block_table=block_table,
            query_start_loc=query_start_loc[:3],
        ),
    )
    impl = flashattn_mla_module.FlashAttnMLAImpl.__new__(
        flashattn_mla_module.FlashAttnMLAImpl
    )
    impl.kv_cache_dtype = "auto"
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.scale = 0.125

    q_nope, q_pe = torch.randn(2, 4, 512), torch.randn(2, 4, 64)
    q = (q_nope, q_pe) if q_is_tuple else torch.cat([q_nope, q_pe], dim=-1)

    out, lse = impl.forward_mqa(
        q,
        torch.randn(3, 64, 576),
        attn_metadata,
        SimpleNamespace(),
    )

    assert out is expected_out
    assert lse is None
    assert captured["q"].shape == (2, 4, 576)
    torch.testing.assert_close(captured["q"], torch.cat([q_nope, q_pe], dim=-1))
    if not q_is_tuple:
        assert captured["q"] is q
    assert captured["k"].shape == (3, 64, 1, 576)
    assert captured["v"].shape == (3, 64, 1, 512)
    torch.testing.assert_close(captured["cu_seqlens_q"], query_start_loc[:3])
    assert captured["max_seqlen_q"] == 1
    assert captured["causal"] is False


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU-only builder config")
def test_flashattn_mla_xpu_builder_is_single_token_decode_only():
    """``_forward_mqa_xpu`` hardcodes ``max_seqlen_q=1``; the builder must agree."""
    builder = _import_flashattn_mla().FlashAttnMLAMetadataBuilder

    assert builder.query_len_support == QueryLenSupport.SINGLE_ONLY
    assert builder.reorder_batch_threshold == 1
    assert builder._cudagraph_support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_xpu_platform_routes_mla_to_flash_attn_mla():
    """XPU MLA defaults to FlashAttnMLA but still honours an explicit TritonMLA."""
    pytest.importorskip("vllm_xpu_kernels")
    from vllm.platforms.xpu import XPUPlatform

    selector_config = AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=64,
        use_mla=True,
    )

    for selected_backend in (None, AttentionBackendEnum.FLASH_ATTN_MLA):
        assert (
            XPUPlatform.get_attn_backend_cls(
                selected_backend=selected_backend,
                attn_selector_config=selector_config,
            )
            == AttentionBackendEnum.FLASH_ATTN_MLA.get_path()
        )

    assert (
        XPUPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.TRITON_MLA,
            attn_selector_config=selector_config,
        )
        == AttentionBackendEnum.TRITON_MLA.get_path()
    )

    with pytest.raises(ValueError, match="Invalid attention backend"):
        XPUPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.FLASH_ATTN,
            attn_selector_config=selector_config,
        )


def test_xpu_varlen_attn_allocates_output_with_value_head_size(monkeypatch):
    """MLA passes a 576-wide query with a 512-wide value; ``out`` must follow value."""
    pytest.importorskip("vllm_xpu_kernels")
    xpu_ops_module = pytest.importorskip("vllm._xpu_ops")

    monkeypatch.setattr(
        xpu_ops_module, "flash_attn_varlen_func", lambda **kwargs: kwargs["out"]
    )

    out = xpu_ops_module.xpu_ops.flash_attn_varlen_func(
        q=torch.randn(2, 4, 576),
        k=torch.randn(3, 64, 1, 576),
        v=torch.randn(3, 64, 1, 512),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=17,
        seqused_k=torch.tensor([9, 17], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
    )

    assert out.shape == (2, 4, 512)
