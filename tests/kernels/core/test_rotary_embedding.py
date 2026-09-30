# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for miscellaneous utilities."""

from unittest.mock import Mock

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum
from vllm.platforms.spec import PlatformSpec


def rotary_embedding_opcheck(
    rot,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding() is a in-place operation
    # that updates the query and key tensors.
    opcheck(
        torch.ops._C.rotary_embedding,
        (positions, query, key, rot.head_size, cos_sin_cache, rot.is_neox_style),
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contiguous", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_opcheck(
    default_vllm_config,
    dist_init,
    device,
    max_position,
    is_neox_style,
    rotary_dim,
    head_size,
    seq_len,
    use_key,
    head_stride_is_contiguous,
    dtype,
):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(
        head_size, rotary_dim, max_position, base, is_neox_style, dtype
    )

    positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)
    head_stride = head_size + (64 if head_stride_is_contiguous else 0)

    query = torch.randn(
        batch_size, seq_len, num_heads, head_stride, dtype=dtype, device=device
    )
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if key is not None else None

    rotary_embedding_opcheck(rot, positions, query, key)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contiguous:
        rotary_embedding_opcheck(
            rot,
            positions,
            query.flatten(start_dim=-2),
            key.flatten(start_dim=-2) if key is not None else None,
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("oot", [False, True])
def test_rope_spec_preserves_custom_op_dispatch(
    default_vllm_config, monkeypatch, enabled, oot
):
    """Disabling the op bypasses both platform kernels and legacy OOT overrides."""
    config = default_vllm_config.compilation_config
    config.custom_ops = ["none", "+rotary_embedding"] if enabled else ["none"]
    native = PlatformSpec().rope
    platform_call = Mock(wraps=native)
    oot_call = Mock(wraps=native)
    monkeypatch.setattr(
        type(current_platform),
        "spec",
        property(lambda _: PlatformSpec(rope=platform_call)),
    )
    monkeypatch.setattr(
        current_platform, "_enum", PlatformEnum.OOT if oot else PlatformEnum.CPU
    )
    if oot:

        class LegacyRoPE(RotaryEmbedding):
            def forward_oot(self, positions, query, key=None):
                return oot_call(
                    positions,
                    query,
                    key,
                    self.head_size,
                    self.rotary_dim,
                    self.cos_sin_cache,
                    self.is_neox_style,
                )

        monkeypatch.setitem(op_registry_oot, "RotaryEmbedding", LegacyRoPE)

    rope = RotaryEmbedding(8, 4, 16, 10000, True, torch.float32)
    positions, query = torch.tensor([1, 2]), torch.randn(2, 8)
    expected = rope.forward_native(positions, query)
    torch.testing.assert_close(rope(positions, query), expected)
    assert platform_call.call_count == int(enabled and not oot)
    assert oot_call.call_count == int(enabled and oot)
    assert ("rotary_embedding" in config.enabled_custom_ops) == enabled
    assert ("rotary_embedding" in config.disabled_custom_ops) == (not enabled)


@pytest.mark.parametrize("use_key", [False, True])
@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_spec_matches_native(
    default_vllm_config, use_key, is_neox_style, rotary_dim, dtype
):
    """The selected implementation preserves partial, strided and Q-only RoPE."""
    default_vllm_config.compilation_config.custom_ops = ["none", "+rotary_embedding"]
    device = current_platform.device_type
    rope = RotaryEmbedding(64, rotary_dim, 16, 10000, is_neox_style, dtype).to(device)
    positions = torch.tensor([1, 2], device=device)
    query = torch.randn(2, 3, 72, device=device, dtype=dtype)[..., :64]
    key = torch.randn(2, 1, 64, device=device, dtype=dtype) if use_key else None
    expected = rope.forward_native(positions, query, key)
    actual = rope(positions, query, key)
    torch.testing.assert_close(actual, expected)
    if use_key and (current_platform.is_cuda_alike() or current_platform.is_xpu()):
        assert actual[0] is query and actual[1] is key


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm platform policy")
def test_rope_spec_binds_rocm_policy_at_construction(default_vllm_config, monkeypatch):
    from vllm import _custom_ops as ops
    from vllm._aiter_ops import rocm_aiter_ops

    default_vllm_config.compilation_config.custom_ops = ["none", "+rotary_embedding"]
    native, aiter = Mock(), Mock()
    monkeypatch.setattr(ops, "rotary_embedding", native)
    monkeypatch.setattr(rocm_aiter_ops, "get_triton_rotary_embedding_op", lambda: aiter)
    monkeypatch.setattr(rocm_aiter_ops, "is_triton_rotary_embed_enabled", lambda: True)
    old = RotaryEmbedding(8, 8, 16, 10000, True, torch.float32)
    monkeypatch.setattr(rocm_aiter_ops, "is_triton_rotary_embed_enabled", lambda: False)
    new = RotaryEmbedding(8, 8, 16, 10000, True, torch.float32)

    positions, query = torch.tensor([0]), torch.zeros(1, 8)
    old(positions, query)
    new(positions, query)
    assert aiter.call_count == native.call_count == 1
    assert old.cos_sin_cache_alternate is not None
    assert old.cos_sin_cache_alternate.dtype == torch.bfloat16
    assert new.cos_sin_cache_alternate is None


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU fallback policy")
def test_rope_spec_reports_q_only_fallback(default_vllm_config, caplog):
    from vllm.logger import _print_warning_once

    default_vllm_config.compilation_config.custom_ops = ["none", "+rotary_embedding"]
    _print_warning_once.cache_clear()
    rope = RotaryEmbedding(8, 8, 16, 10000, True, torch.float32)
    positions, query = torch.tensor([0]), torch.randn(1, 8)
    for _ in range(2):
        torch.testing.assert_close(
            rope(positions, query), rope.forward_native(positions, query)
        )
    assert caplog.text.count("using native RoPE") == 1
    assert "does not support key=None" in caplog.text
