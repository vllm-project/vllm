# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear.unquantized import packed_bf16_lm_head
from vllm.model_executor.kernels.linear.unquantized.packed_bf16_lm_head import (
    LosslessPackedLMHeadMethod,
    _packed_bf16_lm_head_impl,
    choose_launch_config,
    pack_bf16_weight,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the packed lm-head kernel is CUDA-only"
)


def test_lm_head_backend_config_normalizes_and_validates():
    """KernelConfig accepts CLI spelling and rejects unsafe storage limits."""
    config = KernelConfig(lm_head_backend=cast(Any, "lossless-packed"))

    assert config.lm_head_backend == "lossless_packed"
    with pytest.raises(ValueError):
        KernelConfig(lm_head_max_packed_fraction=0.0)


def _unpack_bits(tensors: tuple[torch.Tensor, ...], numel: int) -> torch.Tensor:
    sign_mantissa, exponent_nibbles, base_exponent, fallback_slot, fallback = tensors
    linear = torch.arange(sign_mantissa.numel(), device=sign_mantissa.device)
    block = linear // 256
    pair = exponent_nibbles[linear // 2].to(torch.int32)
    delta = (pair >> ((linear & 1) * 4)) & 0xF
    sm = sign_mantissa.to(torch.int32)
    bits = (
        ((sm & 0x80) << 8)
        | ((base_exponent[block].to(torch.int32) + delta) << 7)
        | (sm & 0x7F)
    )
    slots = fallback_slot[block].to(torch.int64)
    fallback_values = (
        fallback.flatten()[slots.clamp_min(0) * 256 + linear.remainder(256)].to(
            torch.int32
        )
        & 0xFFFF
    )
    return torch.where(slots >= 0, fallback_values, bits)[:numel].to(torch.int32)


@requires_cuda
@torch.inference_mode()
def test_packing_reconstructs_every_bf16_bit(monkeypatch):
    """Packed and fallback blocks together preserve every original BF16 bit."""
    monkeypatch.setattr(packed_bf16_lm_head, "_MIN_DENSE_BYTES", 0)
    weight = (torch.rand(521, 1023, device="cuda") + 0.5).to(torch.bfloat16)
    weight.view(-1)[:256] = torch.linspace(
        2**-20, 2**20, 256, dtype=torch.bfloat16, device="cuda"
    )

    packed = pack_bf16_weight(weight, max_packed_fraction=1.0)

    assert packed is not None
    plan, tensors = packed
    expected = weight.view(torch.int16).flatten().to(torch.int32) & 0xFFFF
    torch.testing.assert_close(
        _unpack_bits(tensors, plan.numel), expected, atol=0, rtol=0
    )
    assert plan.fallback_blocks > 0


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(33001, 1023), (8192, 2048)])
def test_single_token_projection_matches_torch(monkeypatch, n, k):
    """The fast path matches an unquantized projection at BF16 precision."""
    monkeypatch.setattr(packed_bf16_lm_head, "_MIN_DENSE_BYTES", 0)
    torch.manual_seed(7)
    magnitude = torch.rand(n, k, device="cuda") + 0.5
    signs = torch.where(
        torch.rand(n, k, device="cuda") > 0.5,
        torch.ones((), device="cuda"),
        -torch.ones((), device="cuda"),
    )
    weight = (magnitude * signs).to(torch.bfloat16)
    weight.view(-1)[:256] = torch.linspace(
        2**-20, 2**20, 256, dtype=torch.bfloat16, device="cuda"
    )
    x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
    packed = pack_bf16_weight(weight, max_packed_fraction=1.0)

    assert packed is not None
    plan, tensors = packed
    sign_mantissa, exponent_nibbles, base_exponent, fallback_slot, fallback = tensors
    launch = choose_launch_config(k)
    actual = _packed_bf16_lm_head_impl(
        x,
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback,
        plan.n,
        plan.k,
        launch.block_n,
        launch.num_warps,
    )
    expected = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)


class _FallbackMethod:
    interface_marker = "wrapped interface"

    def apply(self, layer, x, bias=None):
        del layer, bias
        return x + 1


def test_method_falls_back_for_non_cuda_input():
    """Unsupported runtime shapes retain the wrapped method's behavior."""
    method = LosslessPackedLMHeadMethod(_FallbackMethod())  # type: ignore[arg-type]
    layer = SimpleNamespace(_vllm_lossless_lm_head_meta=(8, 4, 2, 1))
    x = torch.zeros(2, 4)

    torch.testing.assert_close(method.apply(layer, x), x + 1)


def test_method_preserves_wrapped_interface():
    """The decorator exposes attributes implemented by its fallback."""
    method = LosslessPackedLMHeadMethod(_FallbackMethod())  # type: ignore[arg-type]

    assert method.interface_marker == "wrapped interface"


def test_batch_invariant_mode_skips_packed_state(monkeypatch):
    """Batch-invariant execution must always retain the stock linear path."""

    class _PackedStateMustNotBeRead:
        @property
        def _vllm_lossless_lm_head_meta(self):
            raise AssertionError("packed state was inspected")

    monkeypatch.setattr(packed_bf16_lm_head.envs, "VLLM_BATCH_INVARIANT", True)

    assert (
        packed_bf16_lm_head._try_apply_packed_weight(
            _PackedStateMustNotBeRead(), torch.zeros(1, 4), None
        )
        is None
    )


def test_parallel_lm_head_default_is_not_wrapped():
    """The default configuration does not alter the lm-head method."""
    config = VllmConfig(kernel_config=KernelConfig(lm_head_backend="torch"))
    with set_current_vllm_config(config):
        head = ParallelLMHead(64, 16, disable_tp=True)

    assert not isinstance(head.quant_method, LosslessPackedLMHeadMethod)


@requires_cuda
def test_parallel_lm_head_wraps_eligible_method_when_requested():
    """An eligible CUDA BF16 head receives the requested method decorator."""
    config = VllmConfig(kernel_config=KernelConfig(lm_head_backend="lossless_packed"))
    with set_current_vllm_config(config):
        head = ParallelLMHead(64, 16, params_dtype=torch.bfloat16, disable_tp=True)

    assert isinstance(head.quant_method, LosslessPackedLMHeadMethod)
