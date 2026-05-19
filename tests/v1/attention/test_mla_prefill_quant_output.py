# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend fused-quant-output capability gating.

Covers `MLAPrefillBackend.supports_quant_output`, which decides whether the
prefill kernel writes quantized output directly (FA4 native fused FP8, see
flash-attention#135) instead of going through the post-quant path.
"""

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)

_FA_MODULE = "vllm.v1.attention.backends.mla.prefill.flash_attn"


class _DummyPrefillBackend(MLAPrefillBackend):
    """Concrete backend that does NOT override supports_quant_output."""

    @staticmethod
    def get_name() -> str:
        return "DUMMY"

    def run_prefill_new_tokens(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def run_prefill_context_chunk(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.parametrize(
    "quant_key", [kFp8StaticTensorSym, kFp8Dynamic128Sym, kNvfp4Dynamic, None]
)
def test_base_backend_never_supports_quant_output(quant_key):
    """The base default opts every backend out unless it overrides."""
    backend = object.__new__(_DummyPrefillBackend)
    assert backend.supports_quant_output(quant_key) is False


def _make_fa_backend(version: int | None, is_vllm_fa: bool):
    """Build a FlashAttnPrefillBackend without running its heavy __init__."""
    backend = object.__new__(FlashAttnPrefillBackend)
    backend.vllm_flash_attn_version = version
    backend._is_vllm_fa = is_vllm_fa
    return backend


@pytest.mark.parametrize(
    ("version", "is_vllm_fa", "dc_major", "quant_key", "expected"),
    [
        # FA4 + vLLM-FA + Blackwell SM100/SM110 + static FP8 -> fused.
        (4, True, 10, kFp8StaticTensorSym, True),
        (4, True, 11, kFp8StaticTensorSym, True),
        # Wrong compute capability (SM90 / SM120) -> not supported (#135).
        (4, True, 9, kFp8StaticTensorSym, False),
        (4, True, 12, kFp8StaticTensorSym, False),
        # Not FA4.
        (3, True, 10, kFp8StaticTensorSym, False),
        (2, True, 10, kFp8StaticTensorSym, False),
        (None, True, 10, kFp8StaticTensorSym, False),
        # Upstream (ROCm) flash-attn, not vLLM-FA.
        (4, False, 10, kFp8StaticTensorSym, False),
        # Quant keys not wired through FA4 yet.
        (4, True, 10, kFp8Dynamic128Sym, False),
        (4, True, 10, kNvfp4Dynamic, False),
    ],
)
def test_flash_attn_supports_quant_output(
    version, is_vllm_fa, dc_major, quant_key, expected
):
    backend = _make_fa_backend(version, is_vllm_fa)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = DeviceCapability(
            major=dc_major, minor=0
        )
        assert backend.supports_quant_output(quant_key) is expected


def test_flash_attn_supports_quant_output_unknown_device():
    """A None device capability (e.g. capability probe failed) is safe."""
    backend = _make_fa_backend(version=4, is_vllm_fa=True)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = None
        assert backend.supports_quant_output(kFp8StaticTensorSym) is False


def test_flash_attn_prefill_backend_signature_accepts_fused_kwargs():
    """run_prefill_new_tokens must accept out/output_scale so the direct
    (non-**kwargs) call in forward_mha type- and runtime-checks."""
    import inspect

    params = inspect.signature(
        FlashAttnPrefillBackend.run_prefill_new_tokens
    ).parameters
    assert "out" in params
    assert "output_scale" in params
    # The base contract must expose them too (Liskov / direct call site).
    base_params = inspect.signature(MLAPrefillBackend.run_prefill_new_tokens).parameters
    assert "out" in base_params
    assert "output_scale" in base_params


def test_mla_impl_forward_mha_accepts_output_scale():
    """The abstract MLA impl forward_mha must carry output_scale so every
    override (and the unconditional forward_impl call) stays compatible."""
    import inspect

    from vllm.v1.attention.backend import MLAAttentionImpl

    params = inspect.signature(MLAAttentionImpl.forward_mha).parameters
    assert "output_scale" in params
    assert params["output_scale"].default is None
