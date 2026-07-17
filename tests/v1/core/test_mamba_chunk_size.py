# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""get_mamba_chunk_size resolution tests.

GDN / gated-delta models (e.g. Qwen3-Next) carry no mamba_chunk_size/chunk_size
config field; before the GDN branch they fell through to the Mamba1 2048
default, which the all-mode block-size resolution then used as base_chunk_size,
resolving a 3.5x-oversized mamba cache block (2048 instead of 576 for
Qwen3-Next-80B at TP4). The FLA prefill kernel's real tile is FLA_CHUNK_SIZE
(64), a hard kernel constant.
"""

from types import SimpleNamespace

from vllm.config import ModelConfig
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE


def _cfg(**fields):
    """Minimal stand-in exposing only hf_text_config (all the method reads)."""
    return SimpleNamespace(hf_text_config=SimpleNamespace(**fields))


def test_explicit_mamba_chunk_size_honored():
    """REGRESSION: Bamba/FalconH1-style mamba_chunk_size is returned as-is."""
    assert ModelConfig.get_mamba_chunk_size(_cfg(mamba_chunk_size=128)) == 128


def test_explicit_chunk_size_honored():
    """REGRESSION: Mamba2/NemotronH-style chunk_size is returned as-is."""
    assert ModelConfig.get_mamba_chunk_size(_cfg(chunk_size=256)) == 256


def test_mamba1_fallback_unchanged():
    """REGRESSION: no chunk fields and no GDN marker -> Mamba1 default 2048."""
    assert ModelConfig.get_mamba_chunk_size(_cfg()) == 2048


def test_gdn_marker_returns_fla_chunk_size():
    """NEW: GDN models (linear_conv_kernel_dim present, no chunk fields)
    resolve to the FLA kernel tile, not the Mamba1 default."""
    assert FLA_CHUNK_SIZE == 64
    assert (ModelConfig.get_mamba_chunk_size(
        _cfg(linear_conv_kernel_dim=4)) == FLA_CHUNK_SIZE)


def test_gdn_marker_with_explicit_chunk_keeps_explicit():
    """NEW: an explicit chunk field wins over the GDN marker."""
    assert ModelConfig.get_mamba_chunk_size(
        _cfg(linear_conv_kernel_dim=4, chunk_size=256)) == 256
