# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel/tile selection for the MXFP8 32x32-block GEMM (``rocm_block32_gemm``).

Pure CPU tests for which ``_Config`` a shape and token count pick. The packed
kernel fills the MFMA rows with K panels instead of tokens and wins while there
are only a handful of rows; the per-shape ``_TUNED`` table switches to the tiled
kernel above small M. ``_PACKED_DECODE_MAX_M`` keeps packed through the decode
width (DSpark-inflated M lands in the table's tiled range), so these pin that
the override keeps packed at and below the cutoff and defers to the table above.
"""

from vllm.model_executor.kernels.linear.mxfp8 import rocm_block32_gemm as g

# Indexer wq_b at TP4: packed tiers through M=32, then tiled from M=64 up.
_N, _K = 4096, 1280


def test_packed_cutoff_default_is_tuned_value():
    assert g._PACKED_DECODE_MAX_M == 64


def test_tuned_shape_keeps_packed_through_cutoff():
    # At M=64 the table's own tier is tiled, but M <= cutoff keeps the shape's
    # largest packed tier instead.
    assert g._config(32, _N, _K).packed, "M=32 is a packed tier in the table"
    assert g._config(48, _N, _K).packed, "48 <= cutoff: keep the packed tier"
    assert g._config(64, _N, _K).packed, "64 == cutoff: keep the packed tier"


def test_tuned_shape_defers_to_table_above_cutoff():
    assert not g._config(65, _N, _K).packed, "above the cutoff: trust the table"
    assert not g._config(512, _N, _K).packed


def test_kept_packed_tier_is_the_largest_packed():
    tiers = g._TUNED[(_N, _K)]
    largest_packed = g._largest_packed(tiers)
    assert largest_packed is not None and largest_packed.packed
    # _config at the cutoff returns exactly that tier.
    assert g._config(64, _N, _K) == largest_packed


def test_packed_cutoff_env_override_trusts_table(monkeypatch):
    # _config reads the module global each call, so patching it is equivalent
    # to VLLM_PACKED_DECODE_MAX_M=0 without re-importing (which would re-register
    # the custom op).
    monkeypatch.setattr(g, "_PACKED_DECODE_MAX_M", 0)
    assert not g._config(64, _N, _K).packed, "cutoff 0 fully trusts the table"


def test_default_config_untuned_shape_tiers():
    # Shape not in _TUNED: packed through the cutoff, then the tiled fallbacks.
    assert g._default_config(8, 999, 999).packed
    assert g._default_config(64, 999, 999).packed
    mid = g._default_config(200, 999, 999)
    assert not mid.packed and (mid.block_m, mid.block_n) == (64, 64)
    big = g._default_config(300, 999, 999)
    assert not big.packed and (big.block_m, big.block_n) == (128, 128)
