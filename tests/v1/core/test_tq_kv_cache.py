# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV cache support.

Covers TQFullAttentionSpec (kv_cache_interface) and the TQ branch of
unify_kv_cache_spec_page_size (kv_cache_utils), with Gemma4's heterogeneous
head-dim configuration (head_dim=256 for sliding-window layers that fall back
to fp8, and head_dim=512 for global full-attention layers that use TurboQuant).

These tests are CPU-only and do not require ROCm or a GPU.
"""

from dataclasses import replace

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    TQFullAttentionSpec,
    get_kv_quant_mode,
)

# TurboQuant preset used in all tests (matches the user's serve command).
_KV_DTYPE = "turboquant_k3v4_nc"
_QUANT_MODE = get_kv_quant_mode(_KV_DTYPE)
_DTYPE = torch.bfloat16


def _tq_spec(
    head_dim: int,
    block_size: int = 16,
    num_kv_heads: int = 8,
) -> TQFullAttentionSpec:
    """Build a TQFullAttentionSpec for the given head_dim."""
    tq = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, head_dim)
    return TQFullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=_DTYPE,
        kv_quant_mode=_QUANT_MODE,
        tq_slot_size=tq.slot_size_aligned,
    )


# ---------------------------------------------------------------------------
# TQFullAttentionSpec — unit tests (pure Python, no GPU)
# ---------------------------------------------------------------------------


class TestTQFullAttentionSpec:

    def test_real_page_size_bytes_uses_slot_size(self):
        """real_page_size_bytes == block_size * num_kv_heads * tq_slot_size."""
        spec = _tq_spec(head_dim=512, block_size=16, num_kv_heads=8)
        tq = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, 512)
        expected = 16 * 8 * tq.slot_size_aligned
        assert spec.real_page_size_bytes == expected

    def test_real_page_size_bytes_fallback_when_slot_zero(self):
        """tq_slot_size=0 falls back to parent FullAttentionSpec formula."""
        spec = TQFullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=_DTYPE,
            kv_quant_mode=_QUANT_MODE,
            tq_slot_size=0,
        )
        # FullAttentionSpec.real_page_size_bytes = block_size * num_kv_heads
        #   * (head_size + head_size_v) * dtype_size  (both KV heads, bf16)
        parent_bytes = spec.block_size * spec.num_kv_heads * 2 * 128 * 2
        assert spec.real_page_size_bytes == parent_bytes

    @pytest.mark.parametrize(
        "max_page, expected_bs",
        [
            # At 512*8=4096 bytes/block, max_page=65536 fits block_size=16
            # (16*4096=65536) and 32 (32*4096=131072 > 65536 → no).
            (65536, 16),
            # max_page=131072 fits block_size=32 (32*4096=131072 ✓).
            (131072, 32),
            # max_page=524288 fits block_size=128 (128*4096=524288 ✓).
            (524288, 128),
        ],
    )
    def test_largest_block_size_within(self, max_page, expected_bs):
        """largest_block_size_within picks the largest block that fits."""
        tq = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, 512)
        slot = tq.slot_size_aligned  # bytes per (head, token)
        num_kv_heads = 8
        spec = TQFullAttentionSpec(
            block_size=16,
            num_kv_heads=num_kv_heads,
            head_size=512,
            dtype=_DTYPE,
            kv_quant_mode=_QUANT_MODE,
            tq_slot_size=slot,
        )
        result = spec.largest_block_size_within(
            max_page, supported_sizes=[16, 32, 64, 128]
        )
        assert result == expected_bs

    def test_largest_block_size_within_nothing_fits_returns_current(self):
        """If nothing fits, fall back to the current block_size."""
        spec = _tq_spec(head_dim=512, block_size=16, num_kv_heads=8)
        # max_page smaller than even block_size=16
        result = spec.largest_block_size_within(1, [16, 32, 64, 128])
        assert result == 16  # current block_size as fallback

    def test_merge_preserves_tq_slot_size(self):
        """merge() must preserve tq_slot_size from the constituent specs."""
        spec_a = _tq_spec(head_dim=512, block_size=16, num_kv_heads=8)
        spec_b = _tq_spec(head_dim=512, block_size=16, num_kv_heads=8)
        merged = TQFullAttentionSpec.merge([spec_a, spec_b])
        assert isinstance(merged, TQFullAttentionSpec)
        assert merged.tq_slot_size == spec_a.tq_slot_size

    def test_merge_raises_on_mismatched_slot_size(self):
        """merge() must reject specs with different tq_slot_size values."""
        spec_256 = _tq_spec(head_dim=256)
        spec_512 = _tq_spec(head_dim=512)
        # Different head dims → different slot sizes → merge must fail.
        assert spec_256.tq_slot_size != spec_512.tq_slot_size
        with pytest.raises(AssertionError):
            TQFullAttentionSpec.merge([spec_256, spec_512])


# ---------------------------------------------------------------------------
# unify_kv_cache_spec_page_size — TQ heterogeneous head-dim (Gemma4 scenario)
# ---------------------------------------------------------------------------


class TestUnifyKVCacheSpecPageSizeTQ:
    """
    Gemma4-31B has two TQ attention groups:
      - Global (full-attention) layers: head_dim=512, 8 KV heads
      - Sliding-window layers: head_dim=256, but these fall back to fp8 so
        they are NOT TQFullAttentionSpec — only the global layers need unification.

    Here we test the more general case where two TQ groups have different
    slot sizes (e.g. a future model where both layer types use TurboQuant).
    """

    def test_uniform_tq_specs_unchanged(self):
        """All-same TQ specs are returned unmodified."""
        spec = _tq_spec(head_dim=512, block_size=16, num_kv_heads=8)
        specs = {"layer.0": spec, "layer.1": spec}
        assert kv_cache_utils.unify_kv_cache_spec_page_size(specs) == specs

    def test_tq_page_padded_to_max(self):
        """Two TQ specs with different real page sizes are unified to the same
        page_size_bytes after unify_kv_cache_spec_page_size."""
        spec_small = _tq_spec(head_dim=256, block_size=16, num_kv_heads=8)
        spec_large = _tq_spec(head_dim=512, block_size=16, num_kv_heads=8)

        # real_page_size_bytes differs between the two TQ groups.
        assert spec_small.real_page_size_bytes < spec_large.real_page_size_bytes

        unified = kv_cache_utils.unify_kv_cache_spec_page_size(
            {"sw_layer": spec_small, "global_layer": spec_large}
        )

        # After unification every layer must have the same page_size_bytes.
        assert (
            unified["sw_layer"].page_size_bytes
            == unified["global_layer"].page_size_bytes
        )
        # The unified page must be at least as large as the original larger spec.
        assert unified["global_layer"].page_size_bytes >= spec_large.page_size_bytes
        # Both remain TQFullAttentionSpec instances.
        assert isinstance(unified["sw_layer"], TQFullAttentionSpec)
        assert isinstance(unified["global_layer"], TQFullAttentionSpec)

    def test_tq_and_regular_attn_coexist(self):
        """TQ and plain FullAttentionSpec layers can share the same block pool.

        Note: TurboQuant compresses aggressively — real_page_size_bytes for a
        TQ layer can be *smaller* than a plain fp8 layer at the same head_size
        (fp8 stores 2 KV tensors in the native dtype, TQ packs K+V into ~1-2
        bytes/element).  The test checks that unification equalises page sizes
        regardless of which spec starts larger.
        """
        tq_spec = _tq_spec(head_dim=512, block_size=16, num_kv_heads=8)
        fp8_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=256,
            dtype=_DTYPE,
            kv_quant_mode=get_kv_quant_mode("fp8"),
        )

        unified = kv_cache_utils.unify_kv_cache_spec_page_size(
            {"sw_layer": fp8_spec, "global_layer": tq_spec}
        )
        # Both layers must end up with the same page_size_bytes.
        assert (
            unified["sw_layer"].page_size_bytes
            == unified["global_layer"].page_size_bytes
        )


# ---------------------------------------------------------------------------
# TurboQuantConfig — head-dim parametrisation (pure Python, no GPU)
# ---------------------------------------------------------------------------


class TestTurboQuantConfigHeadDim:
    """Verify that TurboQuantConfig is correctly parametric over head_dim."""

    @pytest.mark.parametrize("head_dim", [128, 256, 512])
    def test_slot_size_grows_with_head_dim(self, head_dim):
        """Slot size should scale with head_dim for a fixed preset."""
        tq = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, head_dim)
        assert tq.slot_size_aligned > 0
        assert tq.slot_size_aligned == tq.slot_size_aligned  # idempotent

    def test_slot_sizes_ordered(self):
        """Larger head_dim → larger slot size."""
        slot_128 = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, 128).slot_size_aligned
        slot_256 = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, 256).slot_size_aligned
        slot_512 = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, 512).slot_size_aligned
        assert slot_128 < slot_256 < slot_512

    @pytest.mark.parametrize("head_dim", [256, 512])
    def test_gemma4_head_dims_have_valid_slot_size(self, head_dim):
        """Gemma4 head dims produce valid TurboQuant configs."""
        tq = TurboQuantConfig.from_cache_dtype(_KV_DTYPE, head_dim)
        # slot_size_aligned must be even (required by SoA layout).
        assert tq.slot_size_aligned % 2 == 0
        # key + value packed sizes must fit within slot.
        assert tq.key_packed_size + tq.value_packed_size <= tq.slot_size_aligned
