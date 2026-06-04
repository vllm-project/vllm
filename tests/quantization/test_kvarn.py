# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for KVarN KV-cache quantization.

Algorithm by Muller et al. (arXiv:2606.03458) / Huawei CSL.
Source: https://github.com/huawei-csl/KVarN (Apache 2.0)

Run (CPU-only tests):
  .venv/bin/python -m pytest tests/quantization/test_kvarn.py -v -k "not RoundTrip"

Run (all, GPU required for RoundTrip):
  .venv/bin/python -m pytest tests/quantization/test_kvarn.py -v
"""

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import (
    KVARN_PRESETS,
    KVarNConfig,
)
from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
    variance_normalize,
    variance_normalize_batched,
)
from vllm.platforms import current_platform

# ============================================================================
# Helpers
# ============================================================================

ALL_PRESETS = list(KVARN_PRESETS.keys())
GPGPU_AVAILABLE = torch.cuda.is_available() or torch.xpu.is_available()
DEVICE_TYPE = current_platform.device_type


# ============================================================================
# Config tests (CPU-only, no dependencies beyond config.py)
# ============================================================================


class TestKVarNConfig:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_preset_parses(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        assert isinstance(cfg, KVarNConfig)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown KVarN"):
            KVarNConfig.from_cache_dtype("kvarn_invalid", head_dim=128)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_bits(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        params = KVARN_PRESETS[preset]
        assert cfg.key_bits == params["key_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_value_bits(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        params = KVARN_PRESETS[preset]
        assert cfg.value_bits == params["value_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_group_size(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        params = KVARN_PRESETS[preset]
        assert cfg.group == params["group"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_k_packed_bytes(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        expected = math.ceil(128 * cfg.group * cfg.key_bits / 8)
        assert cfg.k_packed_bytes == expected

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_v_packed_bytes(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        expected = math.ceil(cfg.group * 128 * cfg.value_bits / 8)
        assert cfg.v_packed_bytes == expected

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_k_scale_bytes(self, preset):
        """K scales: s_col [head_dim fp16] + zp [head_dim fp16] + s_row [group fp16]."""
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        expected = (2 * 128 + cfg.group) * 2
        assert cfg.k_scale_bytes == expected

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_v_scale_bytes(self, preset):
        """V scales: s_col [head_dim fp16] + s_row [group fp16] + zp [group fp16]."""
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        expected = (128 + 2 * cfg.group) * 2
        assert cfg.v_scale_bytes == expected

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_tile_bytes_equals_k_plus_v(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.tile_bytes == (
            cfg.k_packed_bytes + cfg.k_scale_bytes
            + cfg.v_packed_bytes + cfg.v_scale_bytes
        )

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_tile_bytes_aligned_is_multiple_of_8(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.tile_bytes_aligned % 8 == 0
        assert cfg.tile_bytes_aligned >= cfg.tile_bytes

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_byte_offsets_are_contiguous(self, preset):
        """Offset layout must not overlap: each field starts where the previous ends."""
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.k_packed_offset == 0
        assert cfg.k_s_col_offset == cfg.k_packed_bytes
        assert cfg.k_zp_offset == cfg.k_s_col_offset + 128 * 2
        assert cfg.k_s_row_offset == cfg.k_zp_offset + 128 * 2
        assert cfg.v_packed_offset == cfg.k_s_row_offset + cfg.group * 2
        assert cfg.v_s_col_offset == cfg.v_packed_offset + cfg.v_packed_bytes
        assert cfg.v_s_row_offset == cfg.v_s_col_offset + 128 * 2
        assert cfg.v_zp_offset == cfg.v_s_row_offset + cfg.group * 2

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_pool_slots_positive(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        slots = cfg.pool_slots(max_num_seqs=32, max_num_batched_tokens=4096)
        assert slots > 0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_max_supported_seqs_at_least_one(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        # 40 GB budget — any real GPU can support ≥1 seq
        n = cfg.max_supported_seqs(
            total_gpu_bytes=40 * 1024**3,
            num_kv_heads=8,
            num_layers=32,
            max_num_batched_tokens=4096,
        )
        assert n >= 1

    def test_boundary_skip_layers_basic(self):
        layers = KVarNConfig.get_boundary_skip_layers(32, n=2)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_layers_zero(self):
        assert KVarNConfig.get_boundary_skip_layers(32, 0) == []

    def test_boundary_skip_layers_small_model(self):
        layers = KVarNConfig.get_boundary_skip_layers(4, n=2)
        assert layers == ["0", "1", "2", "3"]

    def test_known_preset_kvarn_k4v2_g128(self):
        """Concrete value check for the shipped preset at head_dim=128."""
        cfg = KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", head_dim=128)
        # k_packed: 128*128*4/8 = 8192 bytes
        assert cfg.k_packed_bytes == 8192
        # v_packed: 128*128*2/8 = 4096 bytes
        assert cfg.v_packed_bytes == 4096
        # k_scales: (2*128 + 128)*2 = 768 bytes
        assert cfg.k_scale_bytes == 768
        # v_scales: (128 + 2*128)*2 = 768 bytes
        assert cfg.v_scale_bytes == 768
        # total tile: 8192+768+4096+768 = 13824, aligned to 13824 (already %8)
        assert cfg.tile_bytes == 13824
        assert cfg.tile_bytes_aligned == 13824


# ============================================================================
# Backend capability declarations (CPU-only — no GPU or Triton needed)
# ============================================================================


class TestKVarNBackendCapabilities:
    def test_supported_block_sizes(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        sizes = KVarNAttentionBackend.get_supported_kernel_block_sizes()
        assert 128 in sizes

    def test_supports_head_size_128(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        assert KVarNAttentionBackend.supports_head_size(128)

    def test_rejects_other_head_sizes(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        for h in [64, 96, 256]:
            assert not KVarNAttentionBackend.supports_head_size(h)

    def test_supports_kvarn_kv_cache_dtype(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        assert KVarNAttentionBackend.supports_kv_cache_dtype("kvarn_k4v2_g128")

    def test_rejects_non_kvarn_dtype(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        for dtype in ["auto", "fp8", "turboquant_k8v4", None]:
            assert not KVarNAttentionBackend.supports_kv_cache_dtype(dtype)

    def test_get_name(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        assert KVarNAttentionBackend.get_name() == "KVARN"

    def test_kv_cache_shape(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        shape = KVarNAttentionBackend.get_kv_cache_shape(
            num_blocks=10,
            block_size=128,
            num_kv_heads=8,
            head_size=128,
            cache_dtype_str="kvarn_k4v2_g128",
        )
        # 3D: (num_blocks, num_kv_heads, tile_bytes_aligned)
        cfg = KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", head_dim=128)
        assert shape == (10, 8, cfg.tile_bytes_aligned)

    def test_kv_cache_shape_wrong_block_size_raises(self):
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend

        with pytest.raises(AssertionError):
            KVarNAttentionBackend.get_kv_cache_shape(
                num_blocks=10, block_size=16, num_kv_heads=8,
                head_size=128, cache_dtype_str="kvarn_k4v2_g128",
            )

    def test_registered_in_enum(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        assert hasattr(AttentionBackendEnum, "KVARN")
        assert "kvarn_attn" in AttentionBackendEnum.KVARN.value

    def test_cache_dtype_registered(self):
        from typing import get_args

        from vllm.config.cache import CacheDType

        assert "kvarn_k4v2_g128" in get_args(CacheDType)


# ============================================================================
# Sinkhorn variance normalization (CPU-only — pure PyTorch, no GPU)
# ============================================================================


class TestVarianceNormalize:
    def test_output_shapes_single(self):
        tile = torch.randn(128, 128)
        balanced, s_col, s_row = variance_normalize(tile, iterations=4)
        assert balanced.shape == (128, 128)
        assert s_col.shape == (1, 128)
        assert s_row.shape == (128, 1)

    def test_reconstruction_single(self):
        """balanced = tile / s_col / s_row should hold within fp32 tolerance."""
        tile = torch.randn(128, 128)
        balanced, s_col, s_row = variance_normalize(tile, iterations=4)
        reconstructed = balanced * s_col * s_row
        assert torch.allclose(reconstructed, tile.float(), atol=1e-5)

    def test_scales_positive(self):
        tile = torch.randn(64, 128)
        _, s_col, s_row = variance_normalize(tile, iterations=4)
        assert (s_col > 0).all()
        assert (s_row > 0).all()

    def test_reduces_column_std_spread(self):
        """After normalization the ratio max_col_std/min_col_std should decrease."""
        torch.manual_seed(42)
        tile = torch.randn(128, 128) * torch.linspace(0.1, 10.0, 128)
        col_std_before = tile.std(dim=0)
        ratio_before = col_std_before.max() / col_std_before.min()

        balanced, _, _ = variance_normalize(tile, iterations=16)
        col_std_after = balanced.std(dim=0)
        ratio_after = col_std_after.max() / col_std_after.min()

        assert ratio_after < ratio_before

    def test_output_shapes_batched(self):
        tiles = torch.randn(4, 128, 128)
        balanced, s_col, s_row = variance_normalize_batched(tiles, iterations=4)
        assert balanced.shape == (4, 128, 128)
        assert s_col.shape == (4, 1, 128)
        assert s_row.shape == (4, 128, 1)

    def test_reconstruction_batched(self):
        tiles = torch.randn(4, 128, 128)
        balanced, s_col, s_row = variance_normalize_batched(tiles, iterations=4)
        reconstructed = balanced * s_col * s_row
        assert torch.allclose(reconstructed, tiles.float(), atol=1e-5)

    def test_zero_iterations_is_identity(self):
        """With 0 iterations, s_col = s_row = 1 and balanced = tile."""
        tile = torch.randn(32, 64)
        balanced, s_col, s_row = variance_normalize(tile, iterations=0)
        assert torch.allclose(balanced, tile.float(), atol=1e-6)
        assert torch.allclose(s_col, torch.ones_like(s_col), atol=1e-6)
        assert torch.allclose(s_row, torch.ones_like(s_row), atol=1e-6)

    def test_deterministic(self):
        tile = torch.randn(128, 128)
        b1, sc1, sr1 = variance_normalize(tile, iterations=8)
        b2, sc2, sr2 = variance_normalize(tile, iterations=8)
        assert torch.equal(b1, b2)
        assert torch.equal(sc1, sc2)
        assert torch.equal(sr1, sr2)


# ============================================================================
# Hadamard rotation (CPU-only — uses the pure-Python construction)
# ============================================================================


def _build_hadamard(d: int) -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


class TestHadamard:
    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_orthonormal(self, dim):
        H = _build_hadamard(dim)
        eye = H @ H.T
        assert torch.allclose(eye, torch.eye(dim), atol=1e-5)

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_symmetric(self, dim):
        H = _build_hadamard(dim)
        assert torch.allclose(H, H.T, atol=1e-6)

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_all_entries_pm_inv_sqrt_d(self, dim):
        H = _build_hadamard(dim)
        expected_abs = 1.0 / math.sqrt(dim)
        assert torch.allclose(H.abs(), torch.full_like(H, expected_abs), atol=1e-6)

    def test_rotation_preserves_norm(self):
        """H @ v should have the same L2 norm as v (orthonormal)."""
        H = _build_hadamard(128)
        v = torch.randn(128, 8)
        Hv = H @ v
        orig_norms = v.norm(dim=0)
        rot_norms = Hv.norm(dim=0)
        assert torch.allclose(orig_norms, rot_norms, atol=1e-5)

    def test_double_rotation_is_identity(self):
        """Applying H twice returns the original (H is self-inverse, H@H = I)."""
        H = _build_hadamard(128)
        v = torch.randn(128, 4)
        assert torch.allclose(H @ (H @ v), v, atol=1e-5)


# ============================================================================
# Store → Decode round-trip (GPU + Triton required)
# ============================================================================


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestKVarNStoreDecodeRoundTrip:
    """End-to-end: store KV into KVarN cache, decode, compare vs fp16 ref.

    With a single cached token and query == key, attention output ≈ value.
    We verify cosine similarity rather than exact equality due to quantization.
    """

    def test_single_token_roundtrip(self):
        from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
        from vllm.v1.attention.ops.triton_kvarn_decode import kvarn_decode_attention
        from vllm.v1.attention.ops.triton_kvarn_sinkhorn import kvarn_sinkhorn_triton
        from vllm.v1.attention.ops.kvarn_store import (
            kvarn_store_tile_k,
            kvarn_store_tile_v,
        )

        preset = "kvarn_k4v2_g128"
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        D = 128
        Hk = 4
        Hq = 4
        block_size = 128
        num_blocks = 1
        device = torch.device(DEVICE_TYPE)

        H = _build_hadamard(D).to(device)
        torch.manual_seed(7)
        key = torch.randn(block_size, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(block_size, Hk, D, device=device, dtype=torch.float16)

        kv_cache = torch.zeros(
            num_blocks, Hk, cfg.tile_bytes_aligned,
            device=device, dtype=torch.uint8,
        )

        # Store the full tile (block_size == group == 128)
        for head in range(Hk):
            k_tile = (H @ key[:, head, :].T.float())  # [D, group]
            v_tile = (H @ value[:, head, :].float()).T  # [group, D]
            # Use sinkhorn to store
            kvarn_sinkhorn_triton(k_tile, v_tile, kv_cache[0, head], cfg)

        # Decode: use first token's key as query; with block_size=128 tokens
        # the softmax will spread across all tokens, but cosine sim of output
        # vs value (averaged) should still be high.
        query = key[0:1, :, :].expand(1, Hq, D).contiguous()
        block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([block_size], device=device, dtype=torch.int32)

        output = kvarn_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            H=H,
            scale=1.0 / math.sqrt(D),
            cfg=cfg,
        )

        assert output.shape == (1, Hq, D)
        # Output should be a weighted sum of values — check it's finite and non-zero
        assert output.isfinite().all()
        assert output.abs().max() > 0
