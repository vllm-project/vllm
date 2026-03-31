# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for TurboQuant attention backend.

Tests the full paged-cache encode→decode roundtrip for all bit-widths,
lite mode, asymmetric K/V, and the 3-bit packing helpers.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = torch.device("cuda")


# -----------------------------------------------------------------------
# 3-bit packing helpers
# -----------------------------------------------------------------------


class TestPack3Bit:
    """Test _pack_3bit_vectorized / _unpack_3bit_vectorized roundtrip."""

    @pytest.mark.parametrize("normal_size", [109, 120, 128, 96, 80])
    def test_roundtrip(self, normal_size):
        from vllm.v1.attention.backends.turboquant_attn import (
            _pack_3bit_vectorized,
            _unpack_3bit_vectorized,
        )

        N = 16
        packed_bytes = math.ceil(normal_size * 3 / 8)
        indices = torch.randint(
            0, 8, (N, normal_size), device=DEVICE, dtype=torch.uint8
        )
        packed = _pack_3bit_vectorized(indices, normal_size, packed_bytes)
        unpacked = _unpack_3bit_vectorized(packed, normal_size, DEVICE)

        assert torch.equal(indices, unpacked[:N, :normal_size]), (
            f"3-bit pack/unpack roundtrip failed for normal_size={normal_size}"
        )

    def test_boundary_values(self):
        """All 8 possible 3-bit values survive packing."""
        from vllm.v1.attention.backends.turboquant_attn import (
            _pack_3bit_vectorized,
            _unpack_3bit_vectorized,
        )

        normal_size = 10
        packed_bytes = math.ceil(normal_size * 3 / 8)
        # Row with all 8 distinct values
        row = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1],
            device=DEVICE,
            dtype=torch.uint8,
        ).unsqueeze(0)
        packed = _pack_3bit_vectorized(row, normal_size, packed_bytes)
        unpacked = _unpack_3bit_vectorized(packed, normal_size, DEVICE)
        assert torch.equal(row, unpacked[:1, :normal_size])


# -----------------------------------------------------------------------
# Bit-pack roundtrip (core module)
# -----------------------------------------------------------------------


class TestPackIndices:
    """Test pack_indices / unpack_indices for all bit-widths."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_elements", [109, 128, 256, 1])
    def test_roundtrip(self, bits, n_elements):
        from vllm.model_executor.layers.quantization.turboquant import (
            pack_indices,
            unpack_indices,
        )

        max_val = (1 << bits) - 1
        indices = torch.randint(
            0, max_val + 1, (n_elements,), device=DEVICE, dtype=torch.int32
        )
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, n_elements)
        assert torch.equal(indices.long(), unpacked.long()), (
            f"pack/unpack roundtrip failed: bits={bits}, n={n_elements}"
        )


# -----------------------------------------------------------------------
# TurboQuantConfig validation
# -----------------------------------------------------------------------


class TestTurboQuantConfig:
    """Test config validation and properties."""

    def test_invalid_bit_width(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
        )

        with pytest.raises(ValueError, match="bit_width"):
            TurboQuantConfig(bit_width=5)

    def test_outlier_fraction_and_channels_conflict(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
        )

        with pytest.raises(ValueError, match="either"):
            TurboQuantConfig(outlier_channels=[0, 1], outlier_fraction=0.1)

    def test_lite_and_qjl_conflict(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            TurboQuantConfig(lite_mode=True, use_qjl=True)

    def test_asymmetric_value_bit_width(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
        )

        cfg = TurboQuantConfig(bit_width=4, value_bit_width=2)
        assert cfg.effective_value_bit_width == 2
        assert cfg.bit_width == 4

    def test_channel_split_2_5(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
        )

        cfg = TurboQuantConfig(bit_width=2.5)
        assert cfg.is_fractional
        (hi_bits, hi_ratio), (lo_bits, lo_ratio) = cfg.channel_split
        assert abs(hi_bits * hi_ratio + lo_bits * lo_ratio - 2.5) < 1e-6


# -----------------------------------------------------------------------
# TurboQuantVLLMConfig integration
# -----------------------------------------------------------------------


class TestVLLMConfig:
    """Test the vLLM QuantizationConfig integration."""

    def test_from_config_defaults(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantVLLMConfig,
        )

        cfg = TurboQuantVLLMConfig.from_config({})
        assert cfg.get_name() == "turboquant"
        assert cfg.tq_config.bit_width == 3
        assert cfg.tq_config.use_qjl is False

    def test_from_config_custom(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantVLLMConfig,
        )

        cfg = TurboQuantVLLMConfig.from_config(
            {
                "bit_width": 4,
                "outlier_fraction": 0.15,
            }
        )
        assert cfg.tq_config.bit_width == 4
        assert cfg.tq_config.outlier_fraction == 0.15

    def test_override_quantization_method_returns_none(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantVLLMConfig,
        )

        assert TurboQuantVLLMConfig.override_quantization_method({}, None) is None

    def test_get_supported_act_dtypes(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantVLLMConfig,
        )

        cfg = TurboQuantVLLMConfig()
        dtypes = cfg.get_supported_act_dtypes()
        assert torch.bfloat16 in dtypes
        assert torch.half in dtypes

    def test_min_capability(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantVLLMConfig,
        )

        assert TurboQuantVLLMConfig.get_min_capability() == 70


# -----------------------------------------------------------------------
# Outlier calibration
# -----------------------------------------------------------------------


class TestOutlierCalibration:
    """Test variance-based outlier channel detection."""

    def test_calibration_detects_high_variance_channels(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        torch.manual_seed(0)
        head_size = 128
        n_outliers = 5
        config = TurboQuantConfig(bit_width=4, outlier_fraction=n_outliers / head_size)
        state = TurboQuantState(config, head_size, layer_idx=0, device=DEVICE)

        # Create data where channels 0-4 have 100x higher variance
        data = torch.randn(500, head_size, device=DEVICE)
        data[:, :n_outliers] *= 100.0

        state.calibrate_outliers(data, n_outliers=n_outliers)

        # The top-5 variance channels should be 0-4
        detected = set(state.outlier_idx.cpu().tolist())
        expected = set(range(n_outliers))
        assert detected == expected, f"Expected outliers {expected}, got {detected}"
        assert state.normal_size == head_size - n_outliers

    def test_calibration_regenerates_rotation(self):
        """After calibration, rotation matrices should match new normal_size."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        head_size = 128
        config = TurboQuantConfig(bit_width=3, outlier_fraction=0.1)
        state = TurboQuantState(config, head_size, layer_idx=0, device=DEVICE)

        data = torch.randn(100, head_size, device=DEVICE)
        data[:, 50:60] *= 50.0  # channels 50-59 are outliers
        state.calibrate_outliers(data, n_outliers=10)

        # Normal size changed → hadamard_d might change
        assert state.normal_size == 118
        assert state.sign_flips.shape[0] == state._hadamard_d


# -----------------------------------------------------------------------
# Full paged-cache encode→decode roundtrip (unfused path)
# -----------------------------------------------------------------------


class TestUnfusedRoundtrip:
    """Test encode→decode through paged cache for 2/3/4-bit unfused paths."""

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    @pytest.mark.parametrize("head_size", [128, 96])
    def test_roundtrip_quality(self, bit_width, head_size):
        """Encode to cache → decode from cache → check cosine similarity."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )
        from vllm.v1.attention.ops.triton_hadamard_turboquant import (
            hadamard_turboquant_decode,
            hadamard_turboquant_encode,
        )

        torch.manual_seed(42)
        num_tokens = 8
        num_kv_heads = 4
        outlier_fraction = 0.15
        n_outliers = max(1, int(head_size * outlier_fraction))
        normal_size = head_size - n_outliers

        config = TurboQuantConfig(
            bit_width=bit_width, outlier_fraction=outlier_fraction
        )
        state = TurboQuantState(config, head_size, layer_idx=0, device=DEVICE)

        x = torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            device=DEVICE,
            dtype=torch.bfloat16,
        )

        # Split channels
        normal_x = x[..., state.normal_idx].contiguous()
        outlier_x = x[..., state.outlier_idx]

        # Encode
        indices, norms = hadamard_turboquant_encode(
            normal_x.float(),
            state.sign_flips,
            state.codebook,
            state.boundaries,
        )

        # Decode
        decoded_normal = hadamard_turboquant_decode(
            indices,
            norms,
            state.sign_flips,
            state.codebook,
            output_dtype=torch.bfloat16,
        )

        # Reassemble
        N = num_tokens * num_kv_heads
        full = torch.empty(N, head_size, dtype=torch.bfloat16, device=DEVICE)
        full[:, state.normal_idx] = decoded_normal.reshape(N, normal_size)
        full[:, state.outlier_idx] = outlier_x.reshape(N, n_outliers).to(torch.bfloat16)

        # Quality check: cosine similarity
        original = x.reshape(N, head_size).float()
        cos_sim = (
            torch.nn.functional.cosine_similarity(original, full.float(), dim=1)
            .mean()
            .item()
        )

        # Higher bits → better quality
        min_cos = {2: 0.85, 3: 0.95, 4: 0.98}
        assert cos_sim > min_cos[bit_width], (
            f"{bit_width}-bit cos_sim={cos_sim:.4f} < {min_cos[bit_width]}"
        )


# -----------------------------------------------------------------------
# Lite mode
# -----------------------------------------------------------------------


class TestLiteMode:
    """Test TQ_LITE mode: no rotation, pure scalar quantization."""

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_lite_roundtrip(self, bit_width):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        torch.manual_seed(7)
        head_size = 128
        config = TurboQuantConfig(bit_width=bit_width, lite_mode=True)
        state = TurboQuantState(config, head_size, layer_idx=0, device=DEVICE)

        assert state.use_hadamard is False
        assert state.sign_flips is None

        x = torch.randn(4, 2, head_size, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))

        assert x_hat.shape == x.shape
        # Lite mode should still reduce MSE (it quantizes)
        mse = (x.float() - x_hat.float()).pow(2).mean().item()
        # Lite mode has worse quality than full TQ but should still work
        assert mse < 5.0, f"Lite mode MSE too high: {mse}"


# -----------------------------------------------------------------------
# Asymmetric K/V bit-widths
# -----------------------------------------------------------------------


class TestAsymmetricBits:
    """Test different bit-widths for K and V."""

    def test_different_k_v_states(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        torch.manual_seed(0)
        head_size = 128
        k_config = TurboQuantConfig(bit_width=4)
        v_config = TurboQuantConfig(bit_width=2)
        k_state = TurboQuantState(k_config, head_size, layer_idx=0, device=DEVICE)
        v_state = TurboQuantState(v_config, head_size, layer_idx=0, device=DEVICE)

        x = torch.randn(4, 2, head_size, device=DEVICE)

        k_hat = k_state.dequantize(k_state.quantize(x))
        v_hat = v_state.dequantize(v_state.quantize(x))

        k_mse = (x.float() - k_hat.float()).pow(2).mean().item()
        v_mse = (x.float() - v_hat.float()).pow(2).mean().item()

        # 4-bit should have lower MSE than 2-bit
        assert k_mse < v_mse, (
            f"4-bit MSE ({k_mse:.4f}) should be < 2-bit MSE ({v_mse:.4f})"
        )


# -----------------------------------------------------------------------
# Backend encode/decode through paged cache (unfused path, all bit-widths)
# -----------------------------------------------------------------------


class TestPagedCacheRoundtrip:
    """Test encode→store→load→decode through the paged uint8 cache.

    This mimics what the attention backend does: write compressed bytes
    into block-paged cache, then read them back and decode.
    """

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_paged_encode_decode(self, bit_width):
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )
        from vllm.v1.attention.ops.triton_hadamard_turboquant import (
            hadamard_turboquant_decode,
            hadamard_turboquant_encode,
        )

        torch.manual_seed(123)
        num_tokens = 4
        num_kv_heads = 4
        head_size = 128
        block_size = 16
        num_blocks = 8
        outlier_fraction = 0.15

        config = TurboQuantConfig(
            bit_width=bit_width, outlier_fraction=outlier_fraction
        )
        state = TurboQuantState(config, head_size, layer_idx=0, device=DEVICE)

        n_outliers = head_size - state.normal_size
        normal_size = state.normal_size
        packed_bytes = math.ceil(normal_size * bit_width / 8)
        outlier_byte_count = n_outliers * 2
        slot_bytes = outlier_byte_count + packed_bytes + 2

        # Create paged cache
        cache = torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            slot_bytes,
            dtype=torch.uint8,
            device=DEVICE,
        )

        x = torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        normal_x = x[..., state.normal_idx].contiguous()
        outlier_x = x[..., state.outlier_idx]

        # Encode
        indices, norms = hadamard_turboquant_encode(
            normal_x.float(),
            state.sign_flips,
            state.codebook,
            state.boundaries,
        )

        N = num_tokens * num_kv_heads
        flat_indices = indices.reshape(-1, normal_size)

        # Pack indices based on bit_width
        if bit_width == 4:
            if normal_size % 2 != 0:
                flat_indices = torch.nn.functional.pad(flat_indices, (0, 1), value=0)
            packed = flat_indices[:, 0::2] | (flat_indices[:, 1::2] << 4)
        elif bit_width == 2:
            if normal_size % 4 != 0:
                pad = 4 - (normal_size % 4)
                flat_indices = torch.nn.functional.pad(flat_indices, (0, pad), value=0)
            packed = (
                flat_indices[:, 0::4]
                | (flat_indices[:, 1::4] << 2)
                | (flat_indices[:, 2::4] << 4)
                | (flat_indices[:, 3::4] << 6)
            )
        elif bit_width == 3:
            from vllm.v1.attention.backends.turboquant_attn import (
                _pack_3bit_vectorized,
            )

            packed = _pack_3bit_vectorized(flat_indices, normal_size, packed_bytes)
        packed = packed[:, :packed_bytes]

        # Build slot data: [outlier_bytes | packed | norm(2B)]
        parts = []
        ob = (
            outlier_x.reshape(N, n_outliers)
            .to(torch.bfloat16)
            .view(torch.uint8)
            .reshape(N, outlier_byte_count)
        )
        parts.append(ob)
        parts.append(packed)
        norm_bytes = norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)
        parts.append(norm_bytes)
        slot_data = torch.cat(parts, dim=-1)

        # Write to paged cache
        slot_mapping = torch.arange(num_tokens, device=DEVICE)
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size
        slot_3d = slot_data.reshape(num_tokens, num_kv_heads, slot_bytes)
        cache[block_indices, block_offsets] = slot_3d

        # --- Read back and decode ---
        # Keep it simple: just read back the tokens we wrote
        for t in range(num_tokens):
            bi = block_indices[t].item()
            bo = block_offsets[t].item()
            for h in range(num_kv_heads):
                readback = cache[bi, bo, h]
                written = slot_3d[t, h]
                assert torch.equal(readback, written), (
                    f"Cache read mismatch at token={t}, head={h}"
                )

        # Decode from cache and check quality
        read_flat = cache[block_indices, block_offsets].reshape(N, slot_bytes)

        # Parse slot
        pos = 0
        read_outliers = (
            read_flat[:, pos : pos + outlier_byte_count]
            .clone()
            .view(torch.bfloat16)
            .reshape(N, n_outliers)
        )
        pos += outlier_byte_count
        read_packed = read_flat[:, pos : pos + packed_bytes]
        pos += packed_bytes
        read_norms = read_flat[:, pos : pos + 2].clone().view(torch.float16).reshape(N)

        # Unpack
        if bit_width == 4:
            low = read_packed & 0x0F
            high = (read_packed >> 4) & 0x0F
            read_indices = torch.stack([low, high], dim=-1).reshape(N, -1)[
                :, :normal_size
            ]
        elif bit_width == 2:
            b0 = read_packed & 0x03
            b1 = (read_packed >> 2) & 0x03
            b2 = (read_packed >> 4) & 0x03
            b3 = (read_packed >> 6) & 0x03
            read_indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(N, -1)[
                :, :normal_size
            ]
        elif bit_width == 3:
            from vllm.v1.attention.backends.turboquant_attn import (
                _unpack_3bit_vectorized,
            )

            read_indices = _unpack_3bit_vectorized(read_packed, normal_size, DEVICE)[
                :N, :normal_size
            ]

        # Decode: codebook lookup → inverse Hadamard
        indices_3d = read_indices.reshape(N, 1, normal_size)
        norms_2d = read_norms.reshape(N, 1)
        decoded_normal = hadamard_turboquant_decode(
            indices_3d,
            norms_2d,
            state.sign_flips,
            state.codebook,
            output_dtype=torch.bfloat16,
        ).reshape(N, normal_size)

        full = torch.empty(N, head_size, dtype=torch.bfloat16, device=DEVICE)
        full[:, state.normal_idx] = decoded_normal
        full[:, state.outlier_idx] = read_outliers

        # Quality check
        original = x.reshape(N, head_size).float()
        cos_sim = (
            torch.nn.functional.cosine_similarity(original, full.float(), dim=1)
            .mean()
            .item()
        )
        min_cos = {2: 0.85, 3: 0.95, 4: 0.98}
        assert cos_sim > min_cos[bit_width], (
            f"Paged {bit_width}-bit roundtrip cos_sim={cos_sim:.4f}"
        )


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_single_token(self):
        """Encode/decode a single token."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        config = TurboQuantConfig(bit_width=4)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.randn(1, 1, 128, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.shape == x.shape

    def test_large_batch(self):
        """Encode/decode a large batch (stress test)."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        config = TurboQuantConfig(bit_width=3)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.randn(256, 32, 128, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.shape == x.shape

    def test_head_size_80(self):
        """Non-standard head size (e.g., some Falcon models)."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        config = TurboQuantConfig(bit_width=4)
        state = TurboQuantState(config, 80, layer_idx=0, device=DEVICE)
        x = torch.randn(4, 4, 80, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.shape == x.shape
        mse = (x.float() - x_hat.float()).pow(2).mean().item()
        assert mse < 1.0

    def test_zero_input(self):
        """All-zero input should produce all-zero output."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        config = TurboQuantConfig(bit_width=4)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.zeros(2, 4, 128, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.abs().max().item() < 1e-3

    def test_constant_input(self):
        """Constant input (same value per vector) should roundtrip."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        config = TurboQuantConfig(bit_width=4)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.ones(2, 4, 128, device=DEVICE) * 0.5
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.shape == x.shape

    def test_determinism_across_calls(self):
        """Same input → same output across two encode/decode cycles."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        config = TurboQuantConfig(bit_width=3)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.randn(4, 4, 128, device=DEVICE)
        x_hat1 = state.dequantize(state.quantize(x))
        x_hat2 = state.dequantize(state.quantize(x))
        assert torch.equal(x_hat1, x_hat2)

    @pytest.mark.parametrize("head_size", [32, 64, 128, 192, 256])
    def test_various_head_sizes(self, head_size):
        """TurboQuant works for a range of head sizes."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        config = TurboQuantConfig(bit_width=4)
        state = TurboQuantState(config, head_size, layer_idx=0, device=DEVICE)
        x = torch.randn(2, 2, head_size, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.shape == x.shape


# -----------------------------------------------------------------------
# Compression ratio verification
# -----------------------------------------------------------------------


class TestCompressionRatio:
    """Verify actual memory savings."""

    @pytest.mark.parametrize(
        "bit_width,expected_min_ratio",
        [(2, 6.0), (3, 4.0), (4, 3.0)],
    )
    def test_compression_ratio(self, bit_width, expected_min_ratio):
        head_size = 128
        outlier_fraction = 0.15
        n_outliers = max(1, int(head_size * outlier_fraction))
        normal_size = head_size - n_outliers

        fp16_bytes = head_size * 2  # bf16
        packed_bytes = math.ceil(normal_size * bit_width / 8)
        outlier_bytes = n_outliers * 2
        norm_bytes = 2
        tq_bytes = packed_bytes + outlier_bytes + norm_bytes

        ratio = fp16_bytes / tq_bytes
        assert ratio >= expected_min_ratio, (
            f"{bit_width}-bit compression ratio {ratio:.1f}x "
            f"< expected {expected_min_ratio}x"
        )


# -----------------------------------------------------------------------
# MSE bounds (paper Theorem 1)
# -----------------------------------------------------------------------


class TestMSEBounds:
    """Verify MSE stays within theoretical bounds."""

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_mse_within_bounds(self, bit_width):
        from vllm.model_executor.layers.quantization.turboquant import (
            EXPECTED_MSE_NORMALIZED,
            TurboQuantConfig,
            TurboQuantState,
        )

        torch.manual_seed(0)
        config = TurboQuantConfig(bit_width=bit_width, use_qjl=False)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)

        x = torch.randn(500, 1, 128, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        x_hat = state.dequantize(state.quantize(x))
        mse = (x.float() - x_hat.float()).pow(2).sum(dim=-1).mean().item()

        bound = EXPECTED_MSE_NORMALIZED[bit_width]
        # Allow 3x slack for finite-sample variance
        assert mse < bound * 3.0, (
            f"{bit_width}-bit MSE={mse:.4f} exceeds 3x bound={bound * 3:.4f}"
        )

    def test_mse_monotonically_decreases_with_bits(self):
        """Higher bit-width → lower MSE."""
        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        torch.manual_seed(0)
        mses = {}
        for bits in [1, 2, 3, 4]:
            config = TurboQuantConfig(bit_width=bits, use_qjl=False)
            state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
            x = torch.randn(200, 1, 128, device=DEVICE)
            x = x / x.norm(dim=-1, keepdim=True)
            x_hat = state.dequantize(state.quantize(x))
            mses[bits] = (x.float() - x_hat.float()).pow(2).sum(dim=-1).mean().item()

        for b in [1, 2, 3]:
            assert mses[b] > mses[b + 1], (
                f"MSE should decrease: {b}-bit={mses[b]:.4f} "
                f"vs {b + 1}-bit={mses[b + 1]:.4f}"
            )
