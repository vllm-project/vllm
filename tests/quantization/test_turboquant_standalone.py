# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone TurboQuant tests -- core math, codebook, rotation, packing.

Uses pytest so these integrate with CI. Can also run standalone:
    python -m pytest tests/quantization/test_turboquant_standalone.py -v
"""

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant import (
    EXPECTED_MSE_NORMALIZED,
    TurboQuantConfig,
    TurboQuantState,
    _get_codebook,
    random_rotate,
    random_rotate_inverse,
    scalar_dequantize,
    scalar_quantize,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------
# Codebook
# -----------------------------------------------------------------------


class TestCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_symmetric(self, bits):
        cb = _get_codebook(bits, 128, DEVICE)
        assert torch.allclose(cb, -cb.flip(0), atol=1e-5)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_sorted(self, bits):
        cb = _get_codebook(bits, 128, DEVICE)
        assert (cb[1:] > cb[:-1]).all()

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_num_centroids(self, bits):
        cb = _get_codebook(bits, 128, DEVICE)
        assert cb.shape[0] == 2**bits


# -----------------------------------------------------------------------
# Rotation (sign-flip)
# -----------------------------------------------------------------------


class TestRotation:
    def test_norm_preserving(self):
        d = 128
        sf = torch.randint(0, 2, (d,), device=DEVICE).float() * 2 - 1
        x = torch.randn(10, d, device=DEVICE)
        y = random_rotate(x, sf)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)

    def test_invertible(self):
        d = 128
        sf = torch.randint(0, 2, (d,), device=DEVICE).float() * 2 - 1
        x = torch.randn(10, d, device=DEVICE)
        y = random_rotate(x, sf)
        x_rec = random_rotate_inverse(y, sf)
        assert torch.allclose(x, x_rec, atol=1e-4), (
            f"max diff={(x - x_rec).abs().max():.6f}"
        )


# -----------------------------------------------------------------------
# Scalar quantization
# -----------------------------------------------------------------------


class TestScalarQuantize:
    def test_centroid_roundtrip(self):
        cb = _get_codebook(3, 128, DEVICE)
        indices = scalar_quantize(cb, cb)
        recovered = scalar_dequantize(indices, cb)
        assert torch.allclose(cb, recovered)

    def test_index_range(self):
        cb = _get_codebook(3, 128, DEVICE)
        x = torch.randn(100, device=DEVICE) / math.sqrt(128)
        idx = scalar_quantize(x, cb)
        assert int(idx.min()) >= 0 and int(idx.max()) <= 7


# -----------------------------------------------------------------------
# Roundtrip MSE (paper Theorem 1)
# -----------------------------------------------------------------------


class TestRoundtripMSE:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_within_bound(self, bits):
        torch.manual_seed(0)
        n, d = 500, 128
        config = TurboQuantConfig(bit_width=bits, use_qjl=False)
        state = TurboQuantState(config, d, layer_idx=0, device=DEVICE)

        x = torch.randn(n, 1, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        x_hat = state.dequantize(state.quantize(x))
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        bound = EXPECTED_MSE_NORMALIZED[bits]
        assert mse < bound * 3.0, f"ratio={mse / bound:.2f}x"


# -----------------------------------------------------------------------
# QJL unbiasedness (paper Theorem 2)
# -----------------------------------------------------------------------


class TestQJL:
    def test_unbiased(self):
        torch.manual_seed(42)
        d, n = 128, 300
        x = torch.randn(n, 1, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n, 1, d, device=DEVICE)

        state = TurboQuantState(
            TurboQuantConfig(bit_width=2, use_qjl=True),
            d,
            layer_idx=0,
            device=DEVICE,
        )
        x_hat = state.dequantize(state.quantize(x))
        ip_true = (y * x).sum(dim=-1)
        ip_est = (y * x_hat).sum(dim=-1)
        bias = (ip_est - ip_true).mean().abs().item()
        assert bias < 0.05, f"QJL bias={bias:.4f}"


# -----------------------------------------------------------------------
# Non-standard head sizes
# -----------------------------------------------------------------------


class TestNonStandardHeadSize:
    @pytest.mark.parametrize("hs", [32, 64, 80, 96, 128, 192, 256])
    def test_shape_preserved(self, hs):
        config = TurboQuantConfig(bit_width=2, use_qjl=False)
        state = TurboQuantState(config, hs, layer_idx=0, device=DEVICE)
        x = torch.randn(2, 4, hs, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.shape == x.shape


# -----------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_indices(self):
        config = TurboQuantConfig(bit_width=3)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.randn(2, 4, 128, device=DEVICE)
        q1 = state.quantize(x)
        q2 = state.quantize(x)
        assert torch.equal(q1["indices"], q2["indices"])

    def test_different_layers_different_rotation(self):
        """Different layer_idx → different sign flips."""
        config = TurboQuantConfig(bit_width=3)
        s0 = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        s1 = TurboQuantState(config, 128, layer_idx=1, device=DEVICE)
        assert not torch.equal(s0.sign_flips, s1.sign_flips)


# -----------------------------------------------------------------------
# Compression ratio
# -----------------------------------------------------------------------


class TestCompressionRatio:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_ratio(self, bits):
        d = 128
        fp16_bytes = d * 2
        tq_bytes = math.ceil(d * bits / 8) + 2
        ratio = fp16_bytes / tq_bytes
        assert ratio > 1.5


# -----------------------------------------------------------------------
# Fractional bit-widths
# -----------------------------------------------------------------------


class TestFractionalBitWidth:
    @pytest.mark.parametrize("bits", [2.5, 3.5])
    def test_shape_preserved(self, bits):
        torch.manual_seed(7)
        config = TurboQuantConfig(bit_width=bits, use_qjl=False)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.randn(200, 1, 128, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        x_hat = state.dequantize(state.quantize(x))
        assert x_hat.shape == x.shape

    @pytest.mark.parametrize("bits", [2.5, 3.5])
    def test_mse_in_range(self, bits):
        torch.manual_seed(7)
        config = TurboQuantConfig(bit_width=bits, use_qjl=False)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.randn(200, 1, 128, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        x_hat = state.dequantize(state.quantize(x))
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        lo_bits = int(bits)
        assert mse < EXPECTED_MSE_NORMALIZED[lo_bits] * 3.0

    @pytest.mark.parametrize("bits", [2.5, 3.5])
    def test_with_qjl(self, bits):
        torch.manual_seed(99)
        config = TurboQuantConfig(bit_width=bits, use_qjl=True)
        state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
        x = torch.randn(200, 1, 128, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)
        compressed = state.quantize(x)
        assert "qjl_signs" in compressed
        assert "qjl_norms" in compressed
        x_hat = state.dequantize(compressed)
        assert x_hat.shape == x.shape


# -----------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------


class TestConfigValidation:
    def test_1bit_qjl_rejected(self):
        with pytest.raises(ValueError):
            TurboQuantConfig(bit_width=1, use_qjl=True)

    @pytest.mark.parametrize("bits", [2.5, 3.5])
    def test_channel_split_weighted_average(self, bits):
        config = TurboQuantConfig(bit_width=bits)
        split = config.channel_split
        (hi_bits, hi_ratio), (lo_bits, lo_ratio) = split
        actual = hi_bits * hi_ratio + lo_bits * lo_ratio
        assert abs(actual - bits) < 1e-6

    def test_outlier_fraction_out_of_range(self):
        with pytest.raises(ValueError):
            TurboQuantConfig(outlier_fraction=1.0)
        with pytest.raises(ValueError):
            TurboQuantConfig(outlier_fraction=-0.1)


# -----------------------------------------------------------------------
# Triton pre-dequant integration
# -----------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonPreDequant:
    def test_basic(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            turboquant_pre_dequant,
        )

        torch.manual_seed(456)
        d = 128
        device = torch.device("cuda")
        config = TurboQuantConfig(bit_width=3, use_qjl=False)
        k_state = TurboQuantState(config, d, layer_idx=0, device=device)
        v_state = TurboQuantState(config, d, layer_idx=10000, device=device)

        key = torch.randn(5, 8, d, device=device, dtype=torch.bfloat16)
        value = torch.randn(5, 8, d, device=device, dtype=torch.bfloat16)

        k_out, v_out = turboquant_pre_dequant(key, value, k_state, v_state)
        assert k_out.dtype == torch.bfloat16
        assert k_out.shape == key.shape
        k_mse = (key.float() - k_out.float()).pow(2).mean().item()
        assert k_mse < 1.0, f"key MSE={k_mse:.4f}"

    def test_with_outliers(self):
        from vllm.model_executor.layers.quantization.turboquant import (
            turboquant_pre_dequant,
        )

        torch.manual_seed(789)
        d = 128
        device = torch.device("cuda")
        config = TurboQuantConfig(bit_width=4, use_qjl=False, outlier_fraction=0.15)
        k_state = TurboQuantState(config, d, layer_idx=0, device=device)
        v_state = TurboQuantState(config, d, layer_idx=1, device=device)

        key = torch.randn(5, 8, d, device=device, dtype=torch.bfloat16)
        value = torch.randn(5, 8, d, device=device, dtype=torch.bfloat16)

        k_out, v_out = turboquant_pre_dequant(key, value, k_state, v_state)
        assert k_out.shape == key.shape

        # Outlier channels should be preserved exactly
        k_outliers = key[..., k_state.outlier_idx]
        k_out_outliers = k_out[..., k_state.outlier_idx]
        assert torch.allclose(k_outliers, k_out_outliers, atol=1e-6), (
            "Outlier channels not preserved in pre-dequant"
        )
