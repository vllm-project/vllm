# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for TurboQuant KV cache quantization (PolarQuant + QJL).

Tests cover:
  1. Numerical round-trip accuracy (encode -> decode)
  2. QJL unbiasedness (mean error converges to zero)
  3. Deterministic reconstruction (same seed = identical output)
  4. TurboQuantConfig utilities
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant import (
    TurboQuantConfig,
    is_turboquant_kv_cache,
)


# ============================================================================
# Unit tests for TurboQuantConfig (no GPU required)
# ============================================================================


class TestTurboQuantConfig:
    def test_from_kv_cache_dtype_pq4(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("pq4")
        assert cfg.angle_bits == 4
        assert cfg.qjl_residual is False
        assert cfg.qjl_projection_dim is None

    def test_from_kv_cache_dtype_tq3(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("tq3")
        assert cfg.angle_bits == 3
        assert cfg.qjl_residual is True

    def test_from_kv_cache_dtype_tq2(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("tq2")
        assert cfg.angle_bits == 2
        assert cfg.qjl_residual is True

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown TurboQuant dtype"):
            TurboQuantConfig.from_kv_cache_dtype("fp8")

    def test_effective_bits_pq4(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("pq4")
        # For head_size=128: (127*4 + 16) / 128 = 4.09375
        bits = cfg.effective_bits_per_element(128)
        assert 4.0 < bits < 4.2

    def test_effective_bits_tq3(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("tq3")
        # For head_size=128: (127*3 + 16 + 128) / 128 = 4.1015625
        bits = cfg.effective_bits_per_element(128)
        assert 4.0 < bits < 4.2

    def test_effective_bits_tq2(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("tq2")
        # For head_size=128: (127*2 + 16 + 128) / 128 = 3.109375
        bits = cfg.effective_bits_per_element(128)
        assert 3.0 < bits < 3.2

    def test_bytes_per_token_per_head(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("tq3")
        # For head_size=128: angle_bytes=48 (127*3=381 bits -> 48 bytes),
        # radius=2, qjl=16 (128/8) -> total=66
        bpt = cfg.bytes_per_token_per_head(128)
        assert bpt == 48 + 2 + 16  # 66 bytes

    def test_block_bytes(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("tq3")
        # 8 kv heads, head_size=128, block_size=16
        bb = cfg.block_bytes(num_kv_heads=8, head_size=128, block_size=16)
        assert bb == 8 * 16 * 66  # 8448 bytes

    def test_derive_layer_seed_deterministic(self):
        cfg = TurboQuantConfig.from_kv_cache_dtype("tq3", rotation_seed=42)
        s1 = cfg.derive_layer_seed(0)
        s2 = cfg.derive_layer_seed(0)
        assert s1 == s2
        # Different layers get different seeds
        s3 = cfg.derive_layer_seed(1)
        assert s1 != s3

    def test_is_turboquant_kv_cache(self):
        assert is_turboquant_kv_cache("tq2") is True
        assert is_turboquant_kv_cache("tq3") is True
        assert is_turboquant_kv_cache("pq4") is True
        assert is_turboquant_kv_cache("fp8") is False
        assert is_turboquant_kv_cache("auto") is False
        assert is_turboquant_kv_cache("float16") is False


# ============================================================================
# CUDA kernel tests (require GPU)
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTurboQuantKernels:
    """Tests for the CUDA encode/decode kernels.

    These tests use the standalone turboquant_encode/decode ops for
    round-trip testing. They require the vLLM CUDA extensions to be compiled.
    """

    def _try_import_ops(self):
        """Try to import vLLM custom ops. Skip if not compiled."""
        try:
            from vllm import _custom_ops as ops

            # Check if the turboquant ops are available
            if not hasattr(torch.ops, "_C_cache_ops"):
                pytest.skip("vLLM CUDA extensions not compiled")
            if not hasattr(torch.ops._C_cache_ops, "turboquant_encode"):
                pytest.skip("TurboQuant CUDA kernels not compiled")
            return ops
        except (ImportError, AttributeError):
            pytest.skip("vLLM CUDA extensions not available")

    @pytest.mark.parametrize("tq_type", ["pq4", "tq3", "tq2"])
    @pytest.mark.parametrize("head_size", [64, 128])
    def test_roundtrip_accuracy(self, tq_type, head_size):
        """Encode then decode should produce values close to the original."""
        ops = self._try_import_ops()

        num_tokens = 4
        num_kv_heads = 8
        layer_seed = 42
        qjl_proj_dim = head_size

        # Random KV data
        kv_data = torch.randn(
            num_tokens, num_kv_heads, head_size,
            device="cuda", dtype=torch.float32,
        )

        # Allocate output buffers
        num_angles = head_size - 1
        angle_bits_map = {"pq4": 4, "tq3": 3, "tq2": 2}
        bits = angle_bits_map[tq_type]
        angle_bytes = (num_angles * bits + 7) // 8
        has_qjl = tq_type.startswith("tq")
        qjl_bytes = (qjl_proj_dim + 7) // 8 if has_qjl else 0

        angles = torch.zeros(
            num_tokens * num_kv_heads, angle_bytes,
            device="cuda", dtype=torch.uint8,
        )
        radii = torch.zeros(
            num_tokens * num_kv_heads,
            device="cuda", dtype=torch.float16,
        )
        qjl_out = torch.zeros(
            num_tokens * num_kv_heads, max(qjl_bytes, 1),
            device="cuda", dtype=torch.uint8,
        )
        kv_out = torch.zeros_like(kv_data)

        # Encode
        ops.turboquant_encode(
            kv_data, angles, radii, qjl_out,
            num_kv_heads, head_size, tq_type, layer_seed, qjl_proj_dim,
        )

        # Decode
        ops.turboquant_decode(
            angles, radii, qjl_out, kv_out,
            num_kv_heads, head_size, tq_type, layer_seed, qjl_proj_dim,
        )

        # Check reconstruction error
        max_abs_error = (kv_data - kv_out).abs().max().item()
        mean_abs_error = (kv_data - kv_out).abs().mean().item()

        # Error bounds depend on quantization level
        if tq_type == "pq4":
            assert max_abs_error < 0.5, (
                f"pq4 max abs error {max_abs_error} too large"
            )
        elif tq_type == "tq3":
            assert max_abs_error < 1.0, (
                f"tq3 max abs error {max_abs_error} too large"
            )
        elif tq_type == "tq2":
            assert max_abs_error < 2.0, (
                f"tq2 max abs error {max_abs_error} too large"
            )

        # Mean error should be reasonable
        assert mean_abs_error < 0.5, (
            f"{tq_type} mean abs error {mean_abs_error} too large"
        )

    @pytest.mark.parametrize("tq_type", ["tq3", "tq2"])
    def test_qjl_unbiasedness(self, tq_type):
        """QJL residual should produce an unbiased correction on average."""
        ops = self._try_import_ops()

        num_tokens = 100
        num_kv_heads = 4
        head_size = 64
        layer_seed = 42
        qjl_proj_dim = head_size

        # Generate random data
        kv_data = torch.randn(
            num_tokens, num_kv_heads, head_size,
            device="cuda", dtype=torch.float32,
        )

        # Encode + decode
        angle_bits_map = {"tq3": 3, "tq2": 2}
        bits = angle_bits_map[tq_type]
        num_angles = head_size - 1
        angle_bytes = (num_angles * bits + 7) // 8
        qjl_bytes = (qjl_proj_dim + 7) // 8

        angles = torch.zeros(
            num_tokens * num_kv_heads, angle_bytes,
            device="cuda", dtype=torch.uint8,
        )
        radii = torch.zeros(
            num_tokens * num_kv_heads,
            device="cuda", dtype=torch.float16,
        )
        qjl_out = torch.zeros(
            num_tokens * num_kv_heads, qjl_bytes,
            device="cuda", dtype=torch.uint8,
        )
        kv_out = torch.zeros_like(kv_data)

        ops.turboquant_encode(
            kv_data, angles, radii, qjl_out,
            num_kv_heads, head_size, tq_type, layer_seed, qjl_proj_dim,
        )
        ops.turboquant_decode(
            angles, radii, qjl_out, kv_out,
            num_kv_heads, head_size, tq_type, layer_seed, qjl_proj_dim,
        )

        # Mean error across all tokens should converge toward zero
        mean_error = (kv_data - kv_out).mean().item()
        assert abs(mean_error) < 0.1, (
            f"QJL mean error {mean_error} suggests bias (should be ~0)"
        )

    @pytest.mark.parametrize("tq_type", ["pq4", "tq3", "tq2"])
    def test_deterministic_reconstruction(self, tq_type):
        """Same input + same seed should produce identical output."""
        ops = self._try_import_ops()

        num_tokens = 4
        num_kv_heads = 4
        head_size = 64
        layer_seed = 123
        qjl_proj_dim = head_size

        kv_data = torch.randn(
            num_tokens, num_kv_heads, head_size,
            device="cuda", dtype=torch.float32,
        )

        angle_bits_map = {"pq4": 4, "tq3": 3, "tq2": 2}
        bits = angle_bits_map[tq_type]
        num_angles = head_size - 1
        angle_bytes = (num_angles * bits + 7) // 8
        has_qjl = tq_type.startswith("tq")
        qjl_bytes = (qjl_proj_dim + 7) // 8 if has_qjl else 0

        def encode_decode():
            angles = torch.zeros(
                num_tokens * num_kv_heads, angle_bytes,
                device="cuda", dtype=torch.uint8,
            )
            radii = torch.zeros(
                num_tokens * num_kv_heads,
                device="cuda", dtype=torch.float16,
            )
            qjl_out = torch.zeros(
                num_tokens * num_kv_heads, max(qjl_bytes, 1),
                device="cuda", dtype=torch.uint8,
            )
            kv_out = torch.zeros_like(kv_data)

            ops.turboquant_encode(
                kv_data, angles, radii, qjl_out,
                num_kv_heads, head_size, tq_type, layer_seed, qjl_proj_dim,
            )
            ops.turboquant_decode(
                angles, radii, qjl_out, kv_out,
                num_kv_heads, head_size, tq_type, layer_seed, qjl_proj_dim,
            )
            return kv_out

        out1 = encode_decode()
        out2 = encode_decode()

        assert torch.equal(out1, out2), (
            f"Deterministic reconstruction failed for {tq_type}: "
            f"max diff = {(out1 - out2).abs().max().item()}"
        )
