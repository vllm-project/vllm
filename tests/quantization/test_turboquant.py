# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV-cache quantization.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)

# ============================================================================
# Config tests (CPU-only, no dependencies beyond config.py)
# ============================================================================

ALL_PRESETS = list(TQ_PRESETS.keys())


class TestTurboQuantConfig:

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_preset_parses(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert isinstance(cfg, TurboQuantConfig)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown TurboQuant"):
            TurboQuantConfig.from_cache_dtype("tq-invalid", head_dim=128)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_padded_slot_is_power_of_2(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        p = cfg.padded_slot_size
        assert p > 0
        assert (p & (p - 1)) == 0, f"padded_slot_size={p} is not power of 2"

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_slot_size_le_padded(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.slot_size <= cfg.padded_slot_size

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_value_packed_sizes_positive(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.key_packed_size > 0
        assert cfg.value_packed_size > 0

    def test_k8v4_is_fp8_keys(self):
        cfg = TurboQuantConfig.from_cache_dtype("tq-k8v4", head_dim=128)
        assert cfg.key_fp8 is True
        assert cfg.key_packed_size == 128  # 1 byte per element

    def test_t4nc_is_mse_keys(self):
        cfg = TurboQuantConfig.from_cache_dtype("tq-t4nc", head_dim=128)
        assert cfg.key_fp8 is False
        assert cfg.key_mse_bits == 4
        assert cfg.n_centroids == 16
        assert cfg.norm_correction is True

    def test_boundary_skip_layers_basic(self):
        layers = TurboQuantConfig.get_boundary_skip_layers(32, 2)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_layers_zero(self):
        assert TurboQuantConfig.get_boundary_skip_layers(32, 0) == []

    def test_boundary_skip_layers_small_model(self):
        # 4 layers, n=2 → all layers are boundary
        layers = TurboQuantConfig.get_boundary_skip_layers(4, 2)
        assert layers == ["0", "1", "2", "3"]

    def test_boundary_skip_layers_cap_at_half(self):
        # n=10 but only 8 layers → cap at 4
        layers = TurboQuantConfig.get_boundary_skip_layers(8, 10)
        assert len(layers) == 8  # all layers

    @pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
    def test_various_head_dims(self, head_dim):
        cfg = TurboQuantConfig.from_cache_dtype("tq-t4nc", head_dim=head_dim)
        assert cfg.head_dim == head_dim
        assert cfg.padded_slot_size >= cfg.slot_size


# ============================================================================
# Centroids tests (CPU-only, needs scipy)
# ============================================================================

scipy = pytest.importorskip("scipy")

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)


class TestCentroids:

    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_centroids_shape(self, bits, expected_n):
        c = get_centroids(128, bits)
        assert c.shape == (expected_n,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        c = get_centroids(128, bits)
        for i in range(len(c) - 1):
            assert c[i] < c[i + 1], f"Centroids not sorted at index {i}"

    def test_centroids_cached(self):
        c1 = get_centroids(128, 3)
        c2 = get_centroids(128, 3)
        assert c1 is c2, "get_centroids should return cached object"


# ============================================================================
# Rotation matrix + Triton tests (GPU required)
# ============================================================================

CUDA_AVAILABLE = torch.cuda.is_available()

from vllm.model_executor.layers.quantization.turboquant.quantizer import (
    generate_rotation_matrix,
)

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestRotationMatrix:

    def test_rotation_matrix_orthogonal(self):
        Pi = generate_rotation_matrix(128, seed=42, device="cuda")
        eye = Pi @ Pi.T
        assert torch.allclose(eye, torch.eye(128, device="cuda"),
                              atol=1e-5), "Pi is not orthogonal"

    def test_rotation_matrix_deterministic(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=42)
        assert torch.equal(Pi1, Pi2)

    def test_rotation_matrix_different_seeds(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=99)
        assert not torch.equal(Pi1, Pi2)


# ============================================================================
# Triton store/dequant round-trip (GPU required)
# ============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestTritonRoundtrip:

    @staticmethod
    def _run_roundtrip(config_name, head_dim=128):
        """Run store→dequant roundtrip using actual vLLM Triton kernels."""
        # Import here to avoid import errors if triton not available
        from turboquant_standalone.bench_native_tq import (
            make_config,
            vllm_store_and_dequant,
        )

        N, H, D = 256, 2, head_dim
        torch.manual_seed(42)
        keys = torch.randn(N, H, D, device="cuda", dtype=torch.float16)
        values = torch.randn(N, H, D, device="cuda", dtype=torch.float16)

        cfg = make_config(config_name, head_dim=D)
        k_deq, v_deq = vllm_store_and_dequant(keys, values, cfg,
                                                device="cuda")

        # Cosine similarity
        k_cos = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, D).float(),
            k_deq.reshape(-1, D).float(), dim=1).mean()
        v_cos = torch.nn.functional.cosine_similarity(
            values.reshape(-1, D).float(),
            v_deq.reshape(-1, D).float(), dim=1).mean()

        return k_cos.item(), v_cos.item()

    def test_roundtrip_k8v4(self):
        k_cos, v_cos = self._run_roundtrip("k8v4")
        assert k_cos > 0.99, f"K8V4 key cosine {k_cos:.5f} < 0.99"
        assert v_cos > 0.95, f"K8V4 value cosine {v_cos:.5f} < 0.95"

    def test_roundtrip_t4nc(self):
        k_cos, v_cos = self._run_roundtrip("t4nc")
        assert k_cos > 0.98, f"T4NC key cosine {k_cos:.5f} < 0.98"
        assert v_cos > 0.95, f"T4NC value cosine {v_cos:.5f} < 0.95"

    def test_roundtrip_k3v4nc(self):
        k_cos, v_cos = self._run_roundtrip("k3v4nc")
        assert k_cos > 0.95, f"K3V4NC key cosine {k_cos:.5f} < 0.95"
        assert v_cos > 0.95, f"K3V4NC value cosine {v_cos:.5f} < 0.95"

    def test_roundtrip_t3nc(self):
        k_cos, v_cos = self._run_roundtrip("t3nc")
        assert k_cos > 0.95, f"T3NC key cosine {k_cos:.5f} < 0.95"
        assert v_cos > 0.90, f"T3NC value cosine {v_cos:.5f} < 0.90"
