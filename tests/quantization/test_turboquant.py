# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV-cache quantization.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2

# ============================================================================
# Helpers
# ============================================================================

ALL_PRESETS = list(TQ_PRESETS.keys())


def _assert_strictly_sorted(seq, name="sequence"):
    for i in range(len(seq) - 1):
        assert seq[i] < seq[i + 1], f"{name} not sorted at index {i}"


def _is_power_of_2(n: int) -> bool:
    return n > 0 and next_power_of_2(n) == n


# Expected concrete values for each preset at head_dim=128.
# fmt: off
PRESET_EXPECTED = {
    "turboquant_k8v4": dict(
        key_fp8=True,  key_quant_bits=8,
        key_mse_bits=0, value_quant_bits=4,
        mse_bits=4, n_centroids=16, centroid_bits=4,
        norm_correction=False,
        key_packed_size=128, value_packed_size=68,
        slot_size=196, slot_size_aligned=196,
    ),
    "turboquant_4bit_nc": dict(
        key_fp8=False, key_quant_bits=4,
        key_mse_bits=4, value_quant_bits=4,
        mse_bits=4, n_centroids=16, centroid_bits=4,
        norm_correction=True,
        key_packed_size=66, value_packed_size=68,
        slot_size=134, slot_size_aligned=134,
    ),
    "turboquant_k3v4_nc": dict(
        key_fp8=False, key_quant_bits=3,
        key_mse_bits=3, value_quant_bits=4,
        mse_bits=3, n_centroids=8, centroid_bits=3,
        norm_correction=True,
        key_packed_size=50, value_packed_size=68,
        slot_size=118, slot_size_aligned=118,
    ),
    "turboquant_3bit_nc": dict(
        key_fp8=False, key_quant_bits=3,
        key_mse_bits=3, value_quant_bits=3,
        mse_bits=3, n_centroids=8, centroid_bits=3,
        norm_correction=True,
        key_packed_size=50, value_packed_size=52,
        slot_size=102, slot_size_aligned=102,
    ),
}
# fmt: on


# ============================================================================
# Config tests (CPU-only, no dependencies beyond config.py)
# ============================================================================


class TestTurboQuantConfig:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_preset_parses(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert isinstance(cfg, TurboQuantConfig)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown TurboQuant"):
            TurboQuantConfig.from_cache_dtype("turboquant_invalid", head_dim=128)

    # ---- Per-preset concrete value checks (table-driven) ----

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_mode(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.key_fp8 is exp["key_fp8"]
        assert cfg.key_quant_bits == exp["key_quant_bits"]
        assert cfg.key_mse_bits == exp["key_mse_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_value_mode(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.value_quant_bits == exp["value_quant_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_bits_and_centroids(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.mse_bits == exp["mse_bits"]
        assert cfg.n_centroids == exp["n_centroids"]
        assert cfg.centroid_bits == exp["centroid_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_norm_correction(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.norm_correction is PRESET_EXPECTED[preset]["norm_correction"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_packed_sizes(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.key_packed_size == exp["key_packed_size"]
        assert cfg.value_packed_size == exp["value_packed_size"]
        assert cfg.slot_size == exp["slot_size"]
        assert cfg.slot_size_aligned == exp["slot_size_aligned"]

    # ---- Cross-preset structural invariants ----

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_slot_equals_key_plus_value(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.slot_size == cfg.key_packed_size + cfg.value_packed_size

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_padded_slot_is_even(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.slot_size_aligned >= cfg.slot_size
        assert cfg.slot_size_aligned % 2 == 0, (
            f"slot_size_aligned={cfg.slot_size_aligned} is not even"
        )

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_value_packed_sizes_positive(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.key_packed_size > 0
        assert cfg.value_packed_size > 0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_n_centroids_is_2_to_mse_bits(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.n_centroids == 2**cfg.mse_bits

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_centroid_bits_always_positive(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.centroid_bits > 0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_mse_key_or_fp8_exclusive(self, preset):
        """Each preset is either FP8 keys or MSE keys, never both."""
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        if cfg.key_fp8:
            assert cfg.key_mse_bits == 0
            assert cfg.key_quant_bits == 8
        else:
            assert cfg.key_mse_bits > 0
            assert cfg.key_quant_bits in (3, 4)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
    def test_all_presets_all_head_dims(self, preset, head_dim):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=head_dim)
        assert cfg.head_dim == head_dim
        assert cfg.slot_size == cfg.key_packed_size + cfg.value_packed_size
        assert cfg.slot_size_aligned >= cfg.slot_size
        assert cfg.slot_size_aligned % 2 == 0

    # ---- Boundary skip layers ----

    def test_boundary_skip_layers_basic(self):
        layers = TurboQuantConfig.get_boundary_skip_layers(32)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_layers_zero(self):
        assert TurboQuantConfig.get_boundary_skip_layers(32, 0) == []

    def test_boundary_skip_layers_small_model(self):
        layers = TurboQuantConfig.get_boundary_skip_layers(4)
        assert layers == ["0", "1", "2", "3"]

    def test_boundary_skip_layers_cap_at_half(self):
        layers = TurboQuantConfig.get_boundary_skip_layers(8, 10)
        assert len(layers) == 8


# ============================================================================
# Centroids tests (CPU-only)
# ============================================================================


class TestCentroids:
    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_centroids_shape(self, bits, expected_n):
        c = get_centroids(128, bits)
        assert c.shape == (expected_n,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        _assert_strictly_sorted(get_centroids(128, bits), "centroids")

    def test_centroids_cached(self):
        c1 = get_centroids(128, 3)
        c2 = get_centroids(128, 3)
        assert c1 is c2, "get_centroids should return cached object"

    def test_centroids_different_dims_not_identical(self):
        c64 = get_centroids(64, 3)
        c128 = get_centroids(128, 3)
        assert not torch.equal(c64, c128)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_symmetric_around_zero(self, bits):
        """N(0, 1/d) is symmetric, so centroids should be ~symmetric."""
        c = get_centroids(128, bits)
        assert abs(c.mean().item()) < 0.01, "Centroids not centered near 0"
        assert abs(c[0].item() + c[-1].item()) < 0.01

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_within_4sigma(self, bits):
        """All centroids should be within ~4 sigma of N(0, 1/d)."""
        sigma = math.sqrt(1.0 / 128)
        c = get_centroids(128, bits)
        for i, val in enumerate(c):
            assert abs(val.item()) < 4 * sigma, (
                f"Centroid {i}={val:.6f} outside 4*sigma={4 * sigma:.6f}"
            )


class TestLloydMax:
    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_solve_shapes(self, bits, expected_n):
        centroids, boundaries = solve_lloyd_max(128, bits)
        assert centroids.shape == (expected_n,)
        assert boundaries.shape == (expected_n - 1,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        centroids, _ = solve_lloyd_max(128, bits)
        _assert_strictly_sorted(centroids, "centroids")

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_sorted(self, bits):
        _, boundaries = solve_lloyd_max(128, bits)
        _assert_strictly_sorted(boundaries, "boundaries")

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_between_centroids(self, bits):
        """Each boundary must lie between its adjacent centroids."""
        centroids, boundaries = solve_lloyd_max(128, bits)
        for i in range(len(boundaries)):
            assert centroids[i] < boundaries[i] < centroids[i + 1], (
                f"Boundary {i}={boundaries[i]:.6f} not between "
                f"c[{i}]={centroids[i]:.6f} and c[{i + 1}]={centroids[i + 1]:.6f}"
            )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_are_midpoints(self, bits):
        """Lloyd-Max boundaries are midpoints of adjacent centroids."""
        centroids, boundaries = solve_lloyd_max(128, bits)
        for i in range(len(boundaries)):
            expected = (centroids[i] + centroids[i + 1]) / 2.0
            assert abs(boundaries[i].item() - expected.item()) < 1e-6

    def test_solve_deterministic(self):
        c1, b1 = solve_lloyd_max(128, 3)
        c2, b2 = solve_lloyd_max(128, 3)
        assert torch.equal(c1, c2)
        assert torch.equal(b1, b2)

    def test_solve_dtype_float32(self):
        centroids, boundaries = solve_lloyd_max(128, 3)
        assert centroids.dtype == torch.float32
        assert boundaries.dtype == torch.float32

    @pytest.mark.parametrize("bits", [3, 4])
    def test_centroids_match_scipy_reference(self, bits):
        """Verify _trapz(n=200) centroids match scipy.integrate.quad reference.

        This ensures our scipy-free trapezoid integration doesn't silently
        drift from the published Lloyd-Max quality.
        """
        pytest.importorskip("scipy")
        from scipy.integrate import quad

        d = 128
        sigma2 = 1.0 / d
        sigma = math.sqrt(sigma2)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
                -x * x / (2 * sigma2)
            )

        n_levels = 2**bits
        lo, hi = -3.5 * sigma, 3.5 * sigma
        ref_centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [
                (ref_centroids[i] + ref_centroids[i + 1]) / 2.0
                for i in range(n_levels - 1)
            ]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = quad(lambda x: x * pdf(x), a, b)
                den, _ = quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else ref_centroids[i])
            if (
                max(abs(new_centroids[i] - ref_centroids[i]) for i in range(n_levels))
                < 1e-10
            ):
                break
            ref_centroids = new_centroids

        # Compare our _trapz centroids against scipy reference
        our_centroids, _ = solve_lloyd_max(d, bits)
        ref_t = torch.tensor(ref_centroids, dtype=torch.float32)
        max_err = (our_centroids - ref_t).abs().max().item()
        # _trapz(n=200) has ~O(h^2) error vs adaptive quad; 1e-3 is tight
        # enough to catch regression while allowing trapezoid approximation.
        assert max_err < 1e-3, (
            f"d={d}, bits={bits}: max centroid error vs scipy = {max_err:.2e}"
        )


# ============================================================================
# Rotation matrix tests (GPU required)
# ============================================================================

GPGPU_AVAILABLE = torch.cuda.is_available() or torch.xpu.is_available()
DEVICE_TYPE = current_platform.device_type


def generate_rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Haar-distributed random orthogonal matrix via QR (test/benchmark only)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestRotationMatrix:
    """Tests for the QR-based rotation (standalone benchmarks only)."""

    @pytest.mark.parametrize("dim", [64, 96, 128, 256])
    def test_rotation_matrix_shape_and_orthogonal(self, dim):
        Pi = generate_rotation_matrix(dim, seed=42, device=DEVICE_TYPE)
        assert Pi.shape == (dim, dim)
        eye = Pi @ Pi.T
        assert torch.allclose(eye, torch.eye(dim, device=DEVICE_TYPE), atol=1e-5), (
            f"Pi not orthogonal for dim={dim}"
        )

    def test_rotation_matrix_deterministic(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=42)
        assert torch.equal(Pi1, Pi2)

    def test_rotation_matrix_different_seeds(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=99)
        assert not torch.equal(Pi1, Pi2)

    def test_rotation_matrix_det_is_pm1(self):
        """Orthogonal matrix determinant must be +1 or -1."""
        Pi = generate_rotation_matrix(128, seed=42, device=DEVICE_TYPE)
        det = torch.linalg.det(Pi)
        assert abs(abs(det.item()) - 1.0) < 1e-4


# ============================================================================
# Hadamard rotation tests (serving path: _build_hadamard)
# ============================================================================


def _build_hadamard(d: int, device: str = "cpu") -> torch.Tensor:
    """Reproduce the serving-path Hadamard construction."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device))


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestHadamardRotation:
    """Tests for the Hadamard rotation used in serving."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_hadamard_orthonormal(self, dim):
        """H must be orthonormal: H @ H^T = I."""
        H = _build_hadamard(dim, DEVICE_TYPE)
        eye = H @ H.T
        assert torch.allclose(eye, torch.eye(dim, device=DEVICE_TYPE), atol=1e-5), (
            f"Hadamard not orthonormal for dim={dim}"
        )

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_hadamard_symmetric(self, dim):
        """Sylvester Hadamard must be symmetric: H = H^T."""
        H = _build_hadamard(dim, DEVICE_TYPE)
        assert torch.allclose(H, H.T, atol=1e-6), (
            f"Hadamard not symmetric for dim={dim}"
        )


# ============================================================================
# Store → Decode round-trip test (GPU + Triton required)
# ============================================================================


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestStoreDecodeRoundTrip:
    """End-to-end: store KV into TQ cache, decode, compare vs fp16 ref."""

    @pytest.mark.parametrize(
        "preset",
        ["turboquant_k8v4", "turboquant_4bit_nc"],
    )
    def test_single_token_roundtrip(self, preset):
        """Store 1 token, decode with query=key, check attention output.

        For a single token with query=key, attention output should equal
        the value (softmax over single key = 1.0). Quantization error
        means we check cosine similarity rather than exact equality.
        """
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        D = 128
        Hk = 4  # num_kv_heads
        Hq = 4  # num_q_heads (no GQA for simplicity)
        B = 1  # single token
        block_size = 16
        num_blocks = 1

        device = torch.device(DEVICE_TYPE)

        # Pure Hadamard rotation (symmetric: H = H^T, so Pi = PiT = H)
        H = _build_hadamard(D, DEVICE_TYPE)
        PiT = H
        Pi = H

        # Generate centroids
        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        # Random K, V
        torch.manual_seed(123)
        key = torch.randn(B, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(B, Hk, D, device=device, dtype=torch.float16)

        # Allocate KV cache
        padded_slot = cfg.slot_size_aligned
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            Hk,
            padded_slot,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.tensor([0], device=device, dtype=torch.int32)

        # Store
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            PiT,
            midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

        # Decode: use key as query so attention = softmax([1]) * V = V
        query = key.expand(B, Hq, D).contiguous().to(torch.float16)
        block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([1], device=device, dtype=torch.int32)

        output = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_num_kv_splits=4,
        )

        # With single KV, output should approximate the stored value.
        # Check per-head cosine similarity > threshold.
        out_fp32 = output.float()
        val_fp32 = value.expand(B, Hq, D).float()
        for h in range(Hq):
            cos_sim = torch.nn.functional.cosine_similarity(
                out_fp32[0, h].unsqueeze(0),
                val_fp32[0, h].unsqueeze(0),
            ).item()
            # FP8 keys should be very accurate; MSE keys have more error
            threshold = 0.95 if cfg.key_fp8 else 0.85
            assert cos_sim > threshold, (
                f"Preset {preset} head {h}: cosine_sim={cos_sim:.4f} < {threshold}"
            )


# ============================================================================
# v1 vs v2 decode kernel equivalence (GPU + Triton required)
# ============================================================================


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestDecodeV2Equivalence:
    """Verify the opt-in v2 decode kernel produces the same result as v1.

    v2 is a re-tiled implementation (grouped Q heads, pair LUT, exp2,
    load-halving) reading the same cache layout. For identical inputs
    its output should match v1 up to floating-point rounding noise
    (cosine similarity ~= 1, max abs delta dominated by fp16 tile-order
    differences, never catastrophic disagreement).
    """

    @staticmethod
    def _build_and_store(
        preset: str, Hk: int, D: int, seq_len: int, block_size: int, seed: int
    ):
        """Build inputs and populate KV cache via the production store path."""
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)

        # Pure Hadamard rotation (symmetric, so Pi = PiT).
        H = _build_hadamard(D, DEVICE_TYPE)
        Pi = PiT = H

        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        torch.manual_seed(seed)
        key = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)

        num_blocks = (seq_len + block_size - 1) // block_size + 1
        padded_slot = cfg.slot_size_aligned
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            Hk,
            padded_slot,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)

        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            PiT,
            midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )
        return cfg, Pi, PiT, centroids, kv_cache, num_blocks

    @staticmethod
    def _run_both_kernels(
        cfg, Pi, PiT, centroids, kv_cache, num_blocks, B, Hq, D, seq_len, qseed
    ):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode_v2 import (
            triton_turboquant_decode_attention_v2,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(
                num_blocks,
                device=device,
                dtype=torch.int32,
            )
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

        scale = 1.0 / math.sqrt(D)
        common = dict(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=scale,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
        )
        out_v1 = triton_turboquant_decode_attention(
            **common,
            max_num_kv_splits=8,
        )
        out_v2 = triton_turboquant_decode_attention_v2(
            **common,
            value_packed_size=cfg.value_packed_size,
            max_seq_len=int(seq_lens.max().item()),
        )
        return query, out_v1, out_v2

    @staticmethod
    def _assert_v1_v2_close(tag, out_v1, out_v2, cos_thr=0.999, abs_thr=0.05):
        a = out_v1.float().flatten()
        b = out_v2.float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
        max_abs = (a - b).abs().max().item()
        assert cos_sim > cos_thr, (
            f"{tag} v1/v2 cosine_sim={cos_sim:.6f} below {cos_thr}"
        )
        assert max_abs < abs_thr, f"{tag} v1/v2 max_abs={max_abs:.4e} above {abs_thr}"

    # ------------------------------------------------------------------
    # Tier 1a: kernel-to-kernel equivalence across a wide shape matrix
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (1, 4, 4, 128, 16),  # minimal, GQA=1, short seq
            (4, 8, 2, 128, 64),  # GQA=4
            (4, 16, 2, 128, 1024),  # wide GQA=8, multi-split
            (2, 64, 8, 64, 2048),  # gpt-oss-ish: D=64, GQA=8, 2k ctx
            (2, 32, 4, 128, 4096),  # llama-ish:  D=128, GQA=8, 4k ctx
            (1, 64, 8, 64, 8192),  # long context
        ],
    )
    def test_v1_v2_equivalence(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=Hk,
            D=D,
            seq_len=seq_len,
            block_size=16,
            seed=4242,
        )
        _, out_v1, out_v2 = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
        )
        self._assert_v1_v2_close(
            f"[{preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]",
            out_v1,
            out_v2,
        )

    # ------------------------------------------------------------------
    # Tier 1b: seq_len edge cases (non-power-of-2, boundaries)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", [1, 2, 17, 33, 127, 129, 513, 1025])
    def test_v1_v2_equivalence_seqlen_edges(self, seq_len):
        # 3-bit preset is the most quantization-lossy -> strongest stress.
        preset = "turboquant_3bit_nc"
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=2,
            D=128,
            seq_len=seq_len,
            block_size=16,
            seed=111,
        )
        _, out_v1, out_v2 = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=2,
            Hq=8,
            D=128,
            seq_len=seq_len,
            qseed=222,
        )
        self._assert_v1_v2_close(f"[seq_len={seq_len}]", out_v1, out_v2)

    # ------------------------------------------------------------------
    # Tier 1c: FP32 ground-truth absolute accuracy gate
    # ------------------------------------------------------------------
    # Compute true softmax attention on the RAW un-quantized K/V in fp32.
    # Both kernels read the same quantized cache -> same intrinsic quant
    # error vs. the oracle. Assert v2's error is no worse than v1's.

    @staticmethod
    def _reference_attention_fp32(query, raw_k, raw_v, scale):
        B, Hq, D = query.shape
        S, Hk, _ = raw_k.shape
        group = Hq // Hk
        q = query.float()
        k = raw_k.float().repeat_interleave(group, dim=1)
        v = raw_v.float().repeat_interleave(group, dim=1)
        scores = scale * torch.einsum("bhd,shd->bhs", q, k)
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("bhs,shd->bhd", probs, v)

    @staticmethod
    def _build_and_store_return_raw(preset, Hk, D, seq_len, block_size, seed):
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
        H = _build_hadamard(D, DEVICE_TYPE)
        Pi = PiT = H
        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        torch.manual_seed(seed)
        raw_k = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        raw_v = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)

        num_blocks = (seq_len + block_size - 1) // block_size + 1
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            Hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
        triton_turboquant_store(
            raw_k,
            raw_v,
            kv_cache,
            slot_mapping,
            PiT,
            midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )
        return (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (2, 8, 2, 128, 128),  # moderate context
            (2, 64, 8, 64, 1024),  # D=64 GQA=8 long-ish
            (1, 32, 4, 128, 2048),  # D=128 GQA=8 longer
        ],
    )
    def test_v2_no_worse_than_v1_vs_fp32_reference(self, preset, B, Hq, Hk, D, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset,
                Hk=Hk,
                D=D,
                seq_len=seq_len,
                block_size=16,
                seed=4242,
            )
        )
        query, out_v1, out_v2 = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
        )
        ref = self._reference_attention_fp32(
            query,
            raw_k,
            raw_v,
            scale=1.0 / math.sqrt(D),
        )
        e1 = (out_v1.float() - ref).abs()
        e2 = (out_v2.float() - ref).abs()
        v1_max, v1_mean = e1.max().item(), e1.mean().item()
        v2_max, v2_mean = e2.max().item(), e2.mean().item()

        tag = f"[{preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        # Absolute ceiling sanity: attention outputs on N(0,1) inputs are
        # O(1); per-element max-abs error of 1.5 would already mean total
        # loss of signal. Generous bound to catch catastrophic breakage.
        assert v1_max < 1.5, f"{tag} v1 max_err {v1_max:.3f} suspiciously large"
        assert v2_max < 1.5, f"{tag} v2 max_err {v2_max:.3f} suspiciously large"
        # Primary gate: v2 must not be materially worse than v1.
        assert v2_max <= v1_max * 1.10 + 0.02, (
            f"{tag} v2 max_err={v2_max:.4f} exceeds "
            f"v1 max_err={v1_max:.4f} * 1.10 + 0.02"
        )
        assert v2_mean <= v1_mean * 1.10 + 0.005, (
            f"{tag} v2 mean_err={v2_mean:.4f} exceeds "
            f"v1 mean_err={v1_mean:.4f} * 1.10 + 0.005"
        )

    # ------------------------------------------------------------------
    # Tier 1d: v2 determinism (same inputs -> bitwise identical outputs)
    # ------------------------------------------------------------------
    # Guards against any uninitialized memory / race / nondeterministic
    # reduction in the optimized kernel.

    @pytest.mark.parametrize("preset", ["turboquant_k8v4", "turboquant_4bit_nc"])
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (2, 16, 2, 128, 1024),  # multi-split decode
            (1, 64, 8, 64, 2048),  # D=64 GQA=8
        ],
    )
    def test_v2_deterministic(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=Hk,
            D=D,
            seq_len=seq_len,
            block_size=16,
            seed=4242,
        )
        # Two runs with IDENTICAL inputs. We rebuild query with the same
        # seed to guarantee identical input tensors.
        _, _, out_v2_a = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
        )
        _, _, out_v2_b = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
        )
        tag = f"[{preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        assert torch.equal(out_v2_a, out_v2_b), (
            f"{tag} v2 produced non-bitwise-identical outputs across two "
            f"runs with identical inputs; max diff="
            f"{(out_v2_a.float() - out_v2_b.float()).abs().max().item():.4e}"
        )
