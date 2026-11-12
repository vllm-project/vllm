# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV-cache quantization.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import math
from dataclasses import replace
from types import SimpleNamespace

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

    @staticmethod
    def _dense_model_config(num_layers):
        from types import SimpleNamespace

        return SimpleNamespace(
            is_hybrid=False,
            hf_text_config=SimpleNamespace(num_hidden_layers=num_layers),
        )

    def test_boundary_skip_layers_basic(self):
        mc = self._dense_model_config(32)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_layers_zero(self):
        mc = self._dense_model_config(32)
        assert TurboQuantConfig.get_boundary_skip_layers(mc, 0) == []

    def test_boundary_skip_layers_small_model(self):
        mc = self._dense_model_config(4)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc)
        assert layers == ["0", "1", "2", "3"]

    def test_boundary_skip_layers_cap_at_half(self):
        mc = self._dense_model_config(8)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc, 10)
        assert len(layers) == 8


class TestHybridAttentionIndices:
    """Regression tests for boundary protection on hybrid models.

    Hybrid models (attention + Mamba / linear-attention) identify KV-carrying
    layers via layer_types / layers_block_type / attn_type_list. The helper
    must return the *global* layer indices of the full-attention layers so
    that kv_cache_dtype_skip_layers matches what extract_layer_index(prefix)
    reports on the Attention layers at runtime.
    """

    @staticmethod
    def _fake_model_config(text_cfg=None, hf_cfg=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            hf_text_config=text_cfg if text_cfg is not None else SimpleNamespace(),
            hf_config=hf_cfg if hf_cfg is not None else SimpleNamespace(),
        )

    def test_layer_types_full_attention(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        cfg = type("C", (), {})()
        cfg.layer_types = [
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
            "full_attention",
        ]
        mc = self._fake_model_config(text_cfg=cfg)
        assert _get_full_attention_layer_indices(mc) == [2, 4, 5]

    def test_layers_block_type_jamba(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        cfg = type("C", (), {})()
        cfg.layers_block_type = ["mamba", "attention", "mamba", "attention"]
        mc = self._fake_model_config(text_cfg=cfg)
        assert _get_full_attention_layer_indices(mc) == [1, 3]

    def test_attn_type_list_minimax(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        hf = type("C", (), {})()
        hf.attn_type_list = [0, 1, 0, 1, 1]
        mc = self._fake_model_config(hf_cfg=hf)
        assert _get_full_attention_layer_indices(mc) == [1, 3, 4]

    def test_no_hybrid_hints_returns_empty(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        mc = self._fake_model_config()
        assert _get_full_attention_layer_indices(mc) == []


class TestTurboQuantKVCacheSpec:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_kv_cache_spec_sets_kv_quant_mode(self, preset):
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        layer = SimpleNamespace(
            attn_type="decoder",
            kv_cache_dtype=preset,
            kv_cache_torch_dtype=torch.uint8,
            head_size=128,
            head_size_v=128,
            num_kv_heads=4,
            sliding_window=None,
            get_attn_backend=lambda: TurboQuantAttentionBackend,
        )
        vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))

        # The layer builds an unpacked spec; the worker's spec-collection
        # loop applies TQ slot packing via the backend's customize_spec hook.
        spec = Attention.get_kv_cache_spec(layer, vllm_config)
        assert isinstance(spec, FullAttentionSpec)
        assert spec.kv_quant_mode.is_turboquant
        assert spec.state_content_bytes is None

        spec = TurboQuantAttentionBackend.customize_spec(spec)
        expected_slot = TurboQuantConfig.from_cache_dtype(preset, 128).slot_size_aligned
        assert spec.state_content_bytes == expected_slot


class TestTurboQuantWorkspaceReservation:
    @staticmethod
    def _fake_vllm_config(
        *,
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 4096,
        enable_chunked_prefill: bool = True,
        max_model_len: int = 8192,
        dtype: torch.dtype = torch.float16,
        max_num_kv_splits: int = 4,
    ):
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=enable_chunked_prefill,
            ),
            model_config=SimpleNamespace(
                max_model_len=max_model_len,
                dtype=dtype,
                get_num_attention_heads=lambda parallel_config: 8,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                decode_context_parallel_size=1,
            ),
            attention_config=SimpleNamespace(
                tq_max_kv_splits_for_cuda_graph=max_num_kv_splits
            ),
        )

    @staticmethod
    def _fake_kv_cache_spec():
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return FullAttentionSpec(
            block_size=32,
            num_kv_heads=4,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
            state_content_bytes=102,
        )

    def test_metadata_builder_reserves_decode_and_continuation_prefill_workspace(
        self, monkeypatch
    ):
        from vllm.v1.attention.backends import turboquant_attn

        calls = []

        class FakeWorkspaceManager:
            def get_simultaneous(self, *shapes_and_dtypes):
                calls.append(shapes_and_dtypes)

        monkeypatch.setattr(
            turboquant_attn,
            "current_workspace_manager",
            lambda: FakeWorkspaceManager(),
        )
        monkeypatch.setattr(
            turboquant_attn,
            "is_workspace_manager_initialized",
            lambda: True,
        )

        turboquant_attn.TurboQuantMetadataBuilder(
            kv_cache_spec=self._fake_kv_cache_spec(),
            layer_names=["layers.0.self_attn.attn"],
            vllm_config=self._fake_vllm_config(),
            device=torch.device("cuda"),
        )

        assert calls == [
            (
                ((16, 8, 4, 129), torch.float32),
                ((16, 8, 128), torch.float16),
                ((16, 8), torch.float32),
            ),
            (
                ((1, 4, 8192, 128), torch.float16),
                ((1, 4, 8192, 128), torch.float16),
            ),
        ]

    def test_metadata_builder_skips_continuation_prefill_when_disabled(
        self, monkeypatch
    ):
        from vllm.v1.attention.backends import turboquant_attn

        calls = []

        class FakeWorkspaceManager:
            def get_simultaneous(self, *shapes_and_dtypes):
                calls.append(shapes_and_dtypes)

        monkeypatch.setattr(
            turboquant_attn,
            "current_workspace_manager",
            lambda: FakeWorkspaceManager(),
        )
        monkeypatch.setattr(
            turboquant_attn,
            "is_workspace_manager_initialized",
            lambda: True,
        )

        turboquant_attn.TurboQuantMetadataBuilder(
            kv_cache_spec=self._fake_kv_cache_spec(),
            layer_names=["layers.0.self_attn.attn"],
            vllm_config=self._fake_vllm_config(enable_chunked_prefill=False),
            device=torch.device("cuda"),
        )

        assert calls == [
            (
                ((16, 8, 4, 129), torch.float32),
                ((16, 8, 128), torch.float16),
                ((16, 8), torch.float32),
            )
        ]


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
    # torch.linalg.qr on CPU requires LAPACK, which some torch wheels
    # (ROCm) ship without. Run QR on accelerator instead
    qr_device = "cuda" if torch.cuda.is_available() else "cpu"
    Q, R = torch.linalg.qr(G.to(qr_device))
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


def test_fp8_draft_fits_turboquant_hybrid_pages(vllm_runner, monkeypatch):
    """An FP8 drafter must not inflate aligned target pages or break decoding."""
    import json

    from vllm import SamplingParams

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_TRITON_USE_TD", "0")
    # The local test sends a cache-inspection callable to its own worker.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    def cache_layout(worker):
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            KVQuantMode,
            UniformTypeKVCacheSpecs,
        )

        layout = []
        for group in worker.model_runner.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs = (
                spec.kv_cache_specs.values()
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else [spec]
            )
            for layer_spec in specs:
                layout.append(
                    {
                        # Merged specs do not retain is_draft. In this model
                        # pair only the drafter uses per-tensor FP8 storage.
                        "draft": isinstance(layer_spec, AttentionSpec)
                        and layer_spec.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR,
                        "attention": isinstance(layer_spec, AttentionSpec),
                        "block_size": layer_spec.block_size,
                        "page_size": layer_spec.page_size_bytes,
                    }
                )
        return layout

    with vllm_runner(
        "Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        trust_remote_code=False,
        dtype="bfloat16",
        block_size=None,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        kv_cache_dtype="turboquant_4bit_nc",
        speculative_config={
            "method": "dflash",
            "model": "vllm-project/Qwen3.5-4B-DFlash2",
            "revision": "50d7919bb821601f23301bc72d2c34f4046a055d",
            "num_speculative_tokens": 15,
            "attention_backend": "TRITON_ATTN",
            "kv_cache_dtype": "fp8",
        },
    ) as runner:
        llm = runner.llm
        layout = llm.collective_rpc(cache_layout)[0]
        target = [s for s in layout if s["attention"] and not s["draft"]]
        draft = [s for s in layout if s["draft"]]
        assert target and draft, layout
        assert len({s["page_size"] for s in layout}) == 1, layout
        target_block = target[0]["block_size"]
        for spec in draft:
            assert spec["block_size"] < target_block, layout
            assert target_block % spec["block_size"] == 0, layout

        tokenizer = llm.get_tokenizer()
        sentence = "The red fox jumps over the blue fence. "
        filler = tokenizer.encode(sentence * 500, add_special_tokens=False)
        tail = tokenizer.encode(
            "\nRepeat that sentence twelve more times:\n",
            add_special_tokens=False,
        )
        block = draft[0]["block_size"]
        lengths = [block - 1, block + 1]
        prompts = [
            {"prompt_token_ids": filler[: n - len(tail)] + tail} for n in lengths
        ]
        assert [len(p["prompt_token_ids"]) for p in prompts] == lengths
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64))
        assert all("red fox" in output.outputs[0].text for output in outputs)
        accepted = next(
            m.value
            for m in llm.get_metrics()
            if m.name == "vllm:spec_decode_num_accepted_tokens"
        )
        assert accepted > 0, "The FP8 drafter must contribute accepted tokens"
        print(
            "PAGE_FIT_RESULT",
            json.dumps(
                {
                    "layout": layout,
                    "prompt_lengths": lengths,
                    "accepted_draft_tokens": accepted,
                }
            ),
        )


def test_turboquant_speculative_graph_matches_eager(vllm_runner):
    """Graph verification must retain history after chunked prefill."""
    from tests.models.utils import check_logprobs_close
    from vllm import SamplingParams

    sentence = "The red fox jumps over the blue fence. "
    prompts = [
        [
            {
                "role": "user",
                "content": sentence * 60
                + "Repeat the same sentence twelve more times, without commentary.",
            }
        ],
        [
            {
                "role": "user",
                "content": "A store sells 12 boxes with 8 pens each. "
                "It sells 31 pens. How many pens remain? Explain the calculation.",
            }
        ],
    ]
    params = SamplingParams(temperature=0, max_tokens=128, logprobs=5)
    results = []
    for eager in (True, False):
        with vllm_runner(
            "Qwen/Qwen3.5-4B",
            trust_remote_code=False,
            dtype="bfloat16",
            block_size=None,
            max_model_len=2048,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=eager,
            gpu_memory_utilization=0.8,
            disable_log_stats=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            kv_cache_dtype="turboquant_4bit_nc",
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 7,
                "prompt_lookup_min": 3,
                "prompt_lookup_max": 5,
            },
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [8],
            },
        ) as runner:
            outputs = runner.llm.chat(
                prompts, params, chat_template_kwargs={"enable_thinking": False}
            )
            assert len(outputs[0].prompt_token_ids) > 256
            accepted = next(
                m.value
                for m in runner.llm.get_metrics()
                if m.name == "vllm:spec_decode_num_accepted_tokens"
            )
            assert accepted > 0, "The test must exercise accepted speculative tokens"
            results.append(
                [
                    (
                        list(o.outputs[0].token_ids),
                        o.outputs[0].text,
                        o.outputs[0].logprobs,
                    )
                    for o in outputs
                ]
            )
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="eager",
        name_1="graph",
    )


@pytest.fixture
def turboquant_workspace():
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    yield
    reset_workspace_manager()


@pytest.mark.parametrize("query_len", [2, 8, 128, 129, 256])
@pytest.mark.parametrize("num_reqs", [1, 2])
@pytest.mark.parametrize("capture", [False, True])
def test_continuation_uses_device_lengths(
    tmp_path, turboquant_workspace, query_len, num_reqs, capture
):
    """Rejected draft tokens must not extend attention, including graph replay."""
    from transformers import GPT2Config

    from vllm.config import (
        ModelConfig,
        ParallelConfig,
        SpeculativeConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.turboquant_attn import (
        TurboQuantAttentionImpl,
        TurboQuantMetadataBuilder,
    )
    from vllm.v1.attention.ops.triton_turboquant_store import (
        triton_turboquant_store,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    # Only the model configuration is needed; no weights or network access.
    GPT2Config(
        architectures=["GPT2LMHeadModel"],
        n_embd=128,
        n_head=1,
        n_layer=1,
        n_positions=1024,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=1024,
        dtype="float16",
    )
    parallel_config = ParallelConfig()
    config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=query_len - 1,
            prompt_lookup_min=3,
            prompt_lookup_max=5,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        ),
    )
    device = torch.device("cuda")
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
    )
    with set_current_vllm_config(config):
        impl = TurboQuantAttentionImpl(
            num_heads=1,
            head_size=128,
            scale=128**-0.5,
            kv_cache_dtype="turboquant_4bit_nc",
        )
        builder = TurboQuantMetadataBuilder(spec, [], config, device)

    layer = torch.nn.Module()
    impl._ensure_on_device(layer, device)

    cfg = impl.tq_config
    capacity = 1024
    torch.manual_seed(62558)
    key = torch.randn(num_reqs * capacity, 1, 128, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    query = torch.randn(
        num_reqs * query_len, 1, 128, device=device, dtype=torch.float16
    )
    cache = torch.zeros(
        num_reqs * capacity // 16,
        16,
        1,
        cfg.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    rotation = _build_hadamard(128, "cuda")
    centroids = get_centroids(128, cfg.centroid_bits).to(device)
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    triton_turboquant_store(
        key,
        value,
        cache,
        torch.arange(num_reqs * capacity, device=device),
        rotation,
        midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )
    qsl_cpu = torch.arange(num_reqs + 1, dtype=torch.int32) * query_len
    common = CommonAttentionMetadata(
        query_start_loc=qsl_cpu.to(device),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=torch.full((num_reqs,), query_len, device=device, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), query_len, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs * query_len,
        max_query_len=query_len,
        max_seq_len=capacity,
        block_table_tensor=torch.arange(
            num_reqs * capacity // 16, device=device, dtype=torch.int32
        ).reshape(num_reqs, -1),
        slot_mapping=torch.arange(num_reqs * query_len, device=device),
    )
    metadata = (
        builder.build_for_cudagraph_capture(common)
        if capture
        else builder.build(0, common)
    )
    # Raw current-chunk K/V deliberately differ from the cached prefix.
    k_chunk, v_chunk = key[: len(query)], value[: len(query)]

    def attend(meta):
        return impl._prefill_attention(
            query,
            k_chunk,
            v_chunk,
            cache,
            meta,
            rotation,
            centroids,
            rotation,
            layer=layer,
        )

    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            attend(metadata)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = attend(metadata)

    # Replay with both shorter and near-capacity contexts, without recapture.
    for context in [query_len + 128, capacity - 8]:
        exact_cpu = torch.arange(num_reqs, dtype=torch.int32) + context
        metadata.seq_lens.copy_(exact_cpu)
        exact = replace(metadata, seq_lens_cpu=exact_cpu)
        expected = attend(exact)
        if capture:
            graph.replay()
            actual = captured_output
        else:
            optimistic = replace(metadata, seq_lens_cpu=exact_cpu + 5)
            actual = attend(optimistic)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
