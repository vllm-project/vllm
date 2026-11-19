# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV-cache quantization.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import math
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
        from vllm.v1.kv_cache_interface import AttentionSpec

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
        layer_config = Attention.get_kv_cache_config(layer, vllm_config)
        assert layer_config is not None
        spec = layer_config.spec
        assert isinstance(spec, AttentionSpec)
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
        from vllm.v1.kv_cache_interface import AttentionSpec

        return AttentionSpec(
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

    @pytest.mark.parametrize(
        "mode,head_size,num_kv_heads,grouped",
        [
            ("TURBOQUANT_4BIT_NC", 128, 4, True),
            ("TURBOQUANT_4BIT_NC", 64, 4, False),  # untuned head sizes
            ("TURBOQUANT_4BIT_NC", 96, 4, False),
            ("TURBOQUANT_4BIT_NC", 512, 4, False),
            ("TURBOQUANT_K3V4_NC", 128, 8, False),  # group size 1
            ("TURBOQUANT_K8V4", 128, 4, False),  # FP8 keys
            ("ULTRAQUANT_4BIT", 128, 4, False),
        ],
    )
    def test_decode_workspace_matches_kernel_selection(
        self, monkeypatch, mode, head_size, num_kv_heads, grouped
    ):
        """Only the grouped MSE-key kernel tiers the split count, so only it
        reserves mid_o rows for the largest batch x splits product."""
        from dataclasses import replace

        from vllm.v1.attention.backends import turboquant_attn as ta
        from vllm.v1.kv_cache_interface import KVQuantMode

        if grouped and not torch.cuda.is_available():
            pytest.skip("the grouped MSE-key kernel is CUDA-only")
        fp8, group = mode == "TURBOQUANT_K8V4", 8 // num_kv_heads
        if mode.startswith("TURBOQUANT"):  # UltraQuant has its own decode
            assert ta._use_mse_gqa(fp8, group, head_size) == grouped
        calls = []
        manager = SimpleNamespace(get_simultaneous=lambda *a: calls.append(a))
        monkeypatch.setattr(ta, "current_workspace_manager", lambda: manager)
        monkeypatch.setattr(ta, "is_workspace_manager_initialized", lambda: True)
        monkeypatch.setattr(ta, "is_flydsl_available", lambda: False)
        spec = self._fake_kv_cache_spec()
        spec = replace(spec, num_kv_heads=num_kv_heads, head_size=head_size)
        spec = replace(spec, kv_quant_mode=KVQuantMode[mode])
        ta.TurboQuantMetadataBuilder(spec, ["l0"], self._fake_vllm_config(), "cuda")
        # 16 seqs x 4 splits; tiers (x4 up to batch 4, x2 up to 16) need 16 * 8
        assert ta._max_decode_split_rows(16, 4) == 128
        mid_o = (128, 8, 1) if grouped else (16, 8, 4)
        assert calls[0][0] == ((*mid_o, head_size + 1), torch.float32)

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
# Decode: grouped MSE-key stage 1, parallel stage 2, KV split tiers
# ============================================================================


def _tq_cache(preset, D, Hk, n, centroids=None, bs=16):
    """Store n random K/V tokens on shuffled pages of a larger pool."""
    from vllm.v1.attention.ops.triton_turboquant_store import (
        triton_turboquant_store,
    )

    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
    H = _build_hadamard(D, "cuda")
    if centroids is None:
        centroids = solve_lloyd_max(D, cfg.centroid_bits)[0].float().cuda()
    c = centroids.sort()[0]
    g = torch.Generator(device="cuda").manual_seed(0)
    k, v = torch.randn(2, n, Hk, D, device="cuda", generator=g).bfloat16()
    pages = -(-n // bs)
    blocks = torch.randperm(pages + 3, device="cuda", generator=g)[:pages].int()
    cache = torch.zeros(pages + 3, Hk, bs, cfg.slot_size_aligned, dtype=torch.uint8)
    cache = cache.cuda().transpose(1, 2)  # vLLM's physical layout
    pos = torch.arange(n, device="cuda")
    slots = (blocks[pos // bs] * bs + pos % bs).int()
    tq = (cfg.key_mse_bits, cfg.key_packed_size, cfg.effective_value_quant_bits)
    triton_turboquant_store(k, v, cache, slots, H, (c[:-1] + c[1:]) / 2, *tq)
    return cfg, cache, blocks, H, centroids


def _tq_reference(cfg, cache, blocks, H, centroids, q, seq_lens, scale=None):
    """fp32 attention over the MSE-key cache dequantized in PyTorch."""
    D, Hk, bs = q.shape[-1], cache.shape[2], cache.shape[1]
    ar = lambda n: torch.arange(n, device="cuda")  # noqa: E731
    f16 = lambda raw: raw.contiguous().view(torch.float16).float()  # noqa: E731

    def codes(raw, bits):  # LSB-first packed codes -> [.., D]
        b = ((raw.long()[..., None] >> ar(8)) & 1).flatten(-2)[..., : D * bits]
        return (b.view(*raw.shape[:-1], D, bits) << ar(bits)).sum(-1)

    pos = ar(int(seq_lens.max()))
    slot = cache[blocks[pos // bs].long(), pos % bs]  # [n, Hk, slot]
    kbits, vbits = cfg.key_mse_bits, cfg.effective_value_quant_bits
    kb, vb = math.ceil(D * kbits / 8), math.ceil(D * vbits / 8)
    c = centroids[codes(slot[..., :kb], kbits)]
    K = c / c.norm(dim=-1, keepdim=True) * f16(slot[..., kb : kb + 2])
    v = slot[..., cfg.key_packed_size :]
    V = codes(v[..., :vb], vbits) * f16(v[..., vb : vb + 2])
    V = V + f16(v[..., vb + 2 : vb + 4])
    out, lse = torch.zeros(q.shape, device="cuda"), torch.zeros(q.shape[:2]).cuda()
    for b, L in enumerate(seq_lens.tolist()):
        qb = (q[b].float() @ H).view(Hk, -1, D)
        sc = torch.einsum("hgd,nhd->hgn", qb, K[:L]) * (scale or D**-0.5)
        out[b] = torch.einsum("hgn,nhd->hgd", sc.softmax(-1), V[:L]).flatten(0, 1)
        lse[b] = sc.logsumexp(-1).flatten()
    return out, lse


def _tq_decode(cfg, cache, blocks, H, cents, q, seq_lens, splits, scale=None, pi=True):
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        triton_turboquant_decode_attention,
    )

    bt, lse = blocks[None].expand(q.shape[0], -1), torch.empty(q.shape[:2]).cuda()
    tq = (cfg.key_mse_bits, cfg.key_packed_size, cfg.effective_value_quant_bits)
    args = (q, cache, bt, seq_lens, H, cents, scale or q.shape[-1] ** -0.5, *tq)
    kw = dict(norm_correction=True, lse_buf=lse, max_num_kv_splits=splits)
    out = triton_turboquant_decode_attention(*args, **kw, pi_is_hadamard=pi)
    return out.float(), lse


def _assert_close_to_ref(out, ref):
    # Measured on H200 over 4 presets x D 128/256 x B 1-32 x ctx 4K-128K: max
    # 7.8e-3 like the fp32 per-head kernel; rtol covers bf16 output rounding.
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=2**-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGroupedDecode:
    @pytest.mark.parametrize("preset", [p for p in ALL_PRESETS if "k8" not in p])
    @pytest.mark.parametrize("D,Hq,Hk", [(128, 8, 2), (256, 12, 2)])
    @pytest.mark.parametrize("splits", [1, 128])
    def test_matches_fp32_reference(self, preset, D, Hq, Hk, splits):
        """Grouped MSE-key stage 1 (fp16 dots, in-kernel Hadamard rotation) +
        stage 2 vs fp32. Rows: tiles crossing pages, empty, 1 token, partial."""
        torch.manual_seed(0)
        args = _tq_cache(preset, D, Hk, 700)
        q = torch.randn(4, Hq, D, device="cuda").bfloat16()
        seq_lens = torch.tensor([700, 0, 1, 333], device="cuda", dtype=torch.int32)
        out, lse = _tq_decode(*args, q, seq_lens, splits)
        ref_out, ref_lse = _tq_reference(*args, q, seq_lens)
        assert (out[1] == 0).all()
        rows = [0, 2, 3]
        _assert_close_to_ref(out[rows], ref_out[rows])
        _assert_close_to_ref(lse[rows], ref_lse[rows])
        err, ref = (out - ref_out)[rows].abs().mean(), ref_out[rows].abs().mean()
        assert err <= 5e-4 + 2**-8 * ref  # bf16 output rounding grows with |out|

    @pytest.mark.parametrize("pi_is_hadamard", [True, False])
    @pytest.mark.parametrize("mag", [3e4, 1e30])
    def test_large_queries_do_not_overflow_fp16(self, pi_is_hadamard, mag):
        """|q_rot| beyond fp16. (a) every head scaled by mag, with a matching
        attention scale so logits stay ~N(0, 1); (b) one large head per group
        next to normal heads, which must keep their precision."""
        torch.manual_seed(0)
        args = _tq_cache("turboquant_4bit_nc", 256, 2, 300)
        seq_lens = torch.tensor([300], device="cuda", dtype=torch.int32)
        q0 = torch.randn(1, 8, 256, device="cuda")
        mixed = q0.clone()
        mixed[:, ::4] *= mag  # head 0 of each group of 4
        for q, scale in ((q0 * mag, 1 / (16 * mag)), (mixed, None)):
            q = q.bfloat16()
            out, lse = _tq_decode(*args, q, seq_lens, 8, scale, pi_is_hadamard)
            ref_out, ref_lse = _tq_reference(*args, q, seq_lens, scale)
            assert torch.isfinite(out).all() and torch.isfinite(lse).all()
            torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-3)
            normal = slice(None) if scale else [1, 2, 3, 5, 6, 7]
            _assert_close_to_ref(out[:, normal], ref_out[:, normal])

    @pytest.mark.parametrize("splits", [1, 7, 256])
    def test_stage2_matches_lse_merge(self, splits):
        """Rows shorter than the split count leave some splits unwritten."""
        from vllm.v1.attention.ops.triton_turboquant_decode import _tq_decode_stage2

        seq_lens = torch.tensor([400, 1, 37], device="cuda", dtype=torch.int32)
        mid_o = torch.randn(3, 8, splits, 129, device="cuda")
        out, lse = torch.empty(3, 8, 128, device="cuda"), torch.empty(3, 8).cuda()
        args = (mid_o, out, lse, seq_lens, *mid_o.stride()[:3], *out.stride()[:2], 8)
        block_s = min(next_power_of_2(splits), 128)
        kw = dict(NUM_KV_SPLITS=splits, HEAD_DIM=128, BLOCK_S=block_s, BLOCK_DC=32)
        _tq_decode_stage2[(3, 8, 4)](*args, **kw)
        for b, L in enumerate(seq_lens.tolist()):
            m = mid_o[b, :, : sum(s * -(-L // splits) < L for s in range(splits))]
            ref = (m[..., 128].softmax(-1)[..., None] * m[..., :128]).sum(1)
            torch.testing.assert_close(out[b], ref)
            torch.testing.assert_close(lse[b], m[..., 128].logsumexp(-1))

    def test_decode_attention_tiers_and_continuation_rows(self, monkeypatch):
        """_decode_attention picks the split tier from the batch size; a short
        continuation chunk in _prefill_attention runs every causal row through
        the decode kernel. Both forward pi_is_hadamard."""
        from vllm.v1.attention.backends import turboquant_attn as ta

        cfg = TestTurboQuantWorkspaceReservation._fake_vllm_config()
        cfg.attention_config.tq_max_kv_splits_for_cuda_graph = 32
        monkeypatch.setattr(ta, "get_current_vllm_config", lambda: cfg)
        preset = "turboquant_k3v4_nc"
        impl = ta.TurboQuantAttentionImpl(
            8, 128, 128**-0.5, num_kv_heads=2, kv_cache_dtype=preset
        )
        layer = torch.nn.Module()
        impl._ensure_on_device(layer, torch.device("cuda"))
        calls, real = [], ta.triton_turboquant_decode_attention
        spy = lambda **kw: calls.append(kw) or real(**kw)  # noqa: E731
        monkeypatch.setattr(ta, "triton_turboquant_decode_attention", spy)
        torch.manual_seed(0)
        args = _tq_cache(preset, 128, 2, 309, centroids=layer._tq_centroids)
        _, cache, blocks, H, cents = args
        for B, tier in ((1, 128), (5, 64), (17, 32)):
            q = torch.randn(B, 8, 128, device="cuda").bfloat16()
            sl = torch.randint(1, 310, (B,), device="cuda", dtype=torch.int32)
            md = SimpleNamespace(block_table=blocks[None].expand(B, -1), seq_lens=sl)
            out = impl._decode_attention(q, cache, md, H, cents, H, layer)
            assert calls[-1]["max_num_kv_splits"] == (32 if impl._use_flydsl else tier)
            _assert_close_to_ref(out.float(), _tq_reference(*args, q, sl)[0])
        q, kv = torch.randn(9, 8, 128).cuda().bfloat16(), torch.zeros(9, 2, 128).cuda()
        qsl, sl = torch.tensor([0, 9], dtype=torch.int32), torch.tensor([309])
        md = SimpleNamespace(max_query_len=9, max_seq_len=309, block_table=blocks[None])
        md.query_start_loc, md.query_start_loc_cpu = qsl.cuda(), qsl
        md.seq_lens, md.seq_lens_cpu = sl.int().cuda(), sl
        out = impl._prefill_attention(q, kv, kv, cache, md, H, cents, H, layer)
        assert all(c.get("pi_is_hadamard") for c in calls)
        sl = torch.arange(301, 310, device="cuda", dtype=torch.int32)  # causal rows
        _assert_close_to_ref(out.float(), _tq_reference(*args, q, sl)[0])
