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

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_backend_cache_shape_matches_page_layout(self, preset, block_size):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheLayout,
            compute_layer_kv_cache_shape_bytes,
            get_kv_quant_mode,
        )

        num_blocks = 7
        num_kv_heads = 3
        head_dim = 128
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=torch.uint8,
            kv_quant_mode=get_kv_quant_mode(preset),
        )
        spec = TurboQuantAttentionBackend.customize_spec(spec)
        shape = compute_layer_kv_cache_shape_bytes(spec, num_blocks)

        assert shape == (
            num_blocks,
            num_kv_heads,
            block_size,
            cfg.slot_size_aligned,
        )
        assert TurboQuantAttentionBackend.supported_kv_cache_layouts() == (
            KVCacheLayout.LBHNC,
        )
        cache = torch.empty(shape, dtype=torch.uint8)
        page_view = cache.view(num_blocks, num_kv_heads, 1, -1)
        assert page_view.shape == (
            num_blocks,
            num_kv_heads,
            1,
            block_size * cfg.slot_size_aligned,
        )


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
        cache_dtype: str = "turboquant_3bit_nc",
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
            cache_config=SimpleNamespace(cache_dtype=cache_dtype),
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
        monkeypatch.setattr(turboquant_attn, "_turboquant_C", None)

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
        monkeypatch.setattr(turboquant_attn, "_turboquant_C", None)

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

    def test_metadata_builder_reserves_optimized_decode_workspace(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn

        workspace_calls = []
        size_calls = []

        class FakeWorkspaceManager:
            def get_simultaneous(self, *shapes_and_dtypes):
                workspace_calls.append(shapes_and_dtypes)

        class FakeExtension:
            @staticmethod
            def workspace_size(*args):
                size_calls.append(args)
                return args[0] * 100

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
        monkeypatch.setattr(turboquant_attn, "_turboquant_C", FakeExtension())
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

        turboquant_attn.TurboQuantMetadataBuilder(
            kv_cache_spec=self._fake_kv_cache_spec(),
            layer_names=["layers.0.self_attn.attn"],
            vllm_config=self._fake_vllm_config(
                dtype=torch.bfloat16,
                cache_dtype="turboquant_4bit_nc",
            ),
            device=torch.device("cuda:0"),
        )

        assert len(size_calls) == 16
        assert workspace_calls[1] == (
            ((1600,), torch.uint8),
            ((16, 8, 128), torch.bfloat16),
        )


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


def _pack_tq_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    codes = codes.to(torch.int32)
    if bits == 4:
        return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    groups = codes.reshape(*codes.shape[:-1], -1, 8)
    shifts = torch.arange(0, 24, 3, device=codes.device, dtype=torch.int32)
    packed = torch.sum(groups << shifts, dim=-1)
    return (
        torch.stack(
            [packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF],
            dim=-1,
        )
        .flatten(-2)
        .to(torch.uint8)
    )


def _reference_turboquant_store(
    key,
    value,
    slot_mapping,
    rotation,
    midpoints,
    config,
    block_size,
    num_blocks,
    fill_value=0xA5,
):
    num_tokens, num_heads, head_dim = key.shape
    cache = torch.full(
        (
            num_blocks,
            num_heads,
            1,
            block_size * config.slot_size_aligned,
        ),
        fill_value,
        device=key.device,
        dtype=torch.uint8,
    )
    written = torch.zeros_like(cache, dtype=torch.bool)

    if config.key_fp8:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("FP8 byte reference requires native e4m3fn conversion")
        key_data = key.float().to(torch.float8_e4m3fn).view(torch.uint8)
        key_data_bytes = head_dim
        key_norm = None
        key_norm_bytes = 0
    else:
        key_fp32 = key.float()
        key_norm = torch.linalg.vector_norm(key_fp32, dim=-1, keepdim=True)
        rotated = (key_fp32 / (key_norm + 1e-8)) @ rotation
        key_codes = torch.bucketize(rotated, midpoints, right=True)
        key_data = _pack_tq_codes(key_codes, config.key_mse_bits)
        key_data_bytes = math.ceil(head_dim * config.key_mse_bits / 8)
        key_norm = key_norm.to(torch.float16).contiguous().view(torch.uint8)
        key_norm_bytes = 2

    value_fp32 = value.float()
    value_min = value_fp32.amin(dim=-1, keepdim=True)
    value_max = value_fp32.amax(dim=-1, keepdim=True)
    levels = 2**config.effective_value_quant_bits - 1
    value_scale = ((value_max - value_min) / levels).clamp_min(1e-8)
    value_codes = (
        ((value_fp32 - value_min) / value_scale + 0.5).to(torch.int32).clamp_(0, levels)
    )
    value_data = _pack_tq_codes(value_codes, config.effective_value_quant_bits)
    value_scale = value_scale.to(torch.float16).contiguous().view(torch.uint8)
    value_zero = value_min.to(torch.float16).contiguous().view(torch.uint8)
    value_data_bytes = math.ceil(head_dim * config.effective_value_quant_bits / 8)

    for token_idx in range(num_tokens):
        slot = int(slot_mapping[token_idx])
        if slot < 0:
            continue
        block_idx, position = divmod(slot, block_size)
        for head_idx in range(num_heads):
            record = cache[block_idx, head_idx, 0]
            record_mask = written[block_idx, head_idx, 0]
            key_base = position * key_data_bytes
            value_plane = block_size * key_data_bytes
            value_base = value_plane + position * value_data_bytes
            norm_plane = value_plane + block_size * value_data_bytes
            scale_plane = norm_plane + block_size * key_norm_bytes
            zero_plane = scale_plane + block_size * 2

            record[key_base : key_base + key_data_bytes] = key_data[token_idx, head_idx]
            record_mask[key_base : key_base + key_data_bytes] = True
            record[value_base : value_base + value_data_bytes] = value_data[
                token_idx, head_idx
            ]
            record_mask[value_base : value_base + value_data_bytes] = True
            if key_norm is not None:
                norm_base = norm_plane + position * 2
                record[norm_base : norm_base + 2] = key_norm[token_idx, head_idx]
                record_mask[norm_base : norm_base + 2] = True
            scale_base = scale_plane + position * 2
            zero_base = zero_plane + position * 2
            record[scale_base : scale_base + 2] = value_scale[token_idx, head_idx]
            record[zero_base : zero_base + 2] = value_zero[token_idx, head_idx]
            record_mask[scale_base : scale_base + 2] = True
            record_mask[zero_base : zero_base + 2] = True
    return cache, written


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestStoreDecodeRoundTrip:
    """End-to-end: store KV into TQ cache, decode, compare vs fp16 ref."""

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
    def test_store_matches_reference_all_modes(self, preset, head_dim, input_dtype):
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        config = TurboQuantConfig.from_cache_dtype(preset, head_dim)
        rotation = _build_hadamard(head_dim, DEVICE_TYPE)
        centroids, _ = solve_lloyd_max(head_dim, config.centroid_bits)
        centroids = centroids.to(device=device, dtype=torch.float32)
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        num_tokens, num_heads, block_size, num_blocks = 4, 2, 32, 2
        torch.manual_seed(20260818)
        key = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=input_dtype,
        )
        value = torch.randn_like(key) * 0.25
        slots = torch.arange(num_tokens, device=device, dtype=torch.int64)
        expected, written = _reference_turboquant_store(
            key,
            value,
            slots,
            rotation,
            midpoints,
            config,
            block_size,
            num_blocks,
        )
        actual = torch.full_like(expected, 0xA5)
        triton_turboquant_store(
            key,
            value,
            actual,
            slots,
            rotation,
            midpoints,
            mse_bits=config.key_mse_bits,
            key_packed_size=config.key_packed_size,
            value_quant_bits=config.effective_value_quant_bits,
            key_fp8=config.key_fp8,
        )

        assert torch.equal(actual[~written], expected[~written])
        byte_diff = (actual[written] != expected[written]).float().mean().item()
        if config.key_fp8:
            assert byte_diff == 0
        else:
            assert byte_diff < 0.05

        query = torch.randn(1, num_heads, head_dim, device=device, dtype=input_dtype)
        block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)

        def decode(cache):
            return triton_turboquant_decode_attention(
                query,
                cache,
                block_table,
                seq_lens,
                rotation,
                centroids,
                scale=1 / math.sqrt(head_dim),
                mse_bits=config.key_mse_bits,
                key_packed_size=config.key_packed_size,
                value_quant_bits=config.effective_value_quant_bits,
                key_fp8=config.key_fp8,
                norm_correction=config.norm_correction,
                PiT=rotation,
                max_num_kv_splits=4,
            )

        torch.testing.assert_close(decode(actual), decode(expected), rtol=0, atol=0.005)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize(
        ("num_heads", "block_size"),
        [(1, 16), (2, 32), (4, 64), (8, 128), (16, 16), (32, 128)],
    )
    def test_store_page_field_abi(self, preset, num_heads, block_size):
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        head_dim = 128
        config = TurboQuantConfig.from_cache_dtype(preset, head_dim)
        rotation = _build_hadamard(head_dim, DEVICE_TYPE)
        centroids, _ = solve_lloyd_max(head_dim, config.centroid_bits)
        centroids = centroids.to(device=device, dtype=torch.float32)
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        slots = torch.tensor(
            [block_size - 1, block_size, -1], device=device, dtype=torch.int64
        )
        torch.manual_seed(20260818)
        key = torch.randn(
            3,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key)
        expected, written = _reference_turboquant_store(
            key,
            value,
            slots,
            rotation,
            midpoints,
            config,
            block_size,
            2,
        )
        actual = torch.full_like(expected, 0xA5)
        triton_turboquant_store(
            key,
            value,
            actual,
            slots,
            rotation,
            midpoints,
            mse_bits=config.key_mse_bits,
            key_packed_size=config.key_packed_size,
            value_quant_bits=config.effective_value_quant_bits,
            key_fp8=config.key_fp8,
        )

        assert torch.equal(actual[~written], expected[~written])
        assert (actual[written] != expected[written]).float().mean() < 0.05

    @pytest.mark.parametrize(
        "preset",
        ["turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc"],
    )
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize(("num_tokens", "num_heads"), [(128, 2), (16, 16), (8, 32)])
    def test_store_m16_matches_reference(self, preset, head_dim, num_tokens, num_heads):
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        block_size = 32
        config = TurboQuantConfig.from_cache_dtype(preset, head_dim)
        rotation = _build_hadamard(head_dim, DEVICE_TYPE)
        centroids, _ = solve_lloyd_max(head_dim, config.centroid_bits)
        centroids = centroids.to(device=device, dtype=torch.float32)
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        slots = torch.arange(num_tokens, device=device, dtype=torch.int64)
        torch.manual_seed(20260818)
        key = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key)
        expected, written = _reference_turboquant_store(
            key,
            value,
            slots,
            rotation,
            midpoints,
            config,
            block_size,
            4,
        )
        actual = torch.full_like(expected, 0xA5)
        triton_turboquant_store(
            key,
            value,
            actual,
            slots,
            rotation,
            midpoints,
            mse_bits=config.key_mse_bits,
            key_packed_size=config.key_packed_size,
            value_quant_bits=config.effective_value_quant_bits,
        )
        assert torch.equal(actual[~written], expected[~written])
        assert (actual[written] != expected[written]).float().mean() < 0.05

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_store_cudagraph_replay(self, preset, head_dim):
        if not torch.cuda.is_available():
            pytest.skip("CUDA graph requires CUDA")
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        config = TurboQuantConfig.from_cache_dtype(preset, head_dim)
        rotation = _build_hadamard(head_dim, "cuda")
        centroids, _ = solve_lloyd_max(head_dim, config.centroid_bits)
        centroids = centroids.to(device="cuda", dtype=torch.float32)
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        key = torch.randn(4, 2, head_dim, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        slots = torch.arange(4, device="cuda", dtype=torch.int64)
        cache_shape = (
            1,
            2,
            1,
            16 * config.slot_size_aligned,
        )
        graph_cache = torch.zeros(cache_shape, device="cuda", dtype=torch.uint8)
        eager_cache = torch.zeros_like(graph_cache)

        def store(cache):
            triton_turboquant_store(
                key,
                value,
                cache,
                slots,
                rotation,
                midpoints,
                mse_bits=config.key_mse_bits,
                key_packed_size=config.key_packed_size,
                value_quant_bits=config.effective_value_quant_bits,
                key_fp8=config.key_fp8,
            )

        store(graph_cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            store(graph_cache)
        key.normal_()
        value.normal_()
        graph_cache.zero_()
        eager_cache.zero_()
        graph.replay()
        store(eager_cache)
        assert torch.equal(graph_cache, eager_cache)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_single_token_roundtrip(self, preset, block_size):
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
        kv_cache = torch.zeros(
            num_blocks,
            Hk,
            1,
            block_size * cfg.slot_size_aligned,
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

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_cross_page_uniform_attention(self, preset, block_size):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            _tq_full_dequant_kv,
            _use_fp8_e4b15,
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        head_dim = 128
        num_heads = 1
        seq_len = block_size + 1
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)
        rotation = torch.eye(head_dim, device=device, dtype=torch.float32)
        centroids, _ = solve_lloyd_max(head_dim, cfg.centroid_bits)
        centroids = centroids.to(device=device, dtype=torch.float32)
        sorted_centroids, _ = centroids.sort()
        midpoints = (sorted_centroids[:-1] + sorted_centroids[1:]) / 2

        key = torch.zeros(
            seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )
        token_values = torch.arange(seq_len, device=device, dtype=torch.float16)
        value_pattern = torch.arange(
            head_dim, device=device, dtype=torch.float16
        ).remainder(8)
        value = token_values[:, None, None] + value_pattern[None, None, :]
        cache_shape = (3, num_heads, 1, block_size * cfg.slot_size_aligned)
        kv_cache = torch.zeros(cache_shape, device=device, dtype=torch.uint8)
        slot_mapping = torch.cat(
            [
                torch.arange(
                    2 * block_size,
                    3 * block_size,
                    device=device,
                    dtype=torch.int32,
                ),
                torch.tensor([0], device=device, dtype=torch.int32),
            ]
        )
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            rotation,
            midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

        query = torch.zeros(1, num_heads, head_dim, device=device, dtype=torch.float16)
        block_table = torch.tensor([[2, 0]], device=device, dtype=torch.int32)
        output = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
            Pi=rotation,
            centroids=centroids,
            scale=1.0 / math.sqrt(head_dim),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=rotation,
            max_num_kv_splits=4,
        )

        assert torch.allclose(
            output,
            block_size / 2 + value_pattern[None, None, :],
            atol=0.25,
            rtol=0,
        )

        key_out = torch.empty(
            1, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
        )
        value_out = torch.empty_like(key_out)
        _tq_full_dequant_kv[(seq_len, num_heads)](
            kv_cache,
            block_table,
            centroids,
            key_out,
            value_out,
            key_out.stride(0),
            key_out.stride(1),
            key_out.stride(2),
            value_out.stride(0),
            value_out.stride(1),
            value_out.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            1,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=num_heads,
            KEY_DATA_BYTES=(
                head_dim if cfg.key_fp8 else math.ceil(head_dim * cfg.key_mse_bits / 8)
            ),
            KEY_NORM_BYTES=0 if cfg.key_fp8 else 2,
            VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=math.ceil(head_dim * cfg.effective_value_quant_bits / 8),
            MSE_BITS=cfg.key_mse_bits,
            KEY_FP8=1 if cfg.key_fp8 else 0,
            BLOCK_D=128,
            NORM_CORRECTION=1 if cfg.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            num_warps=4,
        )
        expected_values = (
            token_values[None, None, :, None] + value_pattern[None, None, None, :]
        )
        assert torch.count_nonzero(key_out) == 0
        assert torch.allclose(value_out, expected_values, atol=0.25, rtol=0)

    def test_d256_long_context_uses_optimized_decode(self, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        from vllm.v1.attention.backends import turboquant_attn

        calls = []

        class FakeExtension:
            @staticmethod
            def workspace_size(*args):
                calls.append(("workspace_size", args))
                return 1

            @staticmethod
            def run_with_workspace(*args):
                calls.append(("run_with_workspace", args[-2:]))
                args[7].zero_()

        class FakeWorkspaceManager:
            def get_simultaneous(self, *shapes_and_dtypes):
                return tuple(
                    torch.empty(shape, device="cuda", dtype=dtype)
                    for shape, dtype in shapes_and_dtypes
                )

        def fail_fallback(*args, **kwargs):
            pytest.fail("D256 long context fell back to production Triton")

        monkeypatch.setattr(turboquant_attn, "_turboquant_C", FakeExtension())
        monkeypatch.setattr(
            turboquant_attn, "is_workspace_manager_initialized", lambda: True
        )
        monkeypatch.setattr(
            turboquant_attn,
            "current_workspace_manager",
            lambda: FakeWorkspaceManager(),
        )
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
        monkeypatch.setattr(
            turboquant_attn, "triton_turboquant_decode_attention", fail_fallback
        )

        head_dim = 256
        num_heads = 12
        num_kv_heads = 2
        page_size = 16
        max_seq_len = 65536
        config = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", head_dim)
        impl = turboquant_attn.TurboQuantAttentionImpl.__new__(
            turboquant_attn.TurboQuantAttentionImpl
        )
        impl.num_heads = num_heads
        impl.num_kv_heads = num_kv_heads
        impl.head_size = head_dim
        impl.scale = head_dim**-0.5
        impl.tq_config = config
        impl.alibi_slopes = None
        impl.sliding_window = None
        impl.logits_soft_cap = None

        query = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        kv_cache = torch.zeros(
            1,
            num_kv_heads,
            1,
            page_size * config.slot_size_aligned,
            device="cuda",
            dtype=torch.uint8,
        )
        block_table = torch.zeros(
            1, max_seq_len // page_size, device="cuda", dtype=torch.int32
        )
        seq_lens = torch.tensor([max_seq_len], device="cuda", dtype=torch.int32)
        metadata = turboquant_attn.TurboQuantMetadata(
            seq_lens=seq_lens,
            slot_mapping=torch.empty(0, device="cuda", dtype=torch.int64),
            block_table=block_table,
            query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=max_seq_len,
            num_decodes=1,
            num_decode_tokens=1,
        )
        rotation = torch.eye(head_dim, device="cuda", dtype=torch.float32)
        centroids = torch.zeros(16, device="cuda", dtype=torch.float32)

        output = impl._decode_attention(query, kv_cache, metadata, rotation, centroids)

        assert output.shape == query.shape
        assert [name for name, _ in calls] == [
            "workspace_size",
            "run_with_workspace",
        ]
        assert calls[0][1][5:7] == (head_dim, max_seq_len)

    @pytest.mark.parametrize(
        ("num_heads", "num_kv_heads"),
        [
            pytest.param(32, 32, id="mha_g1"),
            pytest.param(16, 8, id="gqa_g2"),
            pytest.param(24, 8, id="gqa_g3"),
            pytest.param(32, 8, id="qwen3_8b_g4"),
            pytest.param(40, 8, id="qwen3_14b_g5"),
            pytest.param(12, 2, id="gqa_g6"),
            pytest.param(56, 8, id="gqa_g7"),
            pytest.param(64, 8, id="gqa_g8"),
            pytest.param(32, 4, id="qwen3_30b_a3b_g8"),
            pytest.param(72, 8, id="gqa_g9"),
            pytest.param(128, 8, id="gqa_g16"),
            pytest.param(32, 1, id="mqa_g32"),
        ],
    )
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_optimized_decode_matches_triton_across_page_boundary(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        **kwargs,
    ):
        use_graph = kwargs.pop("use_graph", False)
        assert not kwargs
        if not current_platform.is_cuda():
            pytest.skip("TurboQuant optimized decode requires CUDA")
        if torch.cuda.get_device_capability() not in {(8, 0), (9, 0), (10, 0)}:
            pytest.skip("TurboQuant optimized decode requires SM80, SM90, or SM100")
        try:
            from vllm import _turboquant_C
        except ImportError:
            pytest.skip("TurboQuant optimized decode extension is unavailable")
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        batch_size = 3
        seq_lens_list = [1, block_size, block_size + 1]
        config = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", head_dim)
        rotation = _build_hadamard(head_dim, DEVICE_TYPE)
        centroids, _ = solve_lloyd_max(head_dim, config.centroid_bits)
        centroids = centroids.to(device=device, dtype=torch.float32)
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        block_table = torch.tensor(
            [[4, 1, -1, -1], [0, 5, -1, -1], [3, 2, -1, -1]],
            device=device,
            dtype=torch.int32,
        )
        slot_mapping = torch.tensor(
            [
                int(block_table[batch_idx, token_idx // block_size]) * block_size
                + token_idx % block_size
                for batch_idx, seq_len in enumerate(seq_lens_list)
                for token_idx in range(seq_len)
            ],
            device=device,
            dtype=torch.int64,
        )
        torch.manual_seed(20260817)
        key = torch.randn(
            len(slot_mapping),
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key) * 0.25
        query = torch.randn(
            batch_size,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        kv_cache = torch.zeros(
            6,
            num_kv_heads,
            1,
            block_size * config.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
        allocated_max_seq_len = block_table.shape[1] * block_size
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            rotation,
            midpoints,
            mse_bits=config.key_mse_bits,
            key_packed_size=config.key_packed_size,
            value_quant_bits=config.effective_value_quant_bits,
        )
        reference = triton_turboquant_decode_attention(
            query,
            kv_cache,
            block_table,
            seq_lens,
            rotation,
            centroids,
            scale=1 / math.sqrt(head_dim),
            mse_bits=config.key_mse_bits,
            key_packed_size=config.key_packed_size,
            value_quant_bits=config.effective_value_quant_bits,
            norm_correction=True,
            PiT=rotation,
            max_num_kv_splits=16,
        )
        workspace_size = _turboquant_C.workspace_size(
            batch_size,
            num_heads,
            num_kv_heads,
            block_table.stride(0),
            block_size,
            head_dim,
            allocated_max_seq_len,
            device.index or 0,
        )
        workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
        actual = torch.empty_like(query)
        if use_graph:
            _turboquant_C.run_with_workspace(
                query,
                kv_cache,
                block_table,
                seq_lens,
                rotation,
                centroids,
                workspace,
                actual,
                block_size,
                allocated_max_seq_len,
            )
            graph_block_table = torch.zeros_like(block_table)
            graph_seq_lens = torch.ones_like(seq_lens)
            graph = torch.cuda.CUDAGraph()
            torch.accelerator.synchronize()
            with torch.cuda.graph(graph):
                _turboquant_C.run_with_workspace(
                    query,
                    kv_cache,
                    graph_block_table,
                    graph_seq_lens,
                    rotation,
                    centroids,
                    workspace,
                    actual,
                    block_size,
                    allocated_max_seq_len,
                )
            graph_block_table.copy_(block_table)
            graph_seq_lens.copy_(seq_lens)
            graph.replay()
            first_replay = actual.clone()
            graph.replay()
            assert torch.equal(actual, first_replay)
        else:
            _turboquant_C.run_with_workspace(
                query,
                kv_cache,
                block_table,
                seq_lens,
                rotation,
                centroids,
                workspace,
                actual,
                block_size,
                allocated_max_seq_len,
            )

        torch.testing.assert_close(actual, reference, rtol=0, atol=0.01)
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(1), reference.float().flatten(1), dim=1
        )
        assert cosine.min() > 0.99999

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_optimized_decode_cudagraph_replays_live_metadata(
        self, head_dim, block_size
    ):
        self.test_optimized_decode_matches_triton_across_page_boundary(
            num_heads=32,
            num_kv_heads=8,
            head_dim=head_dim,
            block_size=block_size,
            use_graph=True,
        )

    def test_optimized_decode_dynamic_splits_match_tight_plan(self):
        if not current_platform.is_cuda():
            pytest.skip("TurboQuant optimized decode requires CUDA")
        if torch.cuda.get_device_capability() not in {(8, 0), (9, 0), (10, 0)}:
            pytest.skip("TurboQuant optimized decode requires SM80, SM90, or SM100")
        try:
            from vllm import _turboquant_C
        except ImportError:
            pytest.skip("TurboQuant optimized decode extension is unavailable")
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        head_dim = 128
        num_heads = 16
        num_kv_heads = 8
        block_size = 32
        seq_len = 4096
        allocated_max_seq_len = 40960
        num_blocks = seq_len // block_size
        config = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", head_dim)
        rotation = _build_hadamard(head_dim, DEVICE_TYPE)
        centroids, _ = solve_lloyd_max(head_dim, config.centroid_bits)
        centroids = centroids.to(device=device, dtype=torch.float32)
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        torch.manual_seed(20260819)
        key = torch.randn(
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key) * 0.25
        kv_cache = torch.zeros(
            num_blocks,
            num_kv_heads,
            1,
            block_size * config.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            torch.arange(seq_len, device=device, dtype=torch.int64),
            rotation,
            midpoints,
            mse_bits=config.key_mse_bits,
            key_packed_size=config.key_packed_size,
            value_quant_bits=config.effective_value_quant_bits,
        )
        block_table = torch.full(
            (1, allocated_max_seq_len // block_size),
            -1,
            device=device,
            dtype=torch.int32,
        )
        block_table[0, :num_blocks] = torch.arange(
            num_blocks, device=device, dtype=torch.int32
        )
        query = torch.randn(
            1,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        workspace = torch.empty(
            _turboquant_C.workspace_size(
                1,
                num_heads,
                num_kv_heads,
                block_table.stride(0),
                block_size,
                head_dim,
                allocated_max_seq_len,
                device.index or 0,
            ),
            device=device,
            dtype=torch.uint8,
        )
        output = torch.empty_like(query)
        graph_seq_lens = torch.ones(1, device=device, dtype=torch.int32)
        _turboquant_C.run_with_workspace(
            query,
            kv_cache,
            block_table,
            graph_seq_lens,
            rotation,
            centroids,
            workspace,
            output,
            block_size,
            allocated_max_seq_len,
        )
        graph = torch.cuda.CUDAGraph()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            _turboquant_C.run_with_workspace(
                query,
                kv_cache,
                block_table,
                graph_seq_lens,
                rotation,
                centroids,
                workspace,
                output,
                block_size,
                allocated_max_seq_len,
            )

        for live_seq_len in (4096, 2048):
            graph_seq_lens.fill_(live_seq_len)
            graph.replay()
            graph_output = output.clone()
            tight_output = torch.empty_like(query)
            _turboquant_C.run_with_workspace(
                query,
                kv_cache,
                block_table,
                graph_seq_lens,
                rotation,
                centroids,
                workspace,
                tight_output,
                block_size,
                live_seq_len,
            )
            assert torch.equal(graph_output, tight_output)
