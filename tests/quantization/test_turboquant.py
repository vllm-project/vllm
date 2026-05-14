# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV-cache quantization.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import math
from inspect import signature
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
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend

# ============================================================================
# Helpers
# ============================================================================

ALL_PRESETS = list(TQ_PRESETS.keys())


def _assert_strictly_sorted(seq, name="sequence"):
    for i in range(len(seq) - 1):
        assert seq[i] < seq[i + 1], f"{name} not sorted at index {i}"


def _is_power_of_2(n: int) -> bool:
    return n > 0 and next_power_of_2(n) == n


def _make_turboquant_prefill_impl_stub():
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl

    impl = object.__new__(TurboQuantAttentionImpl)
    impl.scale = 1.0
    impl.num_heads = 1
    impl.num_kv_heads = 1
    impl.head_size = 2
    impl.sliding_window = None
    impl.max_num_kv_splits = 4
    impl.kv_sharing_target_layer_name = None
    impl.tq_config = SimpleNamespace(
        key_mse_bits=4,
        key_packed_size=2,
        effective_value_quant_bits=4,
        key_fp8=False,
        norm_correction=True,
    )
    return impl


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

    def test_backend_supports_known_cache_dtypes(self):
        for preset in ALL_PRESETS:
            assert TurboQuantAttentionBackend.supports_kv_cache_dtype(preset)

    def test_backend_supports_mm_prefix(self):
        assert TurboQuantAttentionBackend.supports_mm_prefix()

    def test_backend_supports_k8v4_for_flash_attention_head_size(self):
        reason = TurboQuantAttentionBackend.supports_combination(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="turboquant_k8v4",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            device_capability=DeviceCapability(8, 6),
        )
        assert reason is None

    def test_backend_rejects_k8v4_when_head_size_exceeds_flash_attention_limit(self):
        reason = TurboQuantAttentionBackend.supports_combination(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="turboquant_k8v4",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            device_capability=DeviceCapability(8, 6),
        )
        assert reason is not None
        assert "turboquant_k8v4 requires FlashAttention-compatible" in reason
        assert "head_size <= 256" in reason

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

    def test_boundary_skip_layers_default_dense_uses_all_layers(self):
        mc = self._dense_model_config(32)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc)
        assert layers == []

    def test_boundary_skip_layers_explicit_guard(self):
        mc = self._dense_model_config(32)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc, 2)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_layers_zero(self):
        mc = self._dense_model_config(32)
        assert TurboQuantConfig.get_boundary_skip_layers(mc, 0) == []

    def test_boundary_skip_layers_small_model(self):
        mc = self._dense_model_config(4)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc, 2)
        assert layers == ["0", "1", "2", "3"]

    def test_boundary_skip_layers_cap_at_half(self):
        mc = self._dense_model_config(8)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc, 10)
        assert len(layers) == 8

    def test_boundary_skip_layers_mixed_attention_layout_uses_targeted_protection(
        self,
    ):
        from types import SimpleNamespace

        mc = SimpleNamespace(
            is_hybrid=False,
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(
                num_hidden_layers=6,
                layer_types=[
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
            ),
        )
        assert TurboQuantConfig.get_boundary_skip_layers(mc) == []

    def test_tq_full_attention_spec_preserves_sliding_window(self):
        from vllm.v1.kv_cache_interface import TQFullAttentionSpec

        spec = TQFullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            tq_slot_size=262,
            sliding_window=512,
        )
        assert spec.sliding_window == 512
        assert spec.real_page_size_bytes == 16 * 2 * 262

    def test_tq_sliding_attention_spec_uses_tq_page_size(self):
        from vllm.v1.kv_cache_interface import TQSlidingWindowSpec

        spec = TQSlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            tq_slot_size=262,
            sliding_window=512,
        )

        assert spec.real_page_size_bytes == 16 * 2 * 262
        assert spec.page_size_bytes == 16 * 2 * 262
        assert (
            spec.max_admission_blocks_per_request(
                max_num_batched_tokens=2048,
                max_model_len=4096,
            )
            == math.ceil((512 - 1 + 2048) / 16) + 1
        )

    def test_tq_sliding_attention_spec_uses_sliding_window_manager(self):
        from vllm.v1.core.single_type_kv_cache_manager import (
            SlidingWindowManager,
            spec_manager_map,
        )
        from vllm.v1.kv_cache_interface import TQSlidingWindowSpec

        spec = TQSlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            tq_slot_size=262,
            sliding_window=512,
        )

        assert spec_manager_map[type(spec)] is SlidingWindowManager

    def test_mixed_tq_full_and_sliding_specs_do_not_collapse_to_uniform_type(self):
        from vllm.v1.kv_cache_interface import (
            TQFullAttentionSpec,
            TQSlidingWindowSpec,
            UniformTypeKVCacheSpecs,
        )

        specs = {
            "full": TQFullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=512,
                head_size_v=512,
                dtype=torch.bfloat16,
                tq_slot_size=518,
            ),
            "sliding": TQSlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=256,
                head_size_v=256,
                dtype=torch.bfloat16,
                tq_slot_size=262,
                sliding_window=512,
            ),
        }

        assert UniformTypeKVCacheSpecs.from_specs(specs) is None

    def test_tq_sliding_hybrid_unification_preserves_tq_page_size(self):
        from vllm.v1.core.kv_cache_utils import unify_hybrid_kv_cache_specs
        from vllm.v1.kv_cache_interface import (
            TQFullAttentionSpec,
            TQSlidingWindowSpec,
        )

        specs = {
            "full": TQFullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=512,
                head_size_v=512,
                dtype=torch.bfloat16,
                tq_slot_size=518,
            ),
            "sliding": TQSlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=256,
                head_size_v=256,
                dtype=torch.bfloat16,
                tq_slot_size=262,
                sliding_window=512,
            ),
        }

        unify_hybrid_kv_cache_specs(specs)

        sliding_spec = specs["sliding"]
        assert isinstance(sliding_spec, TQFullAttentionSpec)
        assert sliding_spec.sliding_window == 512
        assert sliding_spec.tq_slot_size == 262
        assert sliding_spec.real_page_size_bytes == 16 * 2 * 262

    def test_tq_page_size_unification_pads_tq_sliding_spec(self):
        from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
        from vllm.v1.kv_cache_interface import (
            TQFullAttentionSpec,
            TQSlidingWindowSpec,
        )

        full_spec = TQFullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=512,
            head_size_v=512,
            dtype=torch.bfloat16,
            tq_slot_size=518,
        )
        specs = {
            "full": full_spec,
            "sliding": TQSlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=256,
                head_size_v=256,
                dtype=torch.bfloat16,
                tq_slot_size=262,
                sliding_window=512,
            ),
        }

        unified = unify_kv_cache_spec_page_size(specs)

        assert unified["full"].page_size_bytes == full_spec.page_size_bytes
        assert unified["sliding"].page_size_bytes == full_spec.page_size_bytes
        assert unified["sliding"].page_size_padded == full_spec.page_size_bytes

    def test_tq_page_size_unification_with_native_skipped_layers(self):
        from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            SlidingWindowSpec,
            TQFullAttentionSpec,
            TQSlidingWindowSpec,
        )

        specs = {
            "native_full": FullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=512,
                head_size_v=512,
                dtype=torch.bfloat16,
            ),
            "native_sliding": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=256,
                head_size_v=256,
                dtype=torch.bfloat16,
                sliding_window=512,
            ),
            "tq_full": TQFullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=512,
                head_size_v=512,
                dtype=torch.bfloat16,
                tq_slot_size=518,
            ),
            "tq_sliding": TQSlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=256,
                head_size_v=256,
                dtype=torch.bfloat16,
                tq_slot_size=262,
                sliding_window=512,
            ),
        }

        unified = unify_kv_cache_spec_page_size(specs)

        max_page_size = unified["native_full"].page_size_bytes
        assert unified["native_sliding"].page_size_bytes == max_page_size
        assert unified["native_sliding"].block_size == 32
        assert unified["tq_full"].page_size_padded == max_page_size
        assert unified["tq_sliding"].page_size_padded == max_page_size

    def test_decode_launcher_accepts_sliding_window(self):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )

        assert (
            "sliding_window" in signature(triton_turboquant_decode_attention).parameters
        )

    def test_decode_launcher_accepts_mm_prefix_range(self):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )

        assert (
            "mm_prefix_range"
            in signature(triton_turboquant_decode_attention).parameters
        )

    def test_sdpa_mask_applies_mm_prefix_after_sliding_window(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )

        impl = object.__new__(TurboQuantAttentionImpl)
        impl.scale = 1.0
        impl.sliding_window = 2

        captured = {}

        def fake_sdpa(q, k, v, *, attn_mask, scale, enable_gqa):
            del k, v, scale, enable_gqa
            captured["mask"] = attn_mask.clone()
            return torch.zeros_like(q)

        monkeypatch.setattr(
            turboquant_attn.F, "scaled_dot_product_attention", fake_sdpa
        )

        query = torch.zeros(3, 1, 2)
        key = torch.zeros(3, 1, 2)
        value = torch.zeros(3, 1, 2)
        mm_prefix_ranges = torch.tensor([[0, 2]], dtype=torch.int32)

        impl._sdpa_with_causal_and_sliding_mask(
            query,
            key,
            value,
            query_start_pos=0,
            mm_prefix_ranges=mm_prefix_ranges,
        )

        assert torch.equal(captured["mask"], torch.ones(3, 3, dtype=torch.bool))

    def test_prefill_with_mm_prefix_skips_flash_attn(self, monkeypatch):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )

        impl = object.__new__(TurboQuantAttentionImpl)
        impl.scale = 1.0
        impl.sliding_window = None
        impl._can_use_flash_attn = True

        def fail_flash(*args, **kwargs):
            raise AssertionError("flash attention must not run with mm-prefix")

        monkeypatch.setattr(impl, "_flash_attn_varlen", fail_flash)
        monkeypatch.setattr(
            impl,
            "_sdpa_with_causal_and_sliding_mask",
            lambda query, key, value, **kwargs: torch.full_like(query, 3),
        )

        query = torch.zeros(2, 1, 2)
        key = torch.zeros(2, 1, 2)
        value = torch.zeros(2, 1, 2)
        out = impl._prefill_attention_with_kv(
            query,
            key,
            value,
            query_start_pos=0,
            mm_prefix_ranges=torch.tensor([[0, 1]], dtype=torch.int32),
        )

        assert torch.equal(out, torch.full_like(query, 3))

    def test_prefill_without_flash_uses_causal_sdpa_fast_path(self, monkeypatch):
        impl = _make_turboquant_prefill_impl_stub()
        impl._can_use_flash_attn = False

        def fail_masked_sdpa(*args, **kwargs):
            raise AssertionError("unmasked first-chunk prefill must not build a mask")

        monkeypatch.setattr(
            impl,
            "_sdpa_with_causal_and_sliding_mask",
            fail_masked_sdpa,
        )
        monkeypatch.setattr(
            impl,
            "_sdpa_causal_prefill",
            lambda query, key, value: torch.full_like(query, 4),
        )

        query = torch.zeros(3, 1, 2)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[1]], dtype=torch.int32),
            max_query_len=3,
            max_seq_len=3,
        )

        out = impl._prefill_attention(
            query,
            torch.zeros_like(query),
            torch.zeros_like(query),
            torch.empty(0),
            metadata,
            torch.empty(0),
            torch.empty(0),
        )

        assert torch.equal(out, torch.full_like(query, 4))

    def test_large_continuation_without_flash_uses_decode_chunks(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn

        impl = _make_turboquant_prefill_impl_stub()
        impl._can_use_flash_attn = False

        def fail_continuation(*args, **kwargs):
            raise AssertionError("full-dequant continuation path must not run")

        calls = []

        def fake_decode_attention(**kwargs):
            calls.append(kwargs)
            return torch.full_like(kwargs["query"], len(calls))

        monkeypatch.setattr(impl, "_continuation_prefill", fail_continuation)
        monkeypatch.setattr(
            turboquant_attn,
            "triton_turboquant_decode_attention",
            fake_decode_attention,
        )

        q_len = turboquant_attn._CONTINUATION_DECODE_THRESHOLD + 1
        cached_len = 200
        seq_len = cached_len + q_len
        query = torch.zeros(q_len, 1, 2)
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            block_table=torch.tensor([[1, 2, 3]], dtype=torch.int32),
            max_query_len=q_len,
            max_seq_len=seq_len,
        )

        out = impl._prefill_attention(
            query,
            key,
            value,
            torch.empty(0),
            metadata,
            torch.empty(0),
            torch.empty(0),
        )

        assert len(calls) == 2
        assert (
            calls[0]["query"].shape[0] == turboquant_attn._CONTINUATION_DECODE_THRESHOLD
        )
        assert calls[1]["query"].shape[0] == 1
        assert calls[0]["seq_lens"].tolist() == list(range(cached_len + 1, seq_len))
        assert calls[1]["seq_lens"].tolist() == [seq_len]
        assert calls[0]["mm_prefix_range"] is None
        assert torch.equal(
            out,
            torch.cat(
                (
                    torch.ones(turboquant_attn._CONTINUATION_DECODE_THRESHOLD, 1, 2),
                    torch.full((1, 1, 2), 2.0),
                )
            ),
        )

    def test_continuation_threshold_uses_single_decode_chunk(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn

        impl = _make_turboquant_prefill_impl_stub()
        impl._can_use_flash_attn = True

        def fail_continuation(*args, **kwargs):
            raise AssertionError("threshold continuation must use decode chunk")

        calls = []

        def fake_decode_attention(**kwargs):
            calls.append(kwargs)
            return torch.zeros_like(kwargs["query"])

        monkeypatch.setattr(impl, "_continuation_prefill", fail_continuation)
        monkeypatch.setattr(
            turboquant_attn,
            "triton_turboquant_decode_attention",
            fake_decode_attention,
        )

        q_len = turboquant_attn._CONTINUATION_DECODE_THRESHOLD
        cached_len = 200
        seq_len = cached_len + q_len
        query = torch.zeros(q_len, 1, 2)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            block_table=torch.tensor([[1, 2, 3]], dtype=torch.int32),
            max_query_len=q_len,
            max_seq_len=seq_len,
        )

        impl._prefill_attention(
            query,
            torch.zeros_like(query),
            torch.zeros_like(query),
            torch.empty(0),
            metadata,
            torch.empty(0),
            torch.empty(0),
        )

        assert len(calls) == 1
        assert calls[0]["query"].shape[0] == q_len
        assert calls[0]["seq_lens"].tolist() == list(range(cached_len + 1, seq_len + 1))

    def test_large_continuation_with_flash_keeps_flash_path(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn

        impl = _make_turboquant_prefill_impl_stub()
        impl._can_use_flash_attn = True

        def fail_decode(*args, **kwargs):
            raise AssertionError("decode chunk path must not run")

        called = {}

        def fake_continuation(*args, **kwargs):
            called["ran"] = True
            query = args[1]
            return torch.full_like(query, 5)

        monkeypatch.setattr(
            turboquant_attn,
            "triton_turboquant_decode_attention",
            fail_decode,
        )
        monkeypatch.setattr(impl, "_continuation_prefill", fake_continuation)

        q_len = turboquant_attn._CONTINUATION_DECODE_THRESHOLD + 1
        seq_len = 200 + q_len
        query = torch.zeros(q_len, 1, 2)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            block_table=torch.tensor([[1, 2, 3]], dtype=torch.int32),
            max_query_len=q_len,
            max_seq_len=seq_len,
        )

        out = impl._prefill_attention(
            query,
            torch.zeros_like(query),
            torch.zeros_like(query),
            torch.empty(0),
            metadata,
            torch.empty(0),
            torch.empty(0),
        )

        assert called["ran"]
        assert torch.equal(out, torch.full_like(query, 5))

    def test_continuation_with_mm_prefix_uses_decode_chunks(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn

        impl = _make_turboquant_prefill_impl_stub()
        impl._can_use_flash_attn = True

        def fail_continuation(*args, **kwargs):
            raise AssertionError("mm-prefix continuation must not use SDPA fallback")

        calls = []

        def fake_decode_attention(**kwargs):
            calls.append(kwargs)
            return torch.zeros_like(kwargs["query"])

        monkeypatch.setattr(impl, "_continuation_prefill", fail_continuation)
        monkeypatch.setattr(
            turboquant_attn,
            "triton_turboquant_decode_attention",
            fake_decode_attention,
        )

        q_len = turboquant_attn._CONTINUATION_DECODE_THRESHOLD + 1
        seq_len = 200 + q_len
        query = torch.zeros(q_len, 1, 2)
        mm_prefix_range_tensor = torch.tensor(
            [[[0, 64], [128, 192]]],
            dtype=torch.int32,
        )
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            block_table=torch.tensor([[1, 2, 3]], dtype=torch.int32),
            max_query_len=q_len,
            max_seq_len=seq_len,
            mm_prefix_range_tensor=mm_prefix_range_tensor,
        )

        impl._prefill_attention(
            query,
            torch.zeros_like(query),
            torch.zeros_like(query),
            torch.empty(0),
            metadata,
            torch.empty(0),
            torch.empty(0),
        )

        assert len(calls) == 2
        assert calls[0]["mm_prefix_range"].shape == (
            turboquant_attn._CONTINUATION_DECODE_THRESHOLD,
            2,
            2,
        )
        assert calls[0]["mm_prefix_range"].is_contiguous()
        assert calls[1]["mm_prefix_range"].shape == (1, 2, 2)
        assert calls[1]["mm_prefix_range"].is_contiguous()
        assert torch.equal(calls[0]["mm_prefix_range"][0], mm_prefix_range_tensor[0])

    def test_kv_shared_prefill_without_flash_uses_decode_chunks(self, monkeypatch):
        from vllm.v1.attention.backends import turboquant_attn

        impl = _make_turboquant_prefill_impl_stub()
        impl._can_use_flash_attn = False

        def fail_dequant(*args, **kwargs):
            raise AssertionError("KV-sharing prefill must not full-dequant for SDPA")

        calls = []

        def fake_decode_attention(**kwargs):
            calls.append(kwargs)
            return torch.zeros_like(kwargs["query"])

        monkeypatch.setattr(impl, "_dequant_cached_kv", fail_dequant)
        monkeypatch.setattr(
            turboquant_attn,
            "triton_turboquant_decode_attention",
            fake_decode_attention,
        )

        q_len = turboquant_attn._CONTINUATION_DECODE_THRESHOLD + 1
        query = torch.zeros(q_len, 1, 2)
        layer = SimpleNamespace(_tq_PiT=torch.full((2, 2), 9.0))
        impl._cache_prefill_attention(
            layer=layer,
            query=query,
            kv_cache=torch.empty(0),
            block_table=torch.tensor([[1, 2, 3]], dtype=torch.int32),
            seq_len=300 + q_len,
            query_start_pos=300,
            Pi=torch.empty(0),
            centroids=torch.empty(0),
        )

        assert len(calls) == 2
        assert calls[0]["seq_lens"].tolist()[0] == 301
        assert torch.equal(calls[0]["PiT"], layer._tq_PiT)

    def test_decode_workspace_reservation_uses_safe_upper_bound(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            _get_turboquant_decode_workspace_shapes,
        )

        shapes = _get_turboquant_decode_workspace_shapes(
            batch_size=128,
            num_heads=4,
            head_size=512,
            max_num_kv_splits=16,
        )

        assert shapes == (
            ((128, 4, 16, 513), torch.float32),
            ((128, 4, 512), torch.float32),
            ((128, 4), torch.float32),
        )

    def test_dequant_workspace_cache_len_uses_sliding_window_bound(self):
        from types import SimpleNamespace

        from vllm.v1.attention.backends.turboquant_attn import (
            _get_turboquant_dequant_workspace_cache_len,
        )

        cfg = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=131072),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
        )

        assert (
            _get_turboquant_dequant_workspace_cache_len(
                vllm_config=cfg,
                sliding_window=512,
            )
            == 512
        )
        assert (
            _get_turboquant_dequant_workspace_cache_len(
                vllm_config=cfg,
                sliding_window=None,
            )
            == 131072
        )

    def test_dequant_workspace_shapes_round_to_cache_block_size(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            _get_turboquant_dequant_workspace_shapes,
        )

        shapes = _get_turboquant_dequant_workspace_shapes(
            cache_len=130,
            block_size=16,
            num_kv_heads=2,
            head_size=256,
        )

        assert shapes == (
            ((1, 2, 144, 256), torch.float16),
            ((1, 2, 144, 256), torch.float16),
        )

    def test_kv_shared_prefill_uses_cache_attention(self):
        from types import MethodType, SimpleNamespace

        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )

        impl = object.__new__(TurboQuantAttentionImpl)
        impl.num_heads = 1
        impl.num_kv_heads = 1
        impl.head_size = 2
        impl.kv_sharing_target_layer_name = "model.layers.0.self_attn.attn"
        impl.sliding_window = None
        impl._can_use_flash_attn = False

        calls = []

        def fake_cache_prefill_attention(
            self,
            layer,
            query,
            kv_cache,
            block_table,
            seq_len,
            query_start_pos,
            Pi,
            centroids,
            mm_prefix_ranges=None,
        ):
            del mm_prefix_ranges
            calls.append((block_table.clone(), seq_len, query_start_pos))
            return torch.full_like(query, 7)

        impl._cache_prefill_attention = MethodType(fake_cache_prefill_attention, impl)

        query = torch.zeros(2, 1, 2)
        # Shared Gemma4 layers intentionally do not normalize/rotate raw K/V.
        # The TQ prefill path must therefore ignore these tensors and read the
        # target layer's TQ cache instead.
        invalid_raw_key = torch.full((2, 1, 2), 1000.0)
        invalid_raw_value = torch.full((2, 1, 2), -1000.0)
        block_table = torch.tensor([[1]], dtype=torch.int32)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
            block_table=block_table,
            max_query_len=2,
            max_seq_len=2,
        )

        out = impl._prefill_attention(
            query,
            invalid_raw_key,
            invalid_raw_value,
            torch.empty(0),
            metadata,
            torch.empty(0),
            torch.empty(0),
            layer=object(),
        )

        assert torch.equal(out, torch.full_like(query, 7))
        assert len(calls) == 1
        called_block_table, seq_len, query_start_pos = calls[0]
        assert torch.equal(called_block_table, block_table)
        assert seq_len == 2
        assert query_start_pos == 0


class TestHybridAttentionIndices:
    """Regression tests for boundary protection on hybrid models.

    Hybrid models (attention + Mamba / linear-attention) identify KV-carrying
    layers via layer_types / layers_block_type / attn_type_list. The helper
    must return the *global* layer indices of the full-attention layers so
    that kv_cache_dtype_skip_layers matches what extract_layer_index(prefix)
    reports on the Attention layers at runtime.
    """

    @staticmethod
    def _fake_model_config(text_cfg=None, hf_cfg=None, is_hybrid=False):
        from types import SimpleNamespace

        return SimpleNamespace(
            hf_text_config=text_cfg if text_cfg is not None else SimpleNamespace(),
            hf_config=hf_cfg if hf_cfg is not None else SimpleNamespace(),
            is_hybrid=is_hybrid,
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

    def test_mixed_attention_disables_generic_boundary_skip_layers(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        cfg = type("C", (), {})()
        cfg.num_hidden_layers = 42
        cfg.layer_types = [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ] * 7
        mc = self._fake_model_config(text_cfg=cfg, is_hybrid=False)

        assert TurboQuantConfig.get_boundary_skip_layers(mc) == []

    def test_kv_sharing_skip_layers_follow_target_layers(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        cfg = type("C", (), {})()
        cfg.num_hidden_layers = 42
        cfg.num_kv_shared_layers = 18
        cfg.layer_types = [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ] * 7
        mc = self._fake_model_config(text_cfg=cfg)

        assert TurboQuantConfig.align_kv_sharing_skip_layers(
            mc, ["0", "1", "40", "41"]
        ) == ["0", "1"]

        assert TurboQuantConfig.align_kv_sharing_skip_layers(
            mc, ["22", "sliding_window"]
        ) == [
            "22",
            "24",
            "25",
            "26",
            "27",
            "28",
            "30",
            "31",
            "32",
            "33",
            "34",
            "36",
            "37",
            "38",
            "39",
            "40",
            "sliding_window",
        ]

    def test_high_fanout_kv_sharing_target_layers_are_protected(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        cfg = type("C", (), {})()
        cfg.num_hidden_layers = 42
        cfg.num_kv_shared_layers = 18
        cfg.layer_types = [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ] * 7
        mc = self._fake_model_config(text_cfg=cfg)

        assert TurboQuantConfig.get_kv_sharing_target_skip_layers(mc) == ["22"]

        default_skip = TurboQuantConfig.get_boundary_skip_layers(
            mc
        ) + TurboQuantConfig.get_kv_sharing_target_skip_layers(mc)
        assert TurboQuantConfig.align_kv_sharing_skip_layers(mc, default_skip) == [
            "22",
            "24",
            "25",
            "26",
            "27",
            "28",
            "30",
            "31",
            "32",
            "33",
            "34",
            "36",
            "37",
            "38",
            "39",
            "40",
        ]

    def test_kv_sharing_target_protection_uses_majority_fanout(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        cfg = type("C", (), {})()
        cfg.num_hidden_layers = 35
        cfg.num_kv_shared_layers = 20
        cfg.layer_types = [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ] * 7
        mc = self._fake_model_config(text_cfg=cfg)

        assert TurboQuantConfig.get_kv_sharing_target_skip_layers(mc) == ["13"]


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

    @pytest.mark.parametrize(
        "preset",
        ["turboquant_k8v4", "turboquant_4bit_nc"],
    )
    def test_single_token_roundtrip_with_padded_cache_view(self, preset):
        """Store/decode must work when page-size padding makes KV strided."""
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
        Hk = 4
        Hq = 4
        B = 1
        block_size = 16
        num_blocks = 2
        device = torch.device(DEVICE_TYPE)

        H = _build_hadamard(D, DEVICE_TYPE)
        PiT = H
        Pi = H

        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        torch.manual_seed(123)
        key = torch.randn(B, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(B, Hk, D, device=device, dtype=torch.float16)

        padded_slot = cfg.slot_size_aligned
        real_page_size = block_size * Hk * padded_slot
        padded_page_size = real_page_size + 128
        raw_cache = torch.zeros(
            padded_page_size * num_blocks, device=device, dtype=torch.uint8
        )
        kv_cache = torch.as_strided(
            raw_cache,
            size=(num_blocks, block_size, Hk, padded_slot),
            stride=(padded_page_size, Hk * padded_slot, padded_slot, 1),
        )
        assert not kv_cache.is_contiguous()

        slot_mapping = torch.tensor([block_size], device=device, dtype=torch.int32)
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

        query = key.expand(B, Hq, D).contiguous().to(torch.float16)
        block_table = torch.tensor([[1]], device=device, dtype=torch.int32)
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

        out_fp32 = output.float()
        val_fp32 = value.expand(B, Hq, D).float()
        for h in range(Hq):
            cos_sim = torch.nn.functional.cosine_similarity(
                out_fp32[0, h].unsqueeze(0),
                val_fp32[0, h].unsqueeze(0),
            ).item()
            threshold = 0.95 if cfg.key_fp8 else 0.85
            assert cos_sim > threshold, (
                f"Preset {preset} head {h}: cosine_sim={cos_sim:.4f} < {threshold}"
            )

    @pytest.mark.parametrize(
        "preset",
        [
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        ],
    )
    def test_decode_sliding_window_ignores_fully_masked_splits(self, preset):
        """Sliding-window decode must not produce NaN for empty split tiles."""
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        D = 128
        Hk = 2
        Hq = 8
        seq_len = 600
        sliding_window = 512
        block_size = 16
        num_blocks = math.ceil(seq_len / block_size)
        device = torch.device(DEVICE_TYPE)

        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
        H = _build_hadamard(D, DEVICE_TYPE)
        PiT = H
        Pi = H

        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        torch.manual_seed(20260511)
        key = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        query = torch.randn(1, Hq, D, device=device, dtype=torch.float16)

        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            Hk,
            cfg.slot_size_aligned,
            device=device,
            dtype=torch.uint8,
        )
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            torch.arange(seq_len, device=device, dtype=torch.int32),
            PiT,
            midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

        output = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=torch.arange(
                num_blocks, device=device, dtype=torch.int32
            ).unsqueeze(0),
            seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_num_kv_splits=8,
            sliding_window=sliding_window,
        )

        assert torch.isfinite(output).all()
