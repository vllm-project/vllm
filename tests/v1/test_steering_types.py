# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for steering vector type helpers.

Covers:
- normalize_layer_entry: bare list and dict-with-scale forms
- resolve_effective_vectors: additive merge semantics
- hash_steering_config: deterministic hashing behaviour
- Co-located scale application
"""

import numpy as np
import pytest

from vllm.config.steering_types import (
    hash_steering_config,
    merge_steering_specs,
    normalize_layer_entry,
    resolve_effective_vectors,
    scale_steering_spec,
)

# -----------------------------------------------------------------------
# normalize_layer_entry
# -----------------------------------------------------------------------


class TestNormalizeLayerEntry:
    """Validate bare-list and dict-with-scale entry normalisation."""

    def test_bare_list_returns_scale_one(self):
        vec = [1.0, 2.0, 3.0]
        result_vec, result_scale = normalize_layer_entry(vec)
        assert result_vec == [1.0, 2.0, 3.0]
        assert result_scale == 1.0

    def test_dict_with_scale(self):
        entry = {"vector": [4.0, 5.0], "scale": 0.5}
        result_vec, result_scale = normalize_layer_entry(entry)
        assert result_vec == [4.0, 5.0]
        assert result_scale == 0.5

    def test_dict_with_integer_scale(self):
        """Integer scale should be coerced to float."""
        entry = {"vector": [1.0], "scale": 2}
        _, scale = normalize_layer_entry(entry)
        assert isinstance(scale, float)
        assert scale == 2.0

    def test_dict_missing_scale_raises(self):
        """Dict with 'vector' but no 'scale' should raise ValueError."""
        entry = {"vector": [1.0, 2.0]}
        with pytest.raises(ValueError, match="missing required key"):
            normalize_layer_entry(entry)

    def test_dict_missing_vector_raises(self):
        """Dict with 'scale' but no 'vector' should raise ValueError."""
        entry = {"scale": 2.0}
        with pytest.raises(ValueError, match="missing required key"):
            normalize_layer_entry(entry)

    def test_dict_missing_both_keys_raises(self):
        """Dict with neither 'vector' nor 'scale' should raise ValueError.

        With extra-key validation, the unexpected key is caught first.
        """
        entry = {"foo": "bar"}
        with pytest.raises(ValueError, match="unexpected keys"):
            normalize_layer_entry(entry)

    def test_dict_missing_key_error_lists_missing(self):
        """Error message should list the specific missing key(s)."""
        entry = {"vector": [1.0]}
        with pytest.raises(ValueError, match=r"\['scale'\]"):
            normalize_layer_entry(entry)

    def test_dict_with_extra_keys_raises(self):
        """Dict with extra keys beyond 'vector' and 'scale' should raise."""
        entry = {"vector": [1.0, 2.0], "scale": 1.0, "extra": "bad"}
        with pytest.raises(ValueError, match="unexpected keys"):
            normalize_layer_entry(entry)

    def test_dict_with_multiple_extra_keys_raises(self):
        """All extra keys should be listed in error."""
        entry = {"vector": [1.0], "scale": 1.0, "foo": 1, "bar": 2}
        with pytest.raises(ValueError, match="unexpected keys"):
            normalize_layer_entry(entry)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="SteeringLayerEntry must be"):
            normalize_layer_entry("not a list or dict")  # type: ignore[arg-type]

    def test_ndarray_returns_scale_one(self):
        """ndarray entries (produced by merge_steering_specs) pass through
        as ``(arr, 1.0)`` so they can be re-fed into resolve_effective_vectors
        / scale_steering_spec without conversion to list."""
        arr = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        result_vec, result_scale = normalize_layer_entry(arr)
        assert isinstance(result_vec, np.ndarray)
        assert np.array_equal(result_vec, arr)
        assert result_scale == 1.0


# -----------------------------------------------------------------------
# resolve_effective_vectors
# -----------------------------------------------------------------------


class TestResolveEffectiveVectors:
    """Additive merge semantics for base + phase-specific vectors."""

    def test_both_none_returns_none(self):
        assert resolve_effective_vectors(None, None) is None

    def test_both_empty_returns_none(self):
        assert resolve_effective_vectors({}, {}) is None

    def test_only_base_returns_prescaled(self):
        base = {"hp": {0: [1.0, 2.0]}}
        result = resolve_effective_vectors(base, None)
        assert result is not None
        assert result["hp"][0].tolist() == [1.0, 2.0]

    def test_only_base_with_scale(self):
        base = {"hp": {0: {"vector": [1.0, 2.0], "scale": 3.0}}}
        result = resolve_effective_vectors(base, None)
        assert result is not None
        assert result["hp"][0].tolist() == [3.0, 6.0]

    def test_only_phase_returns_prescaled(self):
        phase = {"hp": {0: [5.0, 10.0]}}
        result = resolve_effective_vectors(None, phase)
        assert result is not None
        assert result["hp"][0].tolist() == [5.0, 10.0]

    def test_only_phase_with_scale(self):
        phase = {"hp": {0: {"vector": [1.0, 2.0], "scale": 0.5}}}
        result = resolve_effective_vectors(None, phase)
        assert result is not None
        assert result["hp"][0].tolist() == [0.5, 1.0]

    def test_additive_merge_same_hook_layer(self):
        """Overlapping (hook, layer) pairs should sum."""
        base = {"hp": {0: [1.0, 2.0]}}
        phase = {"hp": {0: [10.0, 20.0]}}
        result = resolve_effective_vectors(base, phase)
        assert result is not None
        assert result["hp"][0].tolist() == [11.0, 22.0]

    def test_additive_merge_with_scales(self):
        """Both base and phase are pre-scaled before summing."""
        base = {"hp": {0: {"vector": [1.0, 2.0], "scale": 2.0}}}
        phase = {"hp": {0: {"vector": [10.0, 20.0], "scale": 0.5}}}
        result = resolve_effective_vectors(base, phase)
        assert result is not None
        # base: [2.0, 4.0], phase: [5.0, 10.0] -> [7.0, 14.0]
        assert result["hp"][0].tolist() == [7.0, 14.0]

    def test_non_overlapping_hooks_pass_through(self):
        """Non-overlapping hook points should pass through unchanged."""
        base = {"hp_a": {0: [1.0]}}
        phase = {"hp_b": {0: [2.0]}}
        result = resolve_effective_vectors(base, phase)
        assert result is not None
        assert result["hp_a"][0].tolist() == [1.0]
        assert result["hp_b"][0].tolist() == [2.0]

    def test_non_overlapping_layers_pass_through(self):
        """Non-overlapping layers within same hook pass through."""
        base = {"hp": {0: [1.0]}}
        phase = {"hp": {1: [2.0]}}
        result = resolve_effective_vectors(base, phase)
        assert result is not None
        assert result["hp"][0].tolist() == [1.0]
        assert result["hp"][1].tolist() == [2.0]

    def test_mixed_overlap_and_nonoverlap(self):
        """Overlapping entries add; non-overlapping pass through."""
        base = {
            "hp_a": {0: [1.0], 1: [10.0]},
            "hp_b": {0: [100.0]},
        }
        phase = {
            "hp_a": {0: [0.5], 2: [20.0]},
            "hp_c": {0: [200.0]},
        }
        result = resolve_effective_vectors(base, phase)
        assert result is not None
        # hp_a layer 0: overlapping -> 1.0 + 0.5 = 1.5
        assert result["hp_a"][0].tolist() == [1.5]
        # hp_a layer 1: only base
        assert result["hp_a"][1].tolist() == [10.0]
        # hp_a layer 2: only phase
        assert result["hp_a"][2].tolist() == [20.0]
        # hp_b: only base
        assert result["hp_b"][0].tolist() == [100.0]
        # hp_c: only phase
        assert result["hp_c"][0].tolist() == [200.0]

    def test_mismatched_vector_lengths_raises(self):
        base = {"hp": {0: [1.0, 2.0]}}
        phase = {"hp": {0: [1.0]}}
        with pytest.raises(ValueError, match="different lengths"):
            resolve_effective_vectors(base, phase)

    def test_accepts_merge_specs_output(self):
        """resolve_effective_vectors must accept the ndarray-valued entries
        produced by merge_steering_specs (which is how the worker-side
        named-module resolver feeds merged specs back through the resolver)."""
        merged_base = merge_steering_specs({"hp": {0: [1.0, 2.0, 3.0]}}, None)
        merged_phase = merge_steering_specs({"hp": {0: [10.0, 20.0, 30.0]}}, None)
        # Sanity-check the intermediate shape that triggered the bug.
        assert isinstance(merged_base["hp"][0], np.ndarray)
        assert isinstance(merged_phase["hp"][0], np.ndarray)
        result = resolve_effective_vectors(merged_base, merged_phase)
        assert result is not None
        assert result["hp"][0].tolist() == [11.0, 22.0, 33.0]


# -----------------------------------------------------------------------
# merge_steering_specs
# -----------------------------------------------------------------------


class TestMergeSteeringSpecs:
    """Validate additive merge semantics and downstream-compat of outputs.

    ``merge_steering_specs`` may produce ``np.ndarray``-valued entries (via
    ``_scale_vector``).  Downstream consumers — ``resolve_effective_vectors``,
    ``scale_steering_spec`` — must handle that shape.  These tests pin the
    contract that callers (notably the worker-side named-module resolver
    in ``SteeringModelRunnerMixin._resolve_request_steering_for_phase``) can
    chain merge → resolve / merge → scale without intermediate conversion.
    """

    def test_both_none_returns_none(self):
        assert merge_steering_specs(None, None) is None

    def test_only_a_pass_through(self):
        a = {"hp": {0: [1.0, 2.0]}}
        result = merge_steering_specs(a, None)
        assert result is not None
        assert result["hp"][0].tolist() == [1.0, 2.0]

    def test_only_b_pass_through(self):
        b = {"hp": {0: [3.0, 4.0]}}
        result = merge_steering_specs(None, b)
        assert result is not None
        assert result["hp"][0].tolist() == [3.0, 4.0]

    def test_overlapping_entries_sum(self):
        a = {"hp": {0: [1.0, 2.0]}}
        b = {"hp": {0: [10.0, 20.0]}}
        result = merge_steering_specs(a, b)
        assert result is not None
        assert result["hp"][0].tolist() == [11.0, 22.0]

    def test_chained_merge_resolve(self):
        """merge → merge → resolve, mirroring _resolve_request_steering_for_phase
        when both module spec and inline spec are present on each tier."""
        scaled_base = {"hp": {0: [1.0, 2.0]}}
        inline_base = {"hp": {1: [5.0, 6.0]}}
        merged_base = merge_steering_specs(scaled_base, inline_base)
        result = resolve_effective_vectors(merged_base, None)
        assert result is not None
        assert result["hp"][0].tolist() == [1.0, 2.0]
        assert result["hp"][1].tolist() == [5.0, 6.0]

    def test_chained_merge_scale_spec(self):
        """scale_steering_spec must accept ndarray entries from merge output."""
        merged = merge_steering_specs({"hp": {0: [1.0, 2.0]}}, None)
        scaled = scale_steering_spec(merged, 3.0)
        assert scaled is not None
        # scale_steering_spec wraps non-1.0 multipliers into the dict form.
        entry = scaled["hp"][0]
        assert isinstance(entry, dict)
        assert np.array_equal(np.asarray(entry["vector"]), np.asarray([1.0, 2.0]))
        assert entry["scale"] == 3.0


# -----------------------------------------------------------------------
# hash_steering_config
# -----------------------------------------------------------------------


class TestHashSteeringConfig:
    """Deterministic hashing of pre-resolved steering vectors."""

    def test_none_returns_zero(self):
        assert hash_steering_config(None) == 0

    def test_empty_dict_returns_zero(self):
        assert hash_steering_config({}) == 0

    def test_same_vectors_same_hash(self):
        vecs = {"hp": {0: [1.0, 2.0], 1: [3.0, 4.0]}}
        assert hash_steering_config(vecs) == hash_steering_config(vecs)

    def test_different_vectors_different_hash(self):
        vecs_a = {"hp": {0: [1.0, 2.0]}}
        vecs_b = {"hp": {0: [1.0, 3.0]}}
        assert hash_steering_config(vecs_a) != hash_steering_config(vecs_b)

    def test_hash_is_positive_int64(self):
        """Hash must fit in np.int64 (positive, <= 2^63 - 1)."""
        vecs = {"hp": {0: [1.0, 2.0, 3.0]}}
        h = hash_steering_config(vecs)
        assert h > 0
        assert h <= (2**63 - 1)

    def test_order_independence(self):
        """Dict ordering should not affect the hash."""
        vecs_a = {"hp_b": {1: [2.0]}, "hp_a": {0: [1.0]}}
        vecs_b = {"hp_a": {0: [1.0]}, "hp_b": {1: [2.0]}}
        assert hash_steering_config(vecs_a) == hash_steering_config(vecs_b)


# -----------------------------------------------------------------------
# Co-located scale application
# -----------------------------------------------------------------------


class TestCoLocatedScale:
    """Verify that scale factors correctly multiply vector values."""

    def test_scale_doubles_values(self):
        """scale=2.0 should double all vector components."""
        spec = {"hp": {0: {"vector": [1.0, 2.0, 3.0], "scale": 2.0}}}
        result = resolve_effective_vectors(spec, None)
        assert result is not None
        assert result["hp"][0].tolist() == [2.0, 4.0, 6.0]

    def test_scale_zero_zeros_vector(self):
        """scale=0.0 should zero out the vector."""
        spec = {"hp": {0: {"vector": [1.0, 2.0], "scale": 0.0}}}
        result = resolve_effective_vectors(spec, None)
        assert result is not None
        assert result["hp"][0].tolist() == [0.0, 0.0]

    def test_negative_scale(self):
        """Negative scales should negate the vector."""
        spec = {"hp": {0: {"vector": [1.0, -2.0], "scale": -1.0}}}
        result = resolve_effective_vectors(spec, None)
        assert result is not None
        assert result["hp"][0].tolist() == [-1.0, 2.0]

    def test_fractional_scale(self):
        """Fractional scale applied correctly."""
        spec = {"hp": {0: {"vector": [10.0, 20.0], "scale": 0.1}}}
        result = resolve_effective_vectors(spec, None)
        assert result is not None
        assert result["hp"][0] == pytest.approx([1.0, 2.0])


# -----------------------------------------------------------------------
# Cross-validation: base vs phase dimension mismatch at SamplingParams level
# -----------------------------------------------------------------------


class TestDimensionCrossValidation:
    """Ensure SamplingParams rejects mismatched dimensions between
    base and phase-specific steering vectors at construction time."""

    def test_mismatched_base_prefill_raises(self):
        """base dim=2 vs prefill dim=1 on same (hook, layer) -> ValueError."""
        from vllm.sampling_params import SamplingParams

        with pytest.raises(
            ValueError, match="Overlapping entries must have matching dimensions"
        ):
            SamplingParams(
                steering_vectors={
                    "post_mlp": {15: [1.0, 2.0]},
                },
                prefill_steering_vectors={
                    "post_mlp": {15: [1.0]},
                },
            )

    def test_mismatched_base_decode_raises(self):
        """base dim=3 vs decode dim=2 on same (hook, layer) -> ValueError."""
        from vllm.sampling_params import SamplingParams

        with pytest.raises(
            ValueError, match="Overlapping entries must have matching dimensions"
        ):
            SamplingParams(
                steering_vectors={
                    "post_mlp": {0: [1.0, 2.0, 3.0]},
                },
                decode_steering_vectors={
                    "post_mlp": {0: [1.0, 2.0]},
                },
            )

    def test_matching_dimensions_pass(self):
        """Overlapping entries with same dimension should pass."""
        from vllm.sampling_params import SamplingParams

        params = SamplingParams(
            steering_vectors={
                "post_mlp": {0: [1.0, 2.0]},
            },
            prefill_steering_vectors={
                "post_mlp": {0: [3.0, 4.0]},
            },
        )
        assert params.steering_vectors is not None
        assert params.prefill_steering_vectors is not None

    def test_non_overlapping_different_dims_pass(self):
        """Non-overlapping entries may have different dimensions
        since they never get added together."""
        from vllm.sampling_params import SamplingParams

        params = SamplingParams(
            steering_vectors={
                "post_mlp": {0: [1.0, 2.0]},
            },
            prefill_steering_vectors={
                "post_mlp": {1: [1.0]},
            },
        )
        assert params.steering_vectors is not None
        assert params.prefill_steering_vectors is not None

    def test_mismatched_prefill_decode_without_base_raises(self):
        """prefill dim=2 vs decode dim=1 on same (hook, layer) without base
        -> ValueError."""
        from vllm.sampling_params import SamplingParams

        with pytest.raises(
            ValueError, match="Overlapping entries must have matching dimensions"
        ):
            SamplingParams(
                prefill_steering_vectors={
                    "post_mlp": {0: [1.0, 2.0]},
                },
                decode_steering_vectors={
                    "post_mlp": {0: [1.0]},
                },
            )

    def test_non_overlapping_prefill_decode_pass(self):
        """Non-overlapping (hook, layer) between prefill and decode with no
        base should construct without error even with different dims."""
        from vllm.sampling_params import SamplingParams

        params = SamplingParams(
            prefill_steering_vectors={
                "post_mlp": {0: [1.0, 2.0]},
            },
            decode_steering_vectors={
                "post_mlp": {1: [1.0]},
            },
        )
        assert params.prefill_steering_vectors is not None
        assert params.decode_steering_vectors is not None

    def test_matching_prefill_decode_without_base_pass(self):
        """Matching (hook, layer) and dimensions between prefill and decode
        with no base should construct without error."""
        from vllm.sampling_params import SamplingParams

        params = SamplingParams(
            prefill_steering_vectors={
                "post_mlp": {0: [1.0, 2.0]},
            },
            decode_steering_vectors={
                "post_mlp": {0: [3.0, 4.0]},
            },
        )
        assert params.prefill_steering_vectors is not None
        assert params.decode_steering_vectors is not None

    def test_mismatched_prefill_decode_scaled_entry_raises(self):
        """Scaled {vector, scale} entry on one side should still be caught
        for dimension mismatches between prefill and decode."""
        from vllm.sampling_params import SamplingParams

        with pytest.raises(
            ValueError, match="Overlapping entries must have matching dimensions"
        ):
            SamplingParams(
                prefill_steering_vectors={
                    "post_mlp": {
                        0: {"vector": [1.0, 2.0], "scale": 0.5},
                    },
                },
                decode_steering_vectors={
                    "post_mlp": {0: [1.0]},
                },
            )


# -----------------------------------------------------------------------
# SamplingParams extra-key rejection
# -----------------------------------------------------------------------


class TestSamplingParamsExtraKeyRejection:
    """SamplingParams._validate_layer_entry should reject extra keys
    in scaled steering dict entries."""

    def test_extra_key_in_steering_vectors_raises(self):
        """Extra key in a steering_vectors scaled entry should raise."""
        from vllm.sampling_params import SamplingParams

        with pytest.raises(ValueError, match="unexpected keys"):
            SamplingParams(
                steering_vectors={
                    "post_mlp": {
                        0: {"vector": [1.0, 2.0], "scale": 1.0, "typo": "bad"},
                    },
                },
            )

    def test_extra_key_in_prefill_steering_vectors_raises(self):
        """Extra key in a prefill_steering_vectors scaled entry should raise."""
        from vllm.sampling_params import SamplingParams

        with pytest.raises(ValueError, match="unexpected keys"):
            SamplingParams(
                prefill_steering_vectors={
                    "post_mlp": {
                        0: {"vector": [1.0], "scale": 1.0, "extra": 42},
                    },
                },
            )

    def test_extra_key_in_decode_steering_vectors_raises(self):
        """Extra key in a decode_steering_vectors scaled entry should raise."""
        from vllm.sampling_params import SamplingParams

        with pytest.raises(ValueError, match="unexpected keys"):
            SamplingParams(
                decode_steering_vectors={
                    "post_mlp": {
                        0: {"vector": [1.0], "scale": 1.0, "foo": 1, "bar": 2},
                    },
                },
            )
