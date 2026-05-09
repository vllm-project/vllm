# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the named-module pre-resolved spec cache.

Asserts the fast path in ``SteeringModelRunnerMixin._resolve_request_steering``
returns numerically identical output to the slow merge+resolve path for:

- scale == 1.0, no inline overrides → cache hit, returns cached array
- scale != 1.0, no inline overrides → cache hit, returns cached * scale
- inline overrides present → cache miss, falls through to slow path
- prefill vs decode resolution
- registry/cache invalidation on register (replace=True) and unregister
"""

import numpy as np
import pytest

from vllm.config.steering_types import resolve_effective_vectors
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_model_runner_mixin import (
    SteeringModelRunnerMixin,
)


class _StubMixin(SteeringModelRunnerMixin):
    """Minimal subclass that initialises just the registry state.

    The real mixin's ``_init_steering_state`` walks a loaded model.  For
    pure resolver tests we sidestep that by setting the two cache dicts
    directly, mirroring the post-init shape.
    """

    def __init__(self):
        self._steering_module_registry = {}
        self._steering_module_resolved_cache = {}


def _spec(hook: str, layer_to_vec: dict[int, list[float]]):
    """Convenience: build a one-hook SteeringVectorSpec."""
    return {hook: dict(layer_to_vec)}


def _arrays_equal(
    a: dict[str, dict[int, np.ndarray]] | None,
    b: dict[str, dict[int, np.ndarray]] | None,
) -> bool:
    if a is None or b is None:
        return a is b
    if a.keys() != b.keys():
        return False
    for hook in a:
        if a[hook].keys() != b[hook].keys():
            return False
        for layer in a[hook]:
            if not np.allclose(a[hook][layer], b[hook][layer]):
                return False
    return True


# ---------------------------------------------------------------------------
# Cache-hit fast path
# ---------------------------------------------------------------------------


class TestNamedCacheFastPath:
    def test_scale_one_no_overrides_returns_cached(self):
        """scale=1.0 + no inline → cache hit; output equals slow-path output."""
        mixin = _StubMixin()
        base = _spec("post_mlp", {0: [1.0, 2.0]})
        prefill = _spec("post_mlp", {1: [3.0, 4.0]})
        mixin.register_steering_modules(
            {"m": {"vectors": base, "prefill_vectors": prefill}},
            replace=True,
        )
        sp = SamplingParams(steering_module_ref=("m", 1.0))

        fast = mixin._resolve_request_steering(sp, "prefill")
        slow = resolve_effective_vectors(base, prefill)
        assert _arrays_equal(fast, slow)

    def test_decode_phase_resolves_separately(self):
        """The cache holds (prefill, decode) — decode must use its slot."""
        mixin = _StubMixin()
        base = _spec("post_mlp", {0: [1.0, 2.0]})
        prefill = _spec("post_mlp", {0: [10.0, 20.0]})
        decode = _spec("post_mlp", {0: [100.0, 200.0]})
        mixin.register_steering_modules(
            {
                "m": {
                    "vectors": base,
                    "prefill_vectors": prefill,
                    "decode_vectors": decode,
                }
            },
            replace=True,
        )
        sp = SamplingParams(steering_module_ref=("m", 1.0))

        fast_prefill = mixin._resolve_request_steering(sp, "prefill")
        fast_decode = mixin._resolve_request_steering(sp, "decode")
        assert fast_prefill["post_mlp"][0].tolist() == [11.0, 22.0]
        assert fast_decode["post_mlp"][0].tolist() == [101.0, 202.0]

    def test_scaled_fast_path_multiplies_cached(self):
        """scale=0.5 + no inline → fast path returns cached * 0.5."""
        mixin = _StubMixin()
        base = _spec("post_mlp", {0: [2.0, 4.0]})
        mixin.register_steering_modules({"m": {"vectors": base}}, replace=True)
        sp = SamplingParams(steering_module_ref=("m", 0.5))

        fast = mixin._resolve_request_steering(sp, "prefill")
        # base resolved alone: [2.0, 4.0]; scaled by 0.5: [1.0, 2.0]
        assert fast["post_mlp"][0].tolist() == [1.0, 2.0]

    def test_scaled_fast_path_matches_slow_path(self):
        """For scale!=1.0, fast and slow paths must agree numerically."""
        mixin = _StubMixin()
        base = _spec("post_mlp", {0: [1.0, 2.0], 1: [3.0, 4.0]})
        prefill = _spec("post_mlp", {0: [10.0, 20.0]})
        mixin.register_steering_modules(
            {"m": {"vectors": base, "prefill_vectors": prefill}},
            replace=True,
        )
        sp = SamplingParams(steering_module_ref=("m", 0.25))

        fast = mixin._resolve_request_steering(sp, "prefill")

        # Slow-path reference using the merge+resolve machinery directly.
        from vllm.config.steering_types import (
            merge_steering_specs,
            scale_steering_spec,
        )

        scaled_base = scale_steering_spec(base, 0.25)
        scaled_phase = scale_steering_spec(prefill, 0.25)
        merged_base = merge_steering_specs(scaled_base, None)
        merged_phase = merge_steering_specs(scaled_phase, None)
        slow = resolve_effective_vectors(merged_base, merged_phase)
        assert _arrays_equal(fast, slow)

    def test_decode_only_module_returns_none_for_prefill(self):
        """If module has only decode_vectors and no base, prefill returns None."""
        mixin = _StubMixin()
        decode = _spec("post_mlp", {0: [1.0, 2.0]})
        mixin.register_steering_modules({"m": {"decode_vectors": decode}}, replace=True)
        sp = SamplingParams(steering_module_ref=("m", 1.0))

        assert mixin._resolve_request_steering(sp, "prefill") is None
        decoded = mixin._resolve_request_steering(sp, "decode")
        assert decoded is not None
        assert decoded["post_mlp"][0].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Cache-miss slow path (inline overrides present)
# ---------------------------------------------------------------------------


class TestInlineOverrideFallback:
    def test_inline_base_falls_through(self):
        """Inline ``steering_vectors`` forces the merge path."""
        mixin = _StubMixin()
        base = _spec("post_mlp", {0: [1.0, 2.0]})
        mixin.register_steering_modules({"m": {"vectors": base}}, replace=True)
        inline = _spec("post_mlp", {0: [10.0, 20.0]})
        sp = SamplingParams(
            steering_module_ref=("m", 1.0),
            steering_vectors=inline,
        )

        result = mixin._resolve_request_steering(sp, "prefill")
        assert result is not None
        # base + inline = [1.0, 2.0] + [10.0, 20.0] = [11.0, 22.0]
        assert result["post_mlp"][0].tolist() == [11.0, 22.0]

    def test_inline_phase_falls_through(self):
        """Inline ``prefill_steering_vectors`` forces the merge path."""
        mixin = _StubMixin()
        base = _spec("post_mlp", {0: [1.0, 2.0]})
        mixin.register_steering_modules({"m": {"vectors": base}}, replace=True)
        inline_prefill = _spec("post_mlp", {1: [5.0, 5.0]})
        sp = SamplingParams(
            steering_module_ref=("m", 1.0),
            prefill_steering_vectors=inline_prefill,
        )

        result = mixin._resolve_request_steering(sp, "prefill")
        assert result is not None
        # Layer 0 from base, layer 1 from inline_prefill.
        assert result["post_mlp"][0].tolist() == [1.0, 2.0]
        assert result["post_mlp"][1].tolist() == [5.0, 5.0]


# ---------------------------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------------------------


class TestCacheLifecycle:
    def test_register_replace_clears_cache(self):
        mixin = _StubMixin()
        mixin.register_steering_modules(
            {"a": {"vectors": _spec("post_mlp", {0: [1.0]})}}, replace=True
        )
        assert "a" in mixin._steering_module_resolved_cache

        mixin.register_steering_modules(
            {"b": {"vectors": _spec("post_mlp", {0: [2.0]})}}, replace=True
        )
        assert "a" not in mixin._steering_module_resolved_cache
        assert "b" in mixin._steering_module_resolved_cache

    def test_unregister_drops_cache_entry(self):
        mixin = _StubMixin()
        mixin.register_steering_modules(
            {
                "a": {"vectors": _spec("post_mlp", {0: [1.0]})},
                "b": {"vectors": _spec("post_mlp", {0: [2.0]})},
            },
            replace=True,
        )
        mixin.unregister_steering_modules(["a"])
        assert "a" not in mixin._steering_module_resolved_cache
        assert "b" in mixin._steering_module_resolved_cache

    def test_unknown_name_raises(self):
        mixin = _StubMixin()
        sp = SamplingParams(steering_module_ref=("missing", 1.0))
        with pytest.raises(RuntimeError, match="not registered"):
            mixin._resolve_request_steering(sp, "prefill")
