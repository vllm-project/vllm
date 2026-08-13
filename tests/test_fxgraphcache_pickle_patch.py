# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FxGraphCachePickler.dumps ValueError patch in env_override.py.

Validates that _apply_fxgraphcache_pickle_patch correctly wraps a pickler's
dumps method to convert ValueError into a bypass exception, without affecting
other exception types or normal return values.
"""

import pytest

from vllm.env_override import _apply_fxgraphcache_pickle_patch


class _BypassStub(Exception):
    """Stand-in for BypassFxGraphCache in unit tests."""


class TestApplyFxgraphcachePicklePatch:
    def test_valueerror_converted_to_bypass(self):
        class Pickler:
            def dumps(self, obj):
                raise ValueError("can't serialize blocked layout")

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        with pytest.raises(_BypassStub, match="Failed to pickle cache key"):
            Pickler().dumps(object())

    def test_original_valueerror_chained(self):
        class Pickler:
            def dumps(self, obj):
                raise ValueError("bad tensor layout")

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        with pytest.raises(_BypassStub) as exc_info:
            Pickler().dumps(object())

        cause = exc_info.value.__cause__
        assert isinstance(cause, ValueError)
        assert str(cause) == "bad tensor layout"

    def test_non_valueerror_propagates(self):
        class Pickler:
            def dumps(self, obj):
                raise TypeError("unexpected type")

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        with pytest.raises(TypeError, match="unexpected type"):
            Pickler().dumps(object())

    def test_normal_return_preserved(self):
        sentinel = b"serialized-graph-key"

        class Pickler:
            def dumps(self, obj):
                return sentinel

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        assert Pickler().dumps(object()) is sentinel

    def test_idempotent(self):
        class Pickler:
            def dumps(self, obj):
                return b"ok"

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)
        first_dumps = Pickler.dumps
        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        assert Pickler.dumps is first_dumps

    def test_sentinel_attribute_set(self):
        class Pickler:
            def dumps(self, obj):
                return b"ok"

        assert not hasattr(Pickler.dumps, "_vllm_patched")
        assert not getattr(Pickler, "_vllm_fxgraph_dumps_patched", False)

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        assert Pickler.dumps._vllm_patched is True  # type: ignore[attr-defined]
        assert Pickler._vllm_fxgraph_dumps_patched is True  # type: ignore[attr-defined]


def test_patch_applied_in_current_environment():
    """Integration: verify patch state matches current torch version."""
    from torch._inductor.codecache import FxGraphCachePickler

    from vllm.utils.torch_utils import is_torch_equal_or_newer

    should_be_patched = is_torch_equal_or_newer(
        "2.10.0"
    ) and not is_torch_equal_or_newer("2.11.0")

    assert getattr(FxGraphCachePickler, "_vllm_fxgraph_dumps_patched", False) == (
        should_be_patched
    )
    assert hasattr(FxGraphCachePickler.dumps, "_vllm_patched") == should_be_patched
