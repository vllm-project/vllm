# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-level tests for AsyncLLM.evict_session_token_range.

Round-trip end-to-end behaviour (stream -> evict -> resume against a real
model) lives in CI integration suites. These tests cover the experimental
guard, request-state preflight, and the engine-core RPC wiring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.scheduler import _reindex_mm_features
from vllm.v1.engine.async_llm import AsyncLLM


@pytest.fixture
def mock_async_llm():
    """Minimal AsyncLLM mock with engine_core stubbed out."""
    llm = MagicMock(spec=AsyncLLM)
    llm.engine_core = MagicMock()
    llm.engine_core.evict_token_range_async = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_evict_session_token_range_requires_env_var(mock_async_llm):
    """Without VLLM_ENABLE_EXPERIMENTAL_SESSION_EVICTION, the API must
    refuse to call into engine_core."""
    with (
        patch("vllm.envs.VLLM_ENABLE_EXPERIMENTAL_SESSION_EVICTION", False),
        pytest.raises(RuntimeError, match="experimental"),
    ):
        await AsyncLLM.evict_session_token_range(
            mock_async_llm,
            request_id="req-1",
            num_tokens_to_evict=4,
            num_sink_tokens=2,
        )
    mock_async_llm.engine_core.evict_token_range_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_evict_session_token_range_passes_args_when_enabled(
    mock_async_llm,
):
    """With the env var set, the API forwards (request_id,
    num_tokens_to_evict, num_sink_tokens) to engine_core."""
    with patch("vllm.envs.VLLM_ENABLE_EXPERIMENTAL_SESSION_EVICTION", True):
        await AsyncLLM.evict_session_token_range(
            mock_async_llm,
            request_id="req-1",
            num_tokens_to_evict=4,
            num_sink_tokens=2,
        )
    mock_async_llm.engine_core.evict_token_range_async.assert_awaited_once_with(
        "req-1", 4, 2
    )


# ----------------------------------------------------------------------
# _reindex_mm_features unit tests
# ----------------------------------------------------------------------


def _mm(offset: int, length: int, identifier: str = "x") -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def test_reindex_mm_features_fully_before_cut_unchanged():
    """A mm item that ends at or before the eviction start keeps its offset."""
    features = [_mm(offset=0, length=10, identifier="sink_img")]
    out = _reindex_mm_features(features, evict_start=20, evict_end=40)
    assert len(out) == 1
    assert out[0].identifier == "sink_img"
    assert out[0].mm_position.offset == 0
    assert out[0].mm_position.length == 10


def test_reindex_mm_features_fully_after_cut_shifted():
    """A mm item that starts at or after the eviction end shifts back."""
    features = [_mm(offset=50, length=10, identifier="tail_img")]
    out = _reindex_mm_features(features, evict_start=20, evict_end=40)
    assert len(out) == 1
    assert out[0].mm_position.offset == 30  # 50 - (40 - 20)
    assert out[0].mm_position.length == 10


def test_reindex_mm_features_fully_inside_cut_dropped():
    """A mm item entirely within the eviction range is removed."""
    features = [_mm(offset=22, length=10, identifier="evicted_img")]
    out = _reindex_mm_features(features, evict_start=20, evict_end=40)
    assert out == []


def test_reindex_mm_features_partial_overlap_raises():
    """An mm item that straddles the eviction boundary raises ValueError."""
    # Item [30, 50) overlaps evict [20, 40) at the trailing edge.
    features = [_mm(offset=30, length=20, identifier="bad_img")]
    with pytest.raises(ValueError, match="partially overlaps"):
        _reindex_mm_features(features, evict_start=20, evict_end=40)


def test_reindex_mm_features_multi_image_middle_evicted():
    """Sink image + middle frame (evicted) + trailing image:
    the trailing image's offset must shift back by exactly the
    eviction width so M-RoPE rebuilds with the correct grid offsets
    on the next streaming resume."""
    features = [
        _mm(offset=0, length=10, identifier="sink_img"),  # before cut
        _mm(offset=20, length=15, identifier="evicted_img"),  # inside cut [20,35)
        _mm(offset=50, length=8, identifier="tail_img"),  # after cut
    ]
    out = _reindex_mm_features(features, evict_start=20, evict_end=35)
    assert [f.identifier for f in out] == ["sink_img", "tail_img"]
    assert out[0].mm_position.offset == 0  # sink unchanged
    assert out[1].mm_position.offset == 35  # 50 - (35 - 20)
    assert out[1].mm_position.length == 8


def test_reindex_mm_features_empty_input():
    """Reindex on an empty mm_features list is a no-op."""
    assert _reindex_mm_features([], evict_start=10, evict_end=20) == []
