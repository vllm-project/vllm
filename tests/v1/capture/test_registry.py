# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the capture-consumer entry-point registry."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.capture import CaptureConsumer, UnknownCaptureConsumerError
from vllm.v1.capture.registry import (
    ENTRY_POINT_GROUP,
    _reset_cache_for_testing,
    build_consumer,
    load_consumer_class,
)


class _FakeConsumer(CaptureConsumer):
    location = "worker"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self.vllm_config = vllm_config
        self.params = params

    def on_capture(self, key, tensor, sidecar):  # type: ignore[override]
        pass


class _FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(self, name: str, cls: type[CaptureConsumer]) -> None:
        self.name = name
        self._cls = cls

    def load(self) -> type[CaptureConsumer]:
        return self._cls


class _FakeEntryPointSet(list):
    """Matches the iterable contract ``entry_points(group=...)`` returns."""


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    _reset_cache_for_testing()
    yield
    _reset_cache_for_testing()


def _patch_entry_points(entries: list[_FakeEntryPoint]):
    """Return a patch object for ``importlib.metadata.entry_points``.

    The patched callable tracks how many times it was invoked; tests
    can assert caching by inspecting ``mock.call_count``.
    """

    def fake_entry_points(*, group: str):
        assert group == ENTRY_POINT_GROUP, (
            f"expected group {ENTRY_POINT_GROUP!r}, got {group!r}"
        )
        return _FakeEntryPointSet(entries)

    return patch(
        "vllm.v1.capture.registry.importlib.metadata.entry_points",
        side_effect=fake_entry_points,
    )


def test_unknown_consumer_name_raises():
    with _patch_entry_points([]):
        with pytest.raises(UnknownCaptureConsumerError) as exc_info:
            load_consumer_class("nonexistent")
        assert "nonexistent" in str(exc_info.value)
        assert ENTRY_POINT_GROUP in str(exc_info.value)


def test_load_consumer_class_resolves_fake_entry_point():
    entries = [_FakeEntryPoint("fake", _FakeConsumer)]
    with _patch_entry_points(entries):
        cls = load_consumer_class("fake")
    assert cls is _FakeConsumer


def test_build_consumer_constructs_instance_with_params():
    entries = [_FakeEntryPoint("fake", _FakeConsumer)]
    vllm_config = MagicMock()
    params = {"k": "v"}

    with _patch_entry_points(entries):
        consumer = build_consumer("fake", vllm_config, params)

    assert isinstance(consumer, _FakeConsumer)
    assert consumer.vllm_config is vllm_config
    assert consumer.params == params


def test_entry_point_resolution_is_cached():
    entries = [_FakeEntryPoint("fake", _FakeConsumer)]
    with _patch_entry_points(entries) as mock_ep:
        load_consumer_class("fake")
        load_consumer_class("fake")
        load_consumer_class("fake")
    assert mock_ep.call_count == 1


def test_non_consumer_entry_point_raises_type_error():
    class _NotAConsumer:
        pass

    entries = [_FakeEntryPoint("bogus", _NotAConsumer)]  # type: ignore[arg-type]
    with _patch_entry_points(entries), pytest.raises(TypeError) as exc_info:
        load_consumer_class("bogus")
    assert "CaptureConsumer" in str(exc_info.value)


def test_registry_cache_reset_between_tests_allows_re_patching():
    # First patch: only ``alpha`` is registered.
    with _patch_entry_points([_FakeEntryPoint("alpha", _FakeConsumer)]):
        assert load_consumer_class("alpha") is _FakeConsumer
        with pytest.raises(UnknownCaptureConsumerError):
            load_consumer_class("beta")

    _reset_cache_for_testing()

    # Second patch: only ``beta`` is registered. The cache was reset,
    # so ``alpha`` is now unknown.
    with _patch_entry_points([_FakeEntryPoint("beta", _FakeConsumer)]):
        assert load_consumer_class("beta") is _FakeConsumer
        with pytest.raises(UnknownCaptureConsumerError):
            load_consumer_class("alpha")
