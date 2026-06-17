# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DefaultModelLoader's model_loader_extra_config validation.

Regression tests for https://github.com/vllm-project/vllm/issues/45088,
where string values (e.g. ``{"num_threads": "6"}``) were passed through
unvalidated and crashed deep inside ThreadPoolExecutor during engine
startup. String booleans (e.g. ``"False"``) were also silently truthy,
enabling multithreaded loading when the user intended to disable it.
"""

import pytest

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@pytest.mark.parametrize(
    "extra_config",
    [
        # Issue #45088: JSON string instead of boolean.
        {"enable_multithread_load": "True"},
        # Silently-truthy string: would have *enabled* multithread load.
        {"enable_multithread_load": "False"},
        {"enable_multithread_load": 1},
        # Issue #45088: JSON string instead of integer -> TypeError in
        # ThreadPoolExecutor(max_workers="6").
        {"enable_multithread_load": True, "num_threads": "6"},
        {"enable_multithread_load": True, "num_threads": 0},
        {"enable_multithread_load": True, "num_threads": -1},
        {"enable_multithread_load": True, "num_threads": 6.0},
        {"enable_multithread_load": True, "num_threads": True},
    ],
)
def test_extra_config_rejects_invalid_values(extra_config):
    with pytest.raises(ValueError, match="model_loader_extra_config"):
        DefaultModelLoader(LoadConfig(model_loader_extra_config=extra_config))


@pytest.mark.parametrize(
    "extra_config",
    [
        {},
        {"enable_multithread_load": True},
        {"enable_multithread_load": False},
        {"enable_multithread_load": True, "num_threads": 6},
        {"num_threads": 4},
    ],
)
def test_extra_config_accepts_valid_values(extra_config):
    loader = DefaultModelLoader(LoadConfig(model_loader_extra_config=extra_config))
    assert isinstance(loader.enable_multithread_load, bool)
    assert isinstance(loader.num_threads, int)
    assert loader.num_threads >= 1


def test_extra_config_defaults():
    loader = DefaultModelLoader(LoadConfig())
    assert loader.enable_multithread_load is False
    assert loader.num_threads == DefaultModelLoader.DEFAULT_NUM_THREADS


def test_extra_config_still_rejects_unexpected_keys():
    # Pre-existing key validation must keep working alongside the new
    # value validation.
    with pytest.raises(ValueError, match="Unexpected extra config keys"):
        DefaultModelLoader(
            LoadConfig(model_loader_extra_config={"enable_multithread": True})
        )
