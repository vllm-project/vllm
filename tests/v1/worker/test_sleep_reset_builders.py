# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.worker.gpu_worker import Worker


@pytest.fixture
def mock_worker():
    worker = object.__new__(Worker)
    worker._sleep_saved_buffers = {}
    worker.cache_config = MagicMock()
    worker.cache_config.cache_dtype = "auto"
    worker.model_runner = MagicMock()
    del worker.model_runner.attn_groups
    return worker


def _make_group(builders):
    group = MagicMock()
    group.metadata_builders = builders
    return group


def _wake_up(worker, tags=None):
    mock_allocator = MagicMock()
    with (
        patch(
            "vllm.v1.worker.gpu_worker.CuMemAllocator",
            create=True,
        ),
        patch(
            "vllm.device_allocator.cumem.CuMemAllocator",
        ) as mock_cls,
    ):
        mock_cls.get_instance.return_value = mock_allocator
        worker.wake_up(tags)


def test_iter_builders_no_attn_groups(mock_worker):
    assert list(mock_worker._iter_attn_metadata_builders()) == []


def test_iter_builders_multiple_groups(mock_worker):
    b1, b2, b3 = MagicMock(), MagicMock(), MagicMock()
    mock_worker.model_runner.attn_groups = [
        [_make_group([b1, b2])],
        [_make_group([b3])],
    ]

    assert list(mock_worker._iter_attn_metadata_builders()) == [b1, b2, b3]


@pytest.mark.parametrize("tags", [None, ["kv_cache"]])
def test_wake_up_resets_builders(mock_worker, tags):
    builder = MagicMock(spec=["reset_for_sleep_mode"])
    mock_worker.model_runner.attn_groups = [[_make_group([builder])]]

    _wake_up(mock_worker, tags=tags)
    builder.reset_for_sleep_mode.assert_called_once()


def test_wake_up_skips_reset_on_weights_only_tag(mock_worker):
    builder = MagicMock(spec=["reset_for_sleep_mode"])
    mock_worker.model_runner.attn_groups = [[_make_group([builder])]]

    _wake_up(mock_worker, tags=["weights"])
    builder.reset_for_sleep_mode.assert_not_called()


def test_wake_up_resets_multiple_builders(mock_worker):
    b1 = MagicMock(spec=["reset_for_sleep_mode"])
    b2 = MagicMock(spec=["reset_for_sleep_mode"])
    b_no_reset = MagicMock(spec=[])
    mock_worker.model_runner.attn_groups = [
        [_make_group([b1, b_no_reset])],
        [_make_group([b2])],
    ]

    _wake_up(mock_worker, tags=["kv_cache"])
    b1.reset_for_sleep_mode.assert_called_once()
    b2.reset_for_sleep_mode.assert_called_once()
