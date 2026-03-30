# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.triton_unified_attention import (
    _get_tile_size,
    _use_small_sm8x_sink_tiles,
)


def test_sm89_sink_prefill_uses_smaller_tile():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=0,
            element_size=2,
            is_prefill=True,
            has_sinks=True,
            device_capability=DeviceCapability(8, 9),
        )
        == 16
    )


def test_sm80_sink_prefill_keeps_default_tile():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=0,
            element_size=2,
            is_prefill=True,
            has_sinks=True,
            device_capability=DeviceCapability(8, 0),
        )
        == 32
    )


def test_sink_decode_policy_is_unchanged():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=0,
            element_size=2,
            is_prefill=False,
            has_sinks=True,
            device_capability=DeviceCapability(8, 9),
        )
        == 16
    )


def test_gemma3_special_case_wins_over_sm89_sink_policy():
    assert (
        _get_tile_size(
            head_size=128,
            sliding_window=1024,
            element_size=2,
            is_prefill=True,
            has_sinks=True,
            device_capability=DeviceCapability(8, 9),
        )
        == 32
    )


def test_sm89_sink_detection_uses_current_platform():
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=DeviceCapability(8, 9),
        ),
    ):
        assert _use_small_sm8x_sink_tiles(
            current_platform.get_device_capability(),
            has_sinks=True,
        )
