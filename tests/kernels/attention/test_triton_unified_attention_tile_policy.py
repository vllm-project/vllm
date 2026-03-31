# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.triton_unified_attention import (
    TRITON_ATTN_TILE_SIZE_DECODE_ENV,
    TRITON_ATTN_TILE_SIZE_PREFILL_ENV,
    _get_tile_size,
    _get_tile_size_override,
    _is_gpt_oss_sm8x_sink_attention,
    _use_gpt_oss_sm8x_large_prefill_tile,
    _use_small_sm8x_sink_tiles,
)


def test_sm89_sink_prefill_uses_smaller_tile():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=0,
            num_kv_heads=4,
            element_size=2,
            is_prefill=True,
            has_sinks=True,
            device_capability=DeviceCapability(8, 9),
        )
        == 16
    )


def test_gpt_oss_sm89_sink_prefill_keeps_default_tile():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=128,
            num_kv_heads=8,
            element_size=2,
            is_prefill=True,
            has_sinks=True,
            device_capability=DeviceCapability(8, 9),
            max_seq_len=9000,
        )
        == 32
    )


def test_gpt_oss_sm89_long_prefill_uses_small_tile():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=128,
            num_kv_heads=8,
            element_size=2,
            is_prefill=True,
            has_sinks=True,
            device_capability=DeviceCapability(8, 9),
            max_seq_len=22000,
        )
        == 16
    )


def test_sm80_sink_prefill_keeps_default_tile():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=0,
            num_kv_heads=8,
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
            num_kv_heads=4,
            element_size=2,
            is_prefill=False,
            has_sinks=True,
            device_capability=DeviceCapability(8, 9),
        )
        == 16
    )


def test_gpt_oss_sm89_sink_decode_keeps_default_tile():
    assert (
        _get_tile_size(
            head_size=64,
            sliding_window=0,
            num_kv_heads=8,
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
            num_kv_heads=8,
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


def test_gpt_oss_sm89_sink_detection_uses_geometry():
    assert _is_gpt_oss_sm8x_sink_attention(
        head_size=64,
        num_kv_heads=8,
        device_capability=DeviceCapability(8, 9),
        has_sinks=True,
    )


def test_gpt_oss_sm89_large_prefill_tile_is_length_gated():
    assert _use_gpt_oss_sm8x_large_prefill_tile(
        head_size=64,
        num_kv_heads=8,
        device_capability=DeviceCapability(8, 9),
        has_sinks=True,
        max_seq_len=9000,
    )
    assert not _use_gpt_oss_sm8x_large_prefill_tile(
        head_size=64,
        num_kv_heads=8,
        device_capability=DeviceCapability(8, 9),
        has_sinks=True,
        max_seq_len=22000,
    )


def test_non_gpt_oss_sink_detection_rejects_other_kv_head_counts():
    assert not _is_gpt_oss_sm8x_sink_attention(
        head_size=64,
        num_kv_heads=4,
        device_capability=DeviceCapability(8, 9),
        has_sinks=True,
    )


def test_prefill_tile_size_env_override_wins():
    with patch.dict(
        "os.environ",
        {TRITON_ATTN_TILE_SIZE_PREFILL_ENV: "32"},
        clear=False,
    ):
        assert _get_tile_size_override(is_prefill=True) == 32
        assert (
            _get_tile_size(
                head_size=64,
                sliding_window=0,
                num_kv_heads=4,
                element_size=2,
                is_prefill=True,
                has_sinks=True,
                device_capability=DeviceCapability(8, 9),
            )
            == 32
        )


def test_decode_tile_size_env_override_wins():
    with patch.dict(
        "os.environ",
        {TRITON_ATTN_TILE_SIZE_DECODE_ENV: "32"},
        clear=False,
    ):
        assert _get_tile_size_override(is_prefill=False) == 32
        assert (
            _get_tile_size(
                head_size=64,
                sliding_window=0,
                num_kv_heads=4,
                element_size=2,
                is_prefill=False,
                has_sinks=True,
                device_capability=DeviceCapability(8, 9),
            )
            == 32
        )


def test_tile_size_env_override_requires_power_of_two():
    with patch.dict(
        "os.environ",
        {TRITON_ATTN_TILE_SIZE_PREFILL_ENV: "24"},
        clear=False,
    ):
        try:
            _get_tile_size_override(is_prefill=True)
        except ValueError as exc:
            assert TRITON_ATTN_TILE_SIZE_PREFILL_ENV in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid tile-size override")


def test_fp8_tile_size_override_requires_minimum_tile():
    with patch.dict(
        "os.environ",
        {TRITON_ATTN_TILE_SIZE_DECODE_ENV: "16"},
        clear=False,
    ):
        try:
            _get_tile_size(
                head_size=64,
                sliding_window=0,
                num_kv_heads=4,
                element_size=1,
                is_prefill=False,
                has_sinks=True,
                device_capability=DeviceCapability(8, 9),
            )
        except ValueError as exc:
            assert TRITON_ATTN_TILE_SIZE_DECODE_ENV in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid fp8 tile override")
