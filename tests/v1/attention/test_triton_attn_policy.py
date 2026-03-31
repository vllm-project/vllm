# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.triton_attn import _get_seq_threshold_3d


def test_gpt_oss_sm89_keeps_default_seq_threshold_3d():
    assert (
        _get_seq_threshold_3d(
            16,
            device_capability=DeviceCapability(8, 9),
            model_type="gpt_oss",
            head_size=64,
            num_heads_kv=8,
            sliding_window=128,
        )
        == 16
    )


def test_non_gpt_oss_keeps_default_seq_threshold_3d():
    assert (
        _get_seq_threshold_3d(
            16,
            device_capability=DeviceCapability(8, 9),
            model_type="llama",
            head_size=64,
            num_heads_kv=8,
            sliding_window=128,
        )
        == 16
    )


def test_non_target_geometry_keeps_default_seq_threshold_3d():
    assert (
        _get_seq_threshold_3d(
            16,
            device_capability=DeviceCapability(8, 9),
            model_type="gpt_oss",
            head_size=64,
            num_heads_kv=8,
            sliding_window=0,
        )
        == 16
    )
