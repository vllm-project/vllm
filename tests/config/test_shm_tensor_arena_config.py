# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import ParallelConfig


def test_slot_mb_below_arena_minimum_is_rejected():
    with pytest.raises(ValueError, match="minimum tensor size"):
        ParallelConfig(enable_shm_tensor_arena=True, shm_tensor_arena_slot_mb=4)


def test_arena_size_flags_without_arena_warn(caplog_vllm):
    with caplog_vllm.at_level("WARNING"):
        ParallelConfig(shm_tensor_arena_slots=16)
    assert "enable_shm_tensor_arena is off" in caplog_vllm.text


def test_arena_defaults_do_not_warn(caplog_vllm):
    with caplog_vllm.at_level("WARNING"):
        ParallelConfig()
        ParallelConfig(
            enable_shm_tensor_arena=True,
            shm_tensor_arena_slots=128,
            shm_tensor_arena_slot_mb=48,
        )
    assert "enable_shm_tensor_arena is off" not in caplog_vllm.text
