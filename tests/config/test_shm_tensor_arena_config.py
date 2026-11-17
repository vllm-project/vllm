# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import ParallelConfig


def test_slot_mb_below_arena_minimum_is_rejected():
    with pytest.raises(ValueError, match="minimum tensor size"):
        ParallelConfig(enable_shm_tensor_arena=True, shm_tensor_arena_slot_mb=4)


@pytest.mark.parametrize(
    "field", ["shm_tensor_arena_slots", "shm_tensor_arena_slot_mb"]
)
def test_zero_arena_size_is_rejected(field):
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ParallelConfig(**{field: 0})


def test_arena_size_flags_without_arena_warn(caplog_vllm, disable_log_dedup):
    with caplog_vllm.at_level("WARNING"):
        ParallelConfig(shm_tensor_arena_slots=16)
    assert "enable_shm_tensor_arena is off" in caplog_vllm.text


def test_slot_mb_below_arena_minimum_is_not_rejected_with_the_arena_off():
    """The size has no effect with the arena off, so it only warns."""
    config = ParallelConfig(shm_tensor_arena_slot_mb=4)
    assert config.shm_tensor_arena_slot_mb == 4


def test_arena_defaults_do_not_warn(caplog_vllm, disable_log_dedup):
    with caplog_vllm.at_level("WARNING"):
        ParallelConfig()
        ParallelConfig(
            enable_shm_tensor_arena=True,
            shm_tensor_arena_slots=128,
            shm_tensor_arena_slot_mb=48,
        )
    assert "enable_shm_tensor_arena is off" not in caplog_vllm.text
