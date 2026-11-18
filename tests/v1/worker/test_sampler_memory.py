# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.sample.sampler_memory import (
    check_room_for_sampler_state,
    penalty_state_bytes,
)

# Qwen2.5 vocab and the default max_num_seqs from the report in #68316.
_MAX_NUM_REQS = 256
_VOCAB_SIZE = 152064


def test_penalty_state_bytes_matches_penalties_state_alloc():
    bin_counts = _MAX_NUM_REQS * _VOCAB_SIZE * 4
    prompt_mask = _MAX_NUM_REQS * cdiv(_VOCAB_SIZE, 32) * 4
    assert penalty_state_bytes(_MAX_NUM_REQS, _VOCAB_SIZE) == bin_counts + prompt_mask


def test_check_room_for_sampler_state_rejects_a_full_device():
    penalty_bytes = penalty_state_bytes(_MAX_NUM_REQS, _VOCAB_SIZE)
    # The report had ~93 MiB free and a ~150 MiB bin-count allocation.
    free_bytes = 94 * (1 << 20)
    assert free_bytes < penalty_bytes

    with pytest.raises(ValueError, match="Not enough GPU memory to finish loading"):
        check_room_for_sampler_state(
            model_memory_bytes=14 * (1 << 30) + 300 * (1 << 20),
            free_bytes=free_bytes,
            penalty_bytes=penalty_bytes,
        )


def test_check_room_for_sampler_state_allows_enough_free_memory():
    penalty_bytes = penalty_state_bytes(_MAX_NUM_REQS, _VOCAB_SIZE)
    check_room_for_sampler_state(
        model_memory_bytes=8 * (1 << 30),
        free_bytes=penalty_bytes,
        penalty_bytes=penalty_bytes,
    )
