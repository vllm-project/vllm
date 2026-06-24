# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu_worker import _subtract_model_memory_reserve


def test_model_memory_reserve_noop():
    assert _subtract_model_memory_reserve(1024, 0) == 1024


def test_model_memory_reserve_subtracts_from_available_kv_memory():
    assert _subtract_model_memory_reserve(4096, 1024) == 3072


def test_model_memory_reserve_rejects_negative_reserve():
    with pytest.raises(ValueError, match="non-negative"):
        _subtract_model_memory_reserve(4096, -1)


def test_model_memory_reserve_rejects_exhausting_kv_memory():
    with pytest.raises(ValueError, match="leaves no available KV cache"):
        _subtract_model_memory_reserve(1024, 1024)
