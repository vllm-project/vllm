# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ParallelConfig
from vllm.platforms import current_platform


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("num_tokens", [999, 1000, 1001, 1002])
def test_dense_sp_threshold(monkeypatch, enabled, num_tokens):
    monkeypatch.setattr(current_platform, "device_count", lambda: 2)
    config = ParallelConfig(
        tensor_parallel_size=2,
        is_moe_model=False,
        enable_sequence_parallel=enabled,
    )
    assert config.sequence_parallel_enabled_for_tokens(num_tokens) is (
        enabled is True and num_tokens > 1000
    )


def test_sp_validation_after_model_type_is_known(monkeypatch):
    monkeypatch.setattr(current_platform, "device_count", lambda: 2)
    config = ParallelConfig(tensor_parallel_size=2, enable_sequence_parallel=True)
    config.is_moe_model = True
    with pytest.raises(ValueError, match="MoE sequence parallelism requires"):
        config.validate_sequence_parallel()


def test_pipeline_parallel_disables_sp(monkeypatch):
    monkeypatch.setattr(current_platform, "device_count", lambda: 4)
    config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        is_moe_model=False,
        enable_sequence_parallel=True,
    )
    assert not config.sequence_parallel_enabled_for_tokens(1001)
