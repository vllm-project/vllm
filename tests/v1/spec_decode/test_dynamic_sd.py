# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD batch-size schedule manager."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config.speculative import DynamicSpeculativeConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.manager import DynamicSpeculativeDecodingManager
from vllm.v1.structured_output import StructuredOutputManager


def _make_manager(
    schedule: dict[str, int],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> DynamicSpeculativeDecodingManager:
    config = DynamicSpeculativeConfig(num_speculative_tokens_per_batch_size=schedule)
    return DynamicSpeculativeDecodingManager(
        config,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: dict[str, int],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler, num_requests: int, *, num_tokens: int = 10
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        scheduler.add_request(request)
    return scheduler.schedule()


def test_dynamic_sd_uses_batch_size_schedule():
    manager = _make_manager(
        {
            "1-16": 3,
            "32-128": 2,
            "256-2048": 0,
        }
    )

    assert manager.get_optimal_num_speculative_tokens(1) == 3
    assert manager.get_optimal_num_speculative_tokens(16) == 3
    assert manager.get_optimal_num_speculative_tokens(17) == 3
    assert manager.get_optimal_num_speculative_tokens(31) == 3
    assert manager.get_optimal_num_speculative_tokens(32) == 2
    assert manager.get_optimal_num_speculative_tokens(128) == 2
    assert manager.get_optimal_num_speculative_tokens(129) == 2
    assert manager.get_optimal_num_speculative_tokens(255) == 2
    assert manager.step(256) == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_manager({"2-16": 3})


def test_dynamic_sd_clamps_k_to_runtime_max():
    manager = _make_manager(
        {"1-256": 4},
        runtime_num_speculative_tokens=3,
    )

    assert manager.get_optimal_num_speculative_tokens(1) == 3
    assert manager.get_optimal_num_speculative_tokens(256) == 3


def test_dynamic_sd_rejects_invalid_range_string():
    with pytest.raises(ValueError, match="Expected 'N' or 'N-M'"):
        _make_manager({"foo": 3})


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_manager({"1-16": 3, "16-32": 2})


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_manager({"1-16": -1})


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_manager({})


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        DynamicSpeculativeDecodingManager(
            DynamicSpeculativeConfig(),
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_rejects_invalid_batch_size_queries():
    manager = _make_manager({"1-256": 3})

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        manager.get_optimal_num_speculative_tokens(0)
    with pytest.raises(ValueError, match="batch_size must be <= vllm_max_batch_size"):
        manager.get_optimal_num_speculative_tokens(257)


def test_scheduler_initializes_dynamic_sd_manager_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        {"1-16": 3, "64-128": 2, "256-4096": 0},
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_manager is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            {"1-16": 3, "64-128": 2, "256-4096": 0},
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        {"1-256": 5},
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler.dynamic_sd_manager is None
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        {"1-16": 3, "64-128": 2, "256-4096": 0},
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd({"2-16": 3})


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        {"1-16": 3, "64-128": 2, "256-4096": 0},
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_manager is not None
    assert scheduler.dynamic_sd_manager.vllm_max_batch_size == 16
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3
