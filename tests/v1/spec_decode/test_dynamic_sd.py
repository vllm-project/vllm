# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD schedule helpers."""

import logging

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager


def _make_lookup(
    num_speculative_tokens_per_batch_size,
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    """Build a lookup and flatten it to a 1D ``[bs] -> K`` list.

    Only valid for schedules that produce a single context bucket (i.e. legacy
    3-item entries). New 5-item tests call ``build_dynamic_sd_schedule_lookup``
    directly to see the full 2D structure.
    """
    lookup = build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )
    assert len(lookup.ctx_boundaries) == 1
    return [row[0] for row in lookup.dense]


def _make_scheduler_with_dynamic_sd(
    schedule,
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
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        build_dynamic_sd_schedule_lookup(
            None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_lookup_rejects_invalid_batch_size_queries():
    dynamic_sd_lookup = _make_lookup([(1, 256, 3)])

    assert dynamic_sd_lookup[0] == 0
    with pytest.raises(IndexError):
        _ = dynamic_sd_lookup[257]


def test_dynamic_sd_5tuple_forms_2d_dense_lookup():
    lookup = build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=[
            (1, 32, 1, 512, 3),
            (1, 32, 513, 4096, 2),
            (33, 128, 1, 512, 2),
            (33, 128, 513, 4096, 1),
        ],
        vllm_max_batch_size=128,
        vllm_num_speculative_tokens=3,
    )

    assert len(lookup.ctx_boundaries) == 2
    assert lookup.ctx_boundaries[0] == (1, 512)
    assert lookup.ctx_boundaries[1][0] == 513
    assert lookup.dense[1] == [3, 2]
    assert lookup.dense[32] == [3, 2]
    assert lookup.dense[33] == [2, 1]
    assert lookup.dense[128] == [2, 1]


def test_dynamic_sd_5tuple_carry_forward_across_bs_gap():
    lookup = build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=[
            (1, 8, 1, 100, 3),
            (1, 8, 101, 4096, 2),
            (32, 64, 1, 100, 1),
            (32, 64, 101, 4096, 0),
        ],
        vllm_max_batch_size=64,
        vllm_num_speculative_tokens=3,
    )

    assert lookup.dense[8] == [3, 2]
    assert lookup.dense[15] == [3, 2]
    assert lookup.dense[31] == [3, 2]
    assert lookup.dense[32] == [1, 0]
    assert lookup.dense[64] == [1, 0]


def test_dynamic_sd_bucket_of_maps_context_to_bucket_index():
    lookup = build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=[
            (1, 16, 1, 100, 3),
            (1, 16, 101, 4096, 2),
        ],
        vllm_max_batch_size=16,
        vllm_num_speculative_tokens=3,
    )

    assert lookup.bucket_of(0) == 0
    assert lookup.bucket_of(1) == 0
    assert lookup.bucket_of(100) == 0
    assert lookup.bucket_of(101) == 1
    assert lookup.bucket_of(10_000_000) == 1


def test_dynamic_sd_5tuple_extends_last_ctx_to_cover_any_value():
    lookup = build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=[(1, 16, 1, 4096, 3)],
        vllm_max_batch_size=16,
        vllm_num_speculative_tokens=3,
    )

    assert len(lookup.ctx_boundaries) == 1
    assert lookup.ctx_boundaries[0][0] == 1
    assert lookup.bucket_of(1) == 0
    assert lookup.bucket_of(10_000_000) == 0


def test_dynamic_sd_rejects_mixed_arity():
    with pytest.raises(ValueError, match="all be 3-item or all 5-item"):
        _make_lookup([(1, 16, 3), (17, 32, 1, 100, 2)])


def test_dynamic_sd_rejects_non_rectangular_grid():
    with pytest.raises(ValueError, match="rectangular"):
        build_dynamic_sd_schedule_lookup(
            num_speculative_tokens_per_batch_size=[
                (1, 64, 1, 4096, 3),
                (65, 256, 1, 512, 2),
                (65, 256, 513, 4096, 1),
            ],
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_rejects_ctx_gap():
    with pytest.raises(ValueError, match="contiguous"):
        build_dynamic_sd_schedule_lookup(
            num_speculative_tokens_per_batch_size=[
                (1, 16, 1, 100, 3),
                (1, 16, 200, 4096, 2),
            ],
            vllm_max_batch_size=16,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_rejects_ctx_not_starting_at_1():
    with pytest.raises(ValueError, match="first context range must start at 1"):
        build_dynamic_sd_schedule_lookup(
            num_speculative_tokens_per_batch_size=[(1, 16, 2, 100, 3)],
            vllm_max_batch_size=16,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_rejects_negative_ctx():
    with pytest.raises(ValueError, match="Context range .* must be positive"):
        build_dynamic_sd_schedule_lookup(
            num_speculative_tokens_per_batch_size=[(1, 16, 0, 100, 3)],
            vllm_max_batch_size=16,
            vllm_num_speculative_tokens=3,
        )


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
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

    assert scheduler.dynamic_sd_lookup is None
    assert output.num_spec_tokens_to_schedule == 3


def test_dynamic_sd_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=256,
            max_num_batched_tokens=2560,
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[
                (1, 16, 3),
                (64, 128, 2),
                (256, 4096, 0),
            ],
            data_parallel_size=2,
        )

    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens_per_batch_size is None
    assert scheduler.dynamic_sd_lookup is None
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )

    output = _add_requests_and_schedule(scheduler, 256)
    assert len(output.num_scheduled_tokens) == 256
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup.dense) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_stores_dynamic_sd_lookup_from_5tuple_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 1, 100, 3), (1, 16, 101, 4096, 2)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup.ctx_boundaries) == 2
    assert scheduler.dynamic_sd_lookup.dense[1] == [3, 2]


def test_scheduler_dispatches_k_via_ctx_bucket_for_fresh_requests():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 8, 1, 100, 3), (1, 8, 101, 8192, 1)],
        max_num_seqs=8,
        max_num_batched_tokens=200,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(
        scheduler, num_requests=4, num_tokens=20
    )

    assert len(output.num_scheduled_tokens) == 4
    assert output.num_spec_tokens_to_schedule == 3
