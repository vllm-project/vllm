# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup


def make_scheduler():
    scheduler = Mock()
    scheduler.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
        [(1, 4, 15), (5, 64, 3)],
        vllm_max_batch_size=64,
        vllm_num_speculative_tokens=15,
    )
    return scheduler


def test_dynamic_k_lookup_uses_batch_size():
    scheduler = make_scheduler()
    assert scheduler.dynamic_sd_lookup[4] == 15
    assert scheduler.dynamic_sd_lookup[5] == 3


def test_allocate_slots_mock_records_dynamic_override():
    scheduler = make_scheduler()
    allocate_slots = Mock(return_value=None)
    allocate_slots(num_spec_override=scheduler.dynamic_sd_lookup[5])
    assert allocate_slots.call_args.kwargs["num_spec_override"] == 3


def test_step_projection_crosses_threshold_once():
    scheduler = object.__new__(Scheduler)
    scheduler.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
        [(1, 4, 15), (5, 64, 3)], 64, 15
    )
    scheduler.num_spec_tokens = 15
    scheduler.max_num_running_reqs = 64
    scheduler.running = [object()] * 4
    scheduler.waiting = [object()]

    assert scheduler._get_step_spec_tokens() == 3


def test_step_projection_uses_startup_value_without_table():
    scheduler = object.__new__(Scheduler)
    scheduler.dynamic_sd_lookup = None
    scheduler.num_spec_tokens = 15
    scheduler.max_num_running_reqs = 64
    scheduler.running = []
    scheduler.waiting = []

    assert scheduler._get_step_spec_tokens() == 15


def test_dflash_lookahead_tracks_step_k():
    scheduler = object.__new__(Scheduler)
    scheduler.dynamic_sd_lookup = [0, 15, 3]
    scheduler.num_spec_tokens = 15
    scheduler.num_lookahead_tokens = 16
    assert scheduler._get_step_lookahead_tokens(3) == 4


def test_lookahead_is_unchanged_without_dynamic_schedule():
    scheduler = object.__new__(Scheduler)
    scheduler.dynamic_sd_lookup = None
    scheduler.num_spec_tokens = 15
    scheduler.num_lookahead_tokens = 16
    assert scheduler._get_step_lookahead_tokens(3) == 16
