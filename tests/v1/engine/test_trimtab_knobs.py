# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live scheduler knobs and the validation that guards a warm reinit.

These exercise the pure decision logic: which values are accepted, what the scheduler is left holding, and what a
read-back reports. Nothing here allocates a KV cache or touches a GPU, so it runs in the default CI lane; the
reinit path itself is covered by the hardware bench in the PR description.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.engine.core import EngineCore, _trimtab_validate_reinit


class _Scheduler:
    """Enough of a scheduler for the knob code: the two caps it writes and the running list it must respect."""

    def __init__(self, max_num_seqs: int = 64, max_num_batched_tokens: int = 8192, running: int = 0):
        self.max_num_running_reqs = max_num_seqs
        self.max_num_scheduled_tokens = max_num_batched_tokens
        self.running = list(range(running))

        class _Cfg:
            long_prefill_token_threshold = 0

        self.scheduler_config = _Cfg()


class _Core:
    """EngineCore's knob methods bound to a bare object, so no engine has to be started."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        # the read-back reports the block count alongside the knobs
        self.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=1024))

    trimtab_set_knobs = EngineCore.trimtab_set_knobs
    trimtab_get_knobs = EngineCore.trimtab_get_knobs


def test_knobs_apply_to_the_live_scheduler():
    core = _Core(_Scheduler())
    out = core.trimtab_set_knobs({"max_num_seqs": 8, "max_num_batched_tokens": 2048})
    assert out["ok"] and out["rejected"] == {}
    assert core.scheduler.max_num_running_reqs == 8
    assert core.scheduler.max_num_scheduled_tokens == 2048
    assert core.trimtab_get_knobs()["max_num_seqs"] == 8


@pytest.mark.parametrize(
    "knobs",
    [
        {"max_num_seqs": 0},                       # would admit nothing
        {"max_num_seqs": -1},
        {"max_num_batched_tokens": 0},             # would give every step a zero budget
        {"max_num_seqs": 65},                      # above the boot allocation
        {"max_num_seqs": "many"},
        {"long_prefill_token_threshold": -1},
        {"log_level": "LOUD"},
        {"no_such_knob": 1},
    ],
)
def test_bad_values_are_rejected_and_change_nothing(knobs):
    core = _Core(_Scheduler())
    before = (core.scheduler.max_num_running_reqs, core.scheduler.max_num_scheduled_tokens)
    out = core.trimtab_set_knobs(knobs)
    assert not out["ok"] and out["rejected"]
    assert (core.scheduler.max_num_running_reqs, core.scheduler.max_num_scheduled_tokens) == before


def test_lowering_below_occupancy_stops_admission_without_breaking_the_step_assertion():
    """The scheduler asserts len(running) <= cap every step, so the cap cannot drop below what is already running.

    The target is recorded, the enforced cap stays at the occupancy, and the read-back reports the target so a
    caller can see the difference between what it asked for and what is in force."""
    core = _Core(_Scheduler(max_num_seqs=64, running=20))
    out = core.trimtab_set_knobs({"max_num_seqs": 4})
    assert out["ok"]
    assert core.scheduler.max_num_running_reqs == 20        # unchanged while those 20 are in flight
    assert core.scheduler._trimtab_pending_max_num_seqs == 4
    assert core.trimtab_get_knobs()["max_num_seqs"] == 4


def test_log_level_is_case_insensitive_and_recorded():
    core = _Core(_Scheduler())
    assert core.trimtab_set_knobs({"log_level": "debug"})["applied"]["log_level"] == "DEBUG"


@pytest.mark.parametrize(
    "fields,bad",
    [
        ({"max_num_seqs": 0}, "max_num_seqs"),
        ({"max_num_seqs": 1.5}, "max_num_seqs"),
        ({"max_num_seqs": True}, "max_num_seqs"),
        ({"max_num_batched_tokens": -8}, "max_num_batched_tokens"),
        ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 1.5}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": float("nan")}, "gpu_memory_utilization"),
    ],
)
def test_reinit_validation_refuses_before_anything_is_released(fields, bad):
    """A bad value caught during the rebuild would leave the engine with no KV cache, so it is caught first."""
    assert bad in _trimtab_validate_reinit(fields)


def test_reinit_validation_accepts_a_reasonable_request():
    assert _trimtab_validate_reinit({"gpu_memory_utilization": 0.85, "max_num_seqs": 32}) == {}
