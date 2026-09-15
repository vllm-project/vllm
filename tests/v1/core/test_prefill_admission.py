# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.prefill_admission import PrefillAdmissionPolicy

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("max_num_seqs", "prefill_admission_slots", "expected"),
    [
        (128, 0, 128),
        (128, 25, 153),
        (4, 10, 8),
    ],
)
def test_scheduler_config_resident_capacity(
    max_num_seqs: int,
    prefill_admission_slots: int,
    expected: int,
):
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        prefill_admission_slots=prefill_admission_slots,
        max_model_len=2048,
        is_encoder_decoder=False,
    )

    assert config.max_num_resident_seqs == expected


def test_prefill_admission_changes_scheduler_hash():
    default_config = SchedulerConfig(
        max_num_seqs=128,
        prefill_admission_slots=0,
        max_model_len=2048,
        is_encoder_decoder=False,
    )
    admission_config = SchedulerConfig(
        max_num_seqs=128,
        prefill_admission_slots=25,
        max_model_len=2048,
        is_encoder_decoder=False,
    )

    assert default_config.compute_hash() != admission_config.compute_hash()


def test_prefill_admission_rejects_non_generation_runner():
    with pytest.raises(
        ValueError,
        match="only supported for generation models",
    ):
        SchedulerConfig(
            runner_type="pooling",
            max_num_seqs=128,
            prefill_admission_slots=25,
            max_model_len=2048,
            is_encoder_decoder=False,
        )


def test_prefill_admission_disabled_preserves_default_limits():
    policy = PrefillAdmissionPolicy(
        max_num_seqs=128,
        prefill_admission_slots=0,
    )

    limits = policy.get_limits(
        num_resident_decodes=64,
        num_running_prefills=0,
        num_waiting_prefills=25,
        resident_prefill_capacity=0,
        defer_prefills=False,
    )

    assert not policy.enabled
    assert policy.max_num_resident_reqs == 128
    assert limits.max_scheduled_running == 128
    assert limits.max_new_prefills is None
    assert not limits.defer_running_prefills


@pytest.mark.parametrize(
    (
        "num_resident_decodes",
        "num_running_prefills",
        "num_waiting_prefills",
        "resident_prefill_capacity",
        "expected_running",
        "expected_prefills",
    ),
    [
        (128, 0, 25, 25, 128, 0),
        (127, 0, 1, 25, 127, 1),
        (100, 0, 25, 25, 103, 25),
        (100, 20, 25, 25, 123, 5),
        (100, 0, 25, 3, 125, 3),
    ],
)
def test_prefill_admission_limits(
    num_resident_decodes: int,
    num_running_prefills: int,
    num_waiting_prefills: int,
    resident_prefill_capacity: int,
    expected_running: int,
    expected_prefills: int,
):
    policy = PrefillAdmissionPolicy(
        max_num_seqs=128,
        prefill_admission_slots=25,
    )

    limits = policy.get_limits(
        num_resident_decodes=num_resident_decodes,
        num_running_prefills=num_running_prefills,
        num_waiting_prefills=num_waiting_prefills,
        resident_prefill_capacity=resident_prefill_capacity,
        defer_prefills=False,
    )

    assert policy.max_num_resident_reqs == 153
    assert limits.max_scheduled_running == expected_running
    assert limits.max_new_prefills == expected_prefills
    assert limits.defer_running_prefills == (num_resident_decodes >= 128)


def test_prefill_admission_respects_external_throttling():
    policy = PrefillAdmissionPolicy(
        max_num_seqs=128,
        prefill_admission_slots=25,
    )

    limits = policy.get_limits(
        num_resident_decodes=100,
        num_running_prefills=0,
        num_waiting_prefills=25,
        resident_prefill_capacity=25,
        defer_prefills=True,
    )

    assert limits.max_scheduled_running == 128
    assert limits.max_new_prefills == 0
    assert not limits.defer_running_prefills