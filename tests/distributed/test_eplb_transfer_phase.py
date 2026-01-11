# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.eplb.eplb_state import TransferPhase


def test_transfer_phase_enum_values_distinct():
    """Ensure all enum values are distinct."""
    values = [TransferPhase.IDLE, TransferPhase.PRODUCING, TransferPhase.CONSUMING]
    assert len(values) == len(set(values))


@pytest.mark.parametrize(
    "phase,expected_name",
    [
        (TransferPhase.IDLE, "IDLE"),
        (TransferPhase.PRODUCING, "PRODUCING"),
        (TransferPhase.CONSUMING, "CONSUMING"),
    ],
)
def test_transfer_phase_str_returns_name(phase, expected_name):
    """__str__ should return the enum name."""
    assert str(phase) == expected_name


@pytest.mark.parametrize("phase", list(TransferPhase))
def test_transfer_phase_int_conversion(phase):
    """IntEnum should be convertible to int for distributed ops."""
    assert isinstance(int(phase), int)


@pytest.mark.parametrize(
    "phase,is_active",
    [
        (TransferPhase.IDLE, False),
        (TransferPhase.PRODUCING, True),
        (TransferPhase.CONSUMING, True),
    ],
)
def test_transfer_phase_active_check(phase, is_active):
    """Non-IDLE phases indicate active transfer."""
    assert (phase != TransferPhase.IDLE) == is_active
