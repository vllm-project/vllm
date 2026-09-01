# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine.core import DPEngineCoreProc

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    ("enabled", "scheduler_will_call_worker", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_scheduler_target_generation_ownership(
    enabled: bool,
    scheduler_will_call_worker: bool,
    expected: bool,
) -> None:
    core = object.__new__(DPEngineCoreProc)
    core.dp_execution_contract_enabled = enabled

    assert (
        core._scheduler_owns_target_generation(scheduler_will_call_worker) is expected
    )
