# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.offloader.prefetch import _get_next_prefetch_index

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("module_count", "prefetch_step", "expected_targets"),
    [
        (3, 1, [1, 2, 0]),
        (3, 2, [2, 1, 0]),
        (4, 2, [2, 3, 0, 1]),
        (5, 2, [2, 3, 4, 1, 0]),
        (3, 3, [0, 1, 2]),
        (3, 4, [1, 2, 0]),
    ],
)
def test_next_prefetch_index_preserves_slot_ownership(
    module_count: int,
    prefetch_step: int,
    expected_targets: list[int],
) -> None:
    targets = [
        _get_next_prefetch_index(index, prefetch_step, module_count)
        for index in range(module_count)
    ]

    assert targets == expected_targets
    if prefetch_step < module_count:
        assert all(
            target % prefetch_step == index % prefetch_step
            for index, target in enumerate(targets)
        )
