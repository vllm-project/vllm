# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.eplb.eplb_state import EplbState


def test_build_logical_sleep_rank_mapping_requires_suffix():
    assert EplbState.build_logical_sleep_rank_mapping(4, [2, 3]) == {
        0: 0,
        1: 1,
        2: -1,
        3: -1,
    }

    with pytest.raises(ValueError, match="sleeping a suffix"):
        EplbState.build_logical_sleep_rank_mapping(4, [1, 3])


@pytest.mark.parametrize("sleeping_ranks", [[], [0, 1, 2, 3], [-1], [4]])
def test_build_logical_sleep_rank_mapping_rejects_invalid_ranks(sleeping_ranks):
    with pytest.raises(ValueError):
        EplbState.build_logical_sleep_rank_mapping(4, sleeping_ranks)
