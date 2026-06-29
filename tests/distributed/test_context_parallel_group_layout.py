# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import _build_dcp_group_ranks


def _rank_layout(tp: int, pcp: int, dp: int = 1, pp: int = 1) -> torch.Tensor:
    return torch.arange(dp * pp * pcp * tp).reshape(1, dp, pp, pcp, tp)


def test_dcp_groups_span_pcp_axis_first() -> None:
    all_ranks = _rank_layout(tp=2, pcp=2)

    assert _build_dcp_group_ranks(all_ranks, 1) == [[0], [1], [2], [3]]
    assert _build_dcp_group_ranks(all_ranks, 2) == [[0, 2], [1, 3]]
    assert _build_dcp_group_ranks(all_ranks, 4) == [[0, 2, 1, 3]]


def test_dcp_groups_span_full_tp_pcp_axis() -> None:
    all_ranks = _rank_layout(tp=4, pcp=2)

    assert _build_dcp_group_ranks(all_ranks, 2) == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    assert _build_dcp_group_ranks(all_ranks, 8) == [
        [0, 4, 1, 5, 2, 6, 3, 7],
    ]
