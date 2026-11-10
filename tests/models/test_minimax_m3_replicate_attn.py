# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 replicated attention (VLLM_MINIMAX_M3_REPLICATE_ATTN): layout checks.

CPU-only: the head count attention metadata builders size for, and the parallel
layouts the switch refuses because they turn the TP-synced Triton autotune off.
"""

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.models.minimax_m3.common.replicate import (
    attention_tp_size,
    unsupported_layout,
)


@pytest.mark.parametrize("replicate, expected", [(False, 4), (True, 1)])
def test_attention_tp_size(monkeypatch: pytest.MonkeyPatch, replicate, expected):
    monkeypatch.setattr(
        envs, "VLLM_MINIMAX_M3_REPLICATE_ATTN", replicate, raising=False
    )
    assert attention_tp_size(4) == expected
    # MiniMax-M3.1-Flash: 64 query heads, all of them on every rank when replicated.
    assert 64 // attention_tp_size(4) == (64 if replicate else 16)


def _layout(ep=False, dcp=1, pcp=1):
    return SimpleNamespace(
        enable_expert_parallel=ep,
        decode_context_parallel_size=dcp,
        prefill_context_parallel_size=pcp,
    )


@pytest.mark.parametrize(
    "layout, expected",
    [
        (_layout(), None),
        (_layout(ep=True), "expert parallelism"),
        (_layout(dcp=2), "decode/prefill context parallelism"),
        (_layout(pcp=2), "decode/prefill context parallelism"),
        (
            _layout(ep=True, dcp=2),
            "expert parallelism, decode/prefill context parallelism",
        ),
    ],
)
def test_unsupported_layout(layout, expected):
    assert unsupported_layout(layout) == expected
