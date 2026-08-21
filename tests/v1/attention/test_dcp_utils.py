# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.v1.attention.ops.dcp import resolve_dcp_q_replicate


@pytest.mark.parametrize(
    ("override", "backend", "dcp_size", "pcp_size", "expected"),
    [
        (None, "a2a", 2, 1, True),
        (None, "all_gather", 2, 1, False),
        (True, "all_gather", 2, 1, True),
        (False, "a2a", 2, 1, False),
        (True, "a2a", 1, 1, False),
        (True, "a2a", 2, 2, False),
    ],
)
def test_resolve_dcp_q_replicate(
    monkeypatch,
    override: bool | None,
    backend: str,
    dcp_size: int,
    pcp_size: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(envs, "VLLM_DCP_Q_REPLICATE", override)
    parallel_config = cast(
        ParallelConfig,
        SimpleNamespace(
            dcp_comm_backend=backend,
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )

    assert resolve_dcp_q_replicate(parallel_config) is expected
