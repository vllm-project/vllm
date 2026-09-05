# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.v1.attention.ops.dcp import resolve_dcp_q_replicate


@pytest.mark.parametrize(
    ("env_override", "configured", "dcp_size", "pcp_size", "expected"),
    [
        # Unset env: the model/config default decides.
        (None, True, 2, 1, True),
        (None, False, 2, 1, False),
        # Explicit env wins either way, including forcing it off.
        (True, False, 2, 1, True),
        (False, True, 2, 1, False),
        # DCP inactive: nothing to replicate across.
        (True, True, 1, 1, False),
        # PCP active: DCP groups are strided, so the head mapping breaks.
        (True, True, 2, 2, False),
    ],
)
def test_resolve_dcp_q_replicate(
    monkeypatch,
    env_override: bool | None,
    configured: bool,
    dcp_size: int,
    pcp_size: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        envs,
        "is_set",
        lambda name: name == "VLLM_DCP_Q_REPLICATE" and env_override is not None,
    )
    monkeypatch.setattr(envs, "VLLM_DCP_Q_REPLICATE", bool(env_override))
    parallel_config = cast(
        ParallelConfig,
        SimpleNamespace(
            dcp_q_replicate=configured,
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )

    assert resolve_dcp_q_replicate(parallel_config) is expected
