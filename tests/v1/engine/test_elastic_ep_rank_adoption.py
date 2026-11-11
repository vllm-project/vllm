# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A front-end started for one DP rank adopts the rank its parked elastic EP
engine joins as."""

from types import SimpleNamespace

import pytest

from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineState,
    _adopt_parked_engine_rank,
)

pytestmark = pytest.mark.skip_global_cleanup


def _parallel_config(rank: int) -> SimpleNamespace:
    return SimpleNamespace(data_parallel_rank=rank, data_parallel_index=rank)


def test_parked_engine_rank_is_adopted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ELASTIC_EP_JOIN_FILE", "/tmp/join.json")
    engine = CoreEngine(index=0, local=True)
    parallel_config = _parallel_config(0)

    adopted = _adopt_parked_engine_rank([engine], 3, parallel_config)

    assert adopted is engine
    assert engine.identity == (3).to_bytes(2, "little")
    assert parallel_config.data_parallel_rank == 3
    assert parallel_config.data_parallel_index == 3


@pytest.mark.parametrize(
    ("join_file", "engines", "state"),
    [
        # Not a parked joiner.
        (None, [CoreEngine(0)], CoreEngineState.NEW),
        # The front-end manages more than one engine.
        ("/tmp/join.json", [CoreEngine(0), CoreEngine(1)], CoreEngineState.NEW),
        # The engine is remote.
        ("/tmp/join.json", [CoreEngine(0, local=False)], CoreEngineState.NEW),
        # The engine already said hello under its own rank.
        ("/tmp/join.json", [CoreEngine(0)], CoreEngineState.CONNECTED),
    ],
)
def test_unexpected_rank_is_not_adopted(
    monkeypatch: pytest.MonkeyPatch,
    join_file: str | None,
    engines: list[CoreEngine],
    state: CoreEngineState,
):
    if join_file is None:
        monkeypatch.delenv("VLLM_ELASTIC_EP_JOIN_FILE", raising=False)
    else:
        monkeypatch.setenv("VLLM_ELASTIC_EP_JOIN_FILE", join_file)
    engines[0].state = state
    parallel_config = _parallel_config(0)

    assert _adopt_parked_engine_rank(engines, 3, parallel_config) is None
    assert engines[0].identity == (0).to_bytes(2, "little")
    assert parallel_config.data_parallel_rank == 0
