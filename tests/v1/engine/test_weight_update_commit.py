# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.llm import LLM
from vllm.v1.engine.async_llm import AsyncLLM


class _SyncEngine:
    def __init__(self, fail_prepare: bool = False) -> None:
        self.fail_prepare = fail_prepare
        self.calls: list[str] = []

    def collective_rpc(self, method: str) -> None:
        self.calls.append(method)
        if method == "prepare_weight_update" and self.fail_prepare:
            raise RuntimeError("prepare failed")


class _AsyncClient:
    def __init__(self, fail_prepare: bool = False) -> None:
        self.fail_prepare = fail_prepare
        self.calls: list[str] = []

    async def collective_rpc(self, method: str) -> None:
        self.calls.append(method)
        if method == "prepare_weight_update" and self.fail_prepare:
            raise RuntimeError("prepare failed")


def test_sync_finish_prepares_all_workers_before_commit():
    llm = object.__new__(LLM)
    llm.llm_engine = _SyncEngine()

    LLM.finish_weight_update(llm)

    assert llm.llm_engine.calls == [
        "prepare_weight_update",
        "commit_weight_update",
    ]


def test_sync_finish_aborts_when_prepare_fails():
    llm = object.__new__(LLM)
    llm.llm_engine = _SyncEngine(fail_prepare=True)

    with pytest.raises(RuntimeError, match="prepare failed"):
        LLM.finish_weight_update(llm)

    assert llm.llm_engine.calls == [
        "prepare_weight_update",
        "abort_weight_update",
    ]


@pytest.mark.asyncio
async def test_async_finish_prepares_all_workers_before_commit():
    client = _AsyncClient()

    await AsyncLLM.finish_weight_update(client)  # type: ignore[arg-type]

    assert client.calls == ["prepare_weight_update", "commit_weight_update"]
