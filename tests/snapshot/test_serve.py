# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.entrypoints.cli.serve import (
    _validate_engine_snapshot_execution,
    run_multi_api_server,
)
from vllm.v1.executor import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor


def snapshot_config(**overrides) -> Any:
    values = {
        "data_parallel_size": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "nnodes": 1,
    }
    values.update(overrides)
    return SimpleNamespace(parallel_config=SimpleNamespace(**values))


def test_engine_snapshot_accepts_uniproc_executor():
    _validate_engine_snapshot_execution(snapshot_config(), UniProcExecutor)


def test_engine_snapshot_resolves_provider_at_serve_entry(monkeypatch):
    def fail_provider_check(name):
        raise FileNotFoundError(f"missing {name}")

    monkeypatch.setattr(
        "vllm.snapshot.providers.make_snapshot_provider", fail_provider_check
    )
    args = SimpleNamespace(
        api_server_count=1,
        enable_engine_snapshot=True,
        engine_snapshot_provider="criu_cuda",
        headless=False,
    )

    with pytest.raises(FileNotFoundError, match="missing criu_cuda"):
        run_multi_api_server(args)


class DerivedUniProcExecutor(UniProcExecutor):
    pass


def test_engine_snapshot_accepts_uniproc_executor_subclass():
    _validate_engine_snapshot_execution(snapshot_config(), DerivedUniProcExecutor)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("data_parallel_size", 2),
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("nnodes", 2),
    ),
)
def test_engine_snapshot_rejects_parallel_execution(field, value):
    with pytest.raises(ValueError, match="all parallel sizes to be 1"):
        _validate_engine_snapshot_execution(
            snapshot_config(**{field: value}), UniProcExecutor
        )


@pytest.mark.parametrize("executor_class", (MultiprocExecutor, Executor))
def test_engine_snapshot_rejects_non_uniproc_executor(executor_class):
    with pytest.raises(ValueError, match="require UniProcExecutor"):
        _validate_engine_snapshot_execution(snapshot_config(), executor_class)
