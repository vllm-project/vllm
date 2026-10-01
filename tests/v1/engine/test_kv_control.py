# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control mutations must target one engine even under the DP utility broadcast."""

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from vllm.config import CacheConfig, KVControlConfig, ParallelConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_hints.control import ControlError
from vllm.v1.kv_hints.engine import create_control_service
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


def make_service(*, engine_id="test", rank=0, weight_version="default"):
    config = SimpleNamespace(
        kv_control_config=KVControlConfig(engine_id=engine_id),
        cache_config=CacheConfig(block_size=16),
        model_config=SimpleNamespace(compute_hash=lambda: "model"),
        parallel_config=ParallelConfig(data_parallel_size=2, data_parallel_rank=rank),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    manager = KVCacheManager(
        KVCacheConfig(32, [], [KVCacheGroupSpec(["layer"], spec)]),
        max_model_len=128,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    return create_control_service(config, manager, weight_version)


def test_engine_epochs_differ_but_compatible_dp_ranks_share_namespace():
    a, b = make_service(rank=0), make_service(rank=1)
    assert a.controller.registry.engine_id == "test:dp0"
    assert b.controller.registry.engine_id == "test:dp1"
    assert a.controller.registry.engine_epoch != b.controller.registry.engine_epoch
    assert a.controller.index.namespace == b.controller.index.namespace
    assert (
        make_service(weight_version="updated").controller.index.namespace
        != a.controller.index.namespace
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"max_records": 0},
        {"result_ttl": float("inf")},
        {"attached_consumer": "guess"},
        {"engine_id": ""},
    ],
)
def test_invalid_control_configuration_fails_before_engine_startup(settings):
    with pytest.raises(ValidationError):
        KVControlConfig(**settings)


def test_engine_utility_uses_safe_serialization_and_disabled_path_has_no_state():
    engine = object.__new__(EngineCore)
    engine.kv_control_service = None
    assert engine.get_kv_control_capabilities() is None
    assert engine.kv_control("associations", {}, {})["error"]["code"] == "unsupported"
    engine.kv_control_service = make_service()
    caps = engine.get_kv_control_capabilities()
    invocation = {key: caps[key] for key in ("engine_id", "engine_epoch")}
    reply = engine.kv_control("associations", invocation, {})
    assert reply["result"]["items"] == []
    assert MsgpackDecoder().decode(MsgpackEncoder().encode(reply)) == reply


@pytest.mark.asyncio
async def test_dp_control_mutation_is_sent_only_to_the_named_engine():
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"a", b"b"]
    calls = []

    async def utility(method, *args, engine):
        calls.append((engine, method))
        if method == "get_kv_control_capabilities":
            return {"engine_id": engine.decode(), "engine_epoch": "epoch"}
        return {"ok": True, "result": {"engine_id": engine.decode()}}

    client._call_utility_async = utility
    result = await client.kv_control_async(
        "submit", {"engine_id": "b", "engine_epoch": "epoch"}, {}
    )
    assert result["engine_id"] == "b"
    assert [engine for engine, method in calls if method == "kv_control"] == [b"b"]
    with pytest.raises(ControlError, match="not connected"):
        await client.kv_control_async(
            "submit", {"engine_id": "absent", "engine_epoch": "epoch"}, {}
        )
    assert [engine for engine, method in calls if method == "kv_control"] == [b"b"]


@pytest.mark.asyncio
async def test_dp_discovery_rejects_ambiguous_engine_ids_before_mutation():
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"a", b"b"]

    async def utility(method, *args, engine):
        assert method == "get_kv_control_capabilities"
        return {"engine_id": "duplicate", "engine_epoch": engine.decode()}

    client._call_utility_async = utility
    with pytest.raises(ControlError, match="duplicate"):
        await client.kv_control_async(
            "submit", {"engine_id": "duplicate", "engine_epoch": "epoch"}, {}
        )


@pytest.mark.parametrize("consumer", ["native", "backend", None])
def test_request_envelope_has_one_interpretation_owner_before_connector_hook(consumer):
    engine = object.__new__(EngineCore)
    engine.kv_control_service = make_service() if consumer is not None else None
    engine.vllm_config = SimpleNamespace(
        kv_control_config=KVControlConfig(attached_consumer=consumer or "native")
    )
    forwarded = []
    engine.scheduler = SimpleNamespace(
        add_request=lambda request: forwarded.append(request.kv_hints),
    )
    hints = KvHintsEnvelope(
        "0.1",
        "attached",
        [
            KvHintAction(
                "action",
                "signal",
                "1",
                {
                    "target": {"kind": "session", "session_id": "s"},
                    "signal": "tool_wait",
                },
            )
        ],
    )
    if consumer == "backend":
        hints = KvHintsEnvelope(
            "opaque-version",
            "attached",
            [
                KvHintAction(
                    "action",
                    "backend.future",
                    "v9",
                    {"values": [None, True, {"n": 3.5}]},
                )
            ],
        )
    request = Request(
        "request", [1], SamplingParams(max_tokens=1), None, kv_hints=hints
    )
    engine.add_request(request)
    assert forwarded == [None if consumer == "native" else hints]
    if consumer is not None:
        operation = engine.kv_control_service.controller.registry.get(
            "attached", "action"
        )
        if consumer == "native":
            assert operation.status == "unconsumed"
            assert operation.binding_id == "request"
        else:
            assert operation is None


def test_invalid_attached_wrapper_aborts_only_that_request_and_never_forwards():
    engine = object.__new__(EngineCore)
    engine.kv_control_service = make_service()
    engine.vllm_config = SimpleNamespace(kv_control_config=KVControlConfig())
    forwarded, aborted = [], []
    engine.scheduler = SimpleNamespace(
        add_request=lambda request: forwarded.append(request.kv_hints)
    )
    engine.abort_requests = lambda ids: aborted.extend(ids)
    request = Request(
        "bad",
        [1],
        SamplingParams(max_tokens=1),
        None,
        kv_hints=KvHintsEnvelope("0.1", "", []),
    )
    engine.add_request(request)
    assert forwarded == [None]
    assert aborted == ["bad"]
    assert engine.kv_control_service.controller.index.query().items == ()
