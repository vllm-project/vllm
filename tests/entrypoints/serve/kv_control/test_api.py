# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP and Python must observe the same correlated engine control state."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import msgspec
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from benchmarks.programmable_kv_cache.rollout import (
    RolloutClient,
    ToolDurationEstimator,
)
from vllm.config import KVControlConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.kv_control.api_router import attach_router
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.api import KVControlService, unwrap_control_reply
from vllm.v1.kv_hints.control import OperationRegistry
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex

pytestmark = pytest.mark.cpu_test


def test_tool_estimates_learn_completed_batches_without_future_samples():
    estimator = ToolDurationEstimator(0.25, max_batches=2)
    assert estimator.predict(("lookup",)) == 0.25
    estimator.observe(("lookup",), 0.01)
    assert estimator.predict(("lookup",)) == 0.01
    assert estimator.predict(("calculate",)) == 0.25
    estimator.observe(("calculate", "lookup"), 0.03)
    assert estimator.predict(("lookup", "calculate")) == 0.03
    estimator.observe(("slow",), 2)
    assert estimator.predict(("lookup",)) == 0.25


def test_rollout_analysis_retains_opposite_forecast_errors(tmp_path):
    """Early and late predictions must remain visible when signed errors cancel."""
    import gzip

    from benchmarks.programmable_kv_cache.analyze_rollout import audit

    path = tmp_path / "report.json.gz"
    with gzip.open(path, "wt") as file:
        json.dump(
            {
                "passed": True,
                "requests": [],
                "controls": [],
                "tool_waits": [
                    {"predicted_seconds": 1, "actual_seconds": 2},
                    {"predicted_seconds": 2, "actual_seconds": 1},
                ],
            },
            file,
        )
    result = audit(path)
    assert result["forecast_error_seconds"]["median"] == 0
    assert result["forecast_absolute_error_seconds"]["median"] == 1
    assert result["forecast_underestimates"] == result["forecast_overestimates"] == 1


def test_sync_control_acknowledges_after_preserving_terminal_failure():
    from benchmarks.programmable_kv_cache.client import wait_operation

    methods = []
    failure = {"message_id": "m", "action_id": "a", "status": "failed"}

    def call(method, payload):
        methods.append(method)
        if method == "acknowledge":
            assert methods == ["operation", "operation", "acknowledge"]
            return {"acknowledged": True}
        return {**failure, "status": "running"} if len(methods) == 1 else failure

    assert wait_operation(call, "m", "a", acknowledge=True) == failure
    assert methods[-1] == "acknowledge"


@pytest_asyncio.fixture
async def rollout_client():
    clients = []

    async def create(mode="priority", handler=None):
        client = RolloutClient("http://engine", mode)
        if handler is not None:
            await client.http.aclose()
            client.http = httpx.AsyncClient(
                base_url="http://engine", transport=httpx.MockTransport(handler)
            )
        clients.append(client)
        return client

    yield create
    for client in clients:
        await client.http.aclose()


@pytest.mark.parametrize("offset,shared", [(-1, False), (5, True), (11, False)])
def test_gpu_process_audit_uses_the_measured_workload_window(tmp_path, offset, shared):
    import csv
    from datetime import datetime, timezone

    from benchmarks.programmable_kv_cache.run_rollout_matrix import gpu_process_window

    start = 1700000000.0
    path = tmp_path / "processes.csv"
    with path.open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "gpu_uuid", "pid", "process_name", "used_memory"])
        for seconds, pid in [(0, 10), (5, 10), (10, 10), (offset, 20)]:
            stamp = datetime.fromtimestamp(start + seconds, timezone.utc).strftime(
                "%Y/%m/%d %H:%M:%S.%f"
            )
            writer.writerow([stamp, "GPU-example", pid, "worker", "1024 MiB"])
    result = gpu_process_window(path, start, start + 10, {10})
    assert bool(result["unexpected_processes"]) == shared
    assert result["devices"]["GPU-example"]["max_gap_seconds"] == 5


@pytest.mark.asyncio
async def test_rollout_uses_terminal_receipt_but_waits_for_derived_transfer(
    rollout_client,
):
    methods = []

    async def handle(request):
        body = json.loads(request.content)
        methods.append(body["method"])
        if body["method"] == "submit":
            message = body["payload"]["message_id"]
            result = {
                "message_id": message,
                "status": "completed",
                "derived_operations": [{"message_id": "copy", "action_id": "a"}],
            }
            return httpx.Response(200, json={"results": [result]})
        if body["method"] == "operation":
            assert body["payload"]["message_id"] == "copy"
            return httpx.Response(
                200, json={"message_id": "copy", "status": "completed"}
            )
        assert body["method"] == "acknowledge"
        return httpx.Response(200, json={"acknowledged": True})

    client = await rollout_client("policy", handle)
    result = await client.action("signal", {"signal": "resume"}, version="1")
    assert result["children"][0]["status"] == "completed"
    assert methods.count("operation") == 1
    assert methods.count("acknowledge") == 2


@pytest.mark.asyncio
async def test_rollout_resume_overlaps_control_and_finish_drains_it(rollout_client):
    release, resumed = asyncio.Event(), asyncio.Event()
    queried = []

    async def handle(request):
        if request.url.path == "/v1/completions":
            item = {
                "choices": [{"token_ids": [1], "text": "ok", "finish_reason": "stop"}]
            }
            return httpx.Response(
                200, text=f"data: {json.dumps(item)}\n\ndata: [DONE]\n\n"
            )
        body = json.loads(request.content)
        queried.append(body["method"])
        if body["method"] == "associations":
            return httpx.Response(200, json={"items": []})
        if body["method"] == "operation":
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(200, json={"acknowledged": True})

    client = await rollout_client("priority", handle)
    client.model = "model"
    client.resume_events["r"] = resumed
    client._track("r", asyncio.create_task(release.wait()))
    try:
        generated = await asyncio.wait_for(client.generate("r", [1, 2], {}), 5)
        assert generated["token_ids"] == [1] and resumed.is_set()
        cleanup = asyncio.create_task(client.finish("r"))
        await asyncio.sleep(0)
        assert not cleanup.done()
    finally:
        release.set()
    await cleanup
    assert "associations" in queried and not client.background


@pytest.mark.asyncio
async def test_rollout_collects_association_before_long_generation_can_expire_result(
    rollout_client,
):
    collected = asyncio.Event()
    queries = 0

    class Generation(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"token_ids": [1], "text": "ok"}]}\n\n'
            await asyncio.wait_for(collected.wait(), 5)
            yield b'data: {"choices": [{"finish_reason": "stop"}]}\n\ndata: [DONE]\n\n'

    async def handle(request):
        nonlocal queries
        if request.url.path == "/v1/completions":
            association = json.loads(request.content)["kv_hints"]["actions"][0][
                "payload"
            ]
            assert association["observe_growth"]
            assert "end_token" not in association["target"], (
                "Generated checkpoints must remain inside the observed scope"
            )
            return httpx.Response(200, stream=Generation())
        method = json.loads(request.content)["method"]
        if method == "operation":
            queries += 1
            if queries == 1:
                return httpx.Response(404, json={"error": {"code": "missing"}})
            collected.set()
            return httpx.Response(200, json={"status": "completed"})
        if method == "associations":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(200, json={"acknowledged": True})

    client = await rollout_client("priority", handle)
    client.model = "model"
    row = await client.generate("r", [1, 2], {})
    await client.finish("r")
    assert row["token_ids"] == [1] and queries == 2


@pytest.mark.asyncio
async def test_rollout_drains_other_control_even_when_one_task_fails(rollout_client):
    release = asyncio.Event()

    async def fail():
        raise ValueError("control failed")

    client = await rollout_client("priority")
    client._track("r", asyncio.create_task(fail()))
    client._track("r", asyncio.create_task(release.wait()))
    draining = asyncio.create_task(client.drain("r"))
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(draining), 0.05)
    finally:
        release.set()
    with pytest.raises(ValueError, match="control failed"):
        await draining


@pytest.mark.asyncio
async def test_rollout_preserves_control_order_across_turns(
    rollout_client,
    monkeypatch,
):
    release, first_started, second_started = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    order = []

    async def action(kind, payload, **kwargs):
        continuation = payload["target"]["continuation_id"]
        order.append(continuation)
        if continuation == "turn-0":
            first_started.set()
            await release.wait()
        else:
            second_started.set()
        return {"status": "completed"}

    client = await rollout_client("priority")
    monkeypatch.setattr(client, "action", action)
    client.targets["r"] = {"session_id": "s", "continuation_id": "turn-0"}
    await client.pause("r")
    await first_started.wait()
    client.targets["r"] = {"session_id": "s", "continuation_id": "turn-1"}
    await client.pause("r")
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(second_started.wait(), 0.05)
    finally:
        release.set()
    await client.drain("r")
    assert order == ["turn-0", "turn-1"]


@pytest.mark.asyncio
async def test_duplicate_pause_keeps_a_resume_event_for_every_control_task(
    monkeypatch, rollout_client
):
    async def action(kind, payload, **kwargs):
        if submitted := kwargs.get("submitted"):
            submitted.set()
        return {"status": "completed", "derived_operations": [{"action_id": "copy"}]}

    client = await rollout_client("policy")
    monkeypatch.setattr(client, "action", action)
    client.targets["r"] = {"session_id": "s", "continuation_id": "turn-0"}
    await client.pause("r")
    await client.pause("r")
    await asyncio.wait_for(client.drain("r"), 5)
    assert not client.background and not client.preparations


@pytest.mark.asyncio
async def test_pause_prediction_reaches_policy_before_store_completion(
    monkeypatch, rollout_client
):
    entered, release = asyncio.Event(), asyncio.Event()
    signals = []

    async def action(kind, payload, **kwargs):
        signals.append(payload)
        if payload["signal"] == "tool_wait":
            entered.set()
            await release.wait()
        return {"status": "completed", "derived_operations": []}

    client = await rollout_client("policy")
    monkeypatch.setattr(client, "action", action)
    client.targets["r"] = {"session_id": "s", "continuation_id": "turn-0"}
    await client.pause("r", 5)
    await asyncio.wait_for(entered.wait(), 1)
    assert 0 < signals[0]["resume_after_seconds"] <= 5
    release.set()
    await client.drain("r")
    assert [s["signal"] for s in signals] == ["tool_wait", "resume"]


@pytest.mark.asyncio
async def test_rollout_cleanup_uses_session_revision_and_keeps_unrelated_session(
    rollout_client,
    control_client,
):
    http, service = control_client
    index = service.controller.index
    index.associate("owned", "c", (), expected_revision=0)
    index.associate("other", "c", (), expected_revision=0)

    async def handle(request):
        response = http.post("/v1/kv/control", json=json.loads(request.content))
        return httpx.Response(response.status_code, json=response.json())

    client = await rollout_client("priority", handle)
    client.invocation = {"engine_id": "engine", "engine_epoch": "epoch"}
    client.targets["r"] = {"session_id": "owned", "continuation_id": "c"}
    await client.finish("r")
    assert not client.targets
    assert {row.session_id for row in index.query().items} == {"other"}


@pytest.mark.asyncio
async def test_retention_rollout_applies_policy_priority_without_movement(
    rollout_client,
    control_client,
):
    from tests.v1.core.test_prefix_caching import (
        install_movement_policies,
        make_movement_controller,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.kv_hints.policy import RetentionPolicy

    http, service = control_client
    init_none_hash(sha256)
    manager, request, controller, adapter, target = make_movement_controller()
    manager.allocate_slots(request, 4)
    manager.cache_blocks(request, 4)
    manager.free(request)
    runtime, _ = install_movement_policies(manager, controller)
    runtime.policies["retention"] = RetentionPolicy()
    service.controller = controller

    async def handle(request):
        response = http.post("/v1/kv/control", json=json.loads(request.content))
        return httpx.Response(response.status_code, json=response.json())

    client = await rollout_client("retention", handle)
    client.invocation = {"engine_id": "engine", "engine_epoch": "epoch"}
    client.targets["r"] = {"kind": "session", "session_id": "s"}
    await client.action(
        "associate",
        {
            "session_id": "s",
            "continuation_id": "c",
            "expected_revision": 0,
            "target": target,
        },
        version="1",
    )
    await client.pause("r")
    await client.drain("r")
    (block,) = manager.get_computed_blocks(request)[0].blocks[0]
    assert controller.executor.retention_priority.copies.priority(block) == 50
    assert any("priority_applied" in r["completed_steps"] for r in client.controls)
    assert not adapter.jobs
    await client.finish("r")
    assert not controller.index.query().items
    assert controller.executor.retention_priority.copies.priority(block) is None


@pytest.mark.asyncio
async def test_failed_rollout_cancels_other_requests_before_client_cleanup():
    from benchmarks.programmable_kv_cache.verl_rollout import run_episodes

    started, closed = asyncio.Event(), asyncio.Event()

    async def waiting():
        started.set()
        try:
            await asyncio.Future()
        finally:
            closed.set()

    async def failing():
        await started.wait()
        raise ValueError("original serving failure")

    with pytest.raises(ValueError, match="original serving failure"):
        await run_episodes([waiting(), failing()])
    assert closed.is_set()


@pytest.mark.asyncio
async def test_finish_cancels_unused_preparation_before_draining_results(
    control_client, rollout_client
):
    from tests.v1.core.test_prefix_caching import (
        install_movement_policies,
        make_movement_controller,
        make_request,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.kv_hints.policy import ToolWaitPolicy

    http, service = control_client
    init_none_hash(sha256)
    manager, request, controller, adapter, target = make_movement_controller()
    assert manager.allocate_slots(request, 4) is not None
    manager.cache_blocks(request, 4)
    manager.free(request)
    runtime, _ = install_movement_policies(manager, controller)
    runtime.policies["movement"] = ToolWaitPolicy(prefetch_on_admission=True)
    other = make_request("other", list(range(20, 25)), 4, sha256)
    other.session_id = "other"
    runtime.requests = lambda: (other,)
    service.controller = controller
    predicted = asyncio.Event()
    signals = []

    async def handle(request):
        data = json.loads(request.content)
        response = http.post("/v1/kv/control", json=data)
        if data["method"] == "submit":
            signal = data["payload"]["actions"][0]["payload"].get("signal")
            if signal:
                signals.append(signal)
            if signal == "tool_wait":
                adapter.finish()
                controller.executor.progress()
            if signal == "resume_prediction":
                predicted.set()
        return httpx.Response(response.status_code, json=response.json())

    client = await rollout_client("policy", handle)
    client.invocation = {"engine_id": "engine", "engine_epoch": "epoch"}
    await client.action(
        "associate",
        {
            "session_id": "s",
            "continuation_id": "c",
            "expected_revision": 0,
            "target": target,
        },
        version="1",
    )
    client.targets["r"] = {"kind": "session", "session_id": "s", "continuation_id": "c"}
    await client.pause("r", 10)
    await asyncio.wait_for(predicted.wait(), 5)
    await asyncio.wait_for(client.finish("r"), 5)
    assert signals == ["tool_wait", "resume_prediction"]
    assert not controller.index.query().items
    assert controller.registry.capacity()["records"] == 0
    assert not runtime.has_pending and not adapter.jobs


@pytest.mark.asyncio
async def test_timed_out_control_acknowledges_cancelled_results_but_keeps_worker_hold(
    control_client, rollout_client
):
    from tests.v1.core.test_prefix_caching import (
        install_movement_policies,
        make_movement_controller,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash

    http, service = control_client
    init_none_hash(sha256)
    manager, request, controller, adapter, target = make_movement_controller()
    install_movement_policies(manager, controller, "movement")
    service.controller = controller

    async def handle(request):
        response = http.post("/v1/kv/control", json=json.loads(request.content))
        return httpx.Response(response.status_code, json=response.json())

    client = await rollout_client("policy", handle)
    client.invocation = {"engine_id": "engine", "engine_epoch": "epoch"}
    client.control_timeout = 0.02
    result = await client.action(
        "signal", {"target": target, "signal": "resume"}, version="1", attempts=1
    )
    try:
        assert result["status"] == "failed"
        assert controller.registry.capacity()["records"] == 0
        assert manager.block_pool.get_num_free_blocks() == 2
        assert (await client.call("capacity", {}))["pending_transfers"]
        error = next(r for r in client.controls if r["status"] == "client_error")
        assert error["driver_target"] == target
    finally:
        adapter.finish()
        controller.executor.progress()
    assert manager.block_pool.get_num_free_blocks() == 3
    assert manager.get_computed_blocks(request)[1] == 0
    assert not (await client.wait_control_idle())["pending_transfers"]


@pytest.fixture
def control_client():
    registry = OperationRegistry("engine", engine_epoch="epoch")
    index = SessionPrefixIndex("namespace", registry.engine_epoch, max_events=2)
    service = KVControlService(
        NativeKVController(registry, index, RetentionDeclarations(index))
    )

    class Client:
        async def get_kv_control_capabilities(self):
            return [msgspec.to_builtins(service.controller.capabilities())]

        async def kv_control(self, method, invocation, payload):
            return unwrap_control_reply(service.call(method, invocation, payload))

    app = FastAPI()
    app.state.engine_client = Client()
    app.state.vllm_config = SimpleNamespace(kv_control_config=KVControlConfig())
    attach_router(app)
    with TestClient(app) as client:
        yield client, service


def body(method, payload, **invocation):
    return {
        "method": method,
        "invocation": {"engine_id": "engine", "engine_epoch": "epoch", **invocation},
        "payload": payload,
    }


def hint(message, kind, payload):
    return msgspec.to_builtins(
        KvHintsEnvelope("0.1", message, [KvHintAction("action", kind, "1", payload)])
    )


def test_http_submission_and_python_query_share_results_and_retries(control_client):
    client, service = control_client
    assert (
        client.get("/v1/kv/capabilities").json()["engines"][0]["engine_id"] == "engine"
    )
    envelope = hint(
        "message",
        "signal",
        {
            "target": {"kind": "session", "session_id": "s"},
            "signal": "tool_wait",
        },
    )
    response = client.post("/v1/kv/control", json=body("submit", envelope))
    assert response.status_code == 200
    submitted = response.json()
    query = unwrap_control_reply(
        service.call(
            "operation",
            body("operation", {})["invocation"],
            {"message_id": "message", "action_id": "action"},
        )
    )
    assert query == submitted["results"][0]
    assert query["status"] == "unconsumed"
    assert client.post("/v1/kv/control", json=body("submit", envelope)).json()[
        "replayed"
    ]
    events = client.post(
        "/v1/kv/control", json=body("events", {"stream": "operation"})
    ).json()
    assert events[-1] == query
    acknowledged = client.post(
        "/v1/kv/control", json=body("acknowledge", {"message_id": "message"})
    )
    assert acknowledged.json() == {"acknowledged": True}
    retry = client.post("/v1/kv/control", json=body("submit", envelope))
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "acknowledged"
    capacity = client.post("/v1/kv/control", json=body("capacity", {})).json()
    assert capacity["records"] == 0
    assert capacity["acknowledged_messages"] == 1


def test_http_association_snapshot_supplies_cursor_after_real_history_loss(
    control_client,
):
    client, service = control_client

    def associate(session):
        response = client.post(
            "/v1/kv/control",
            json=body(
                "submit",
                hint(
                    session,
                    "associate",
                    {
                        "session_id": session,
                        "continuation_id": "c",
                        "expected_revision": 0,
                        "target": {
                            "kind": "hashes",
                            "namespace": "namespace",
                            "blocks": [
                                {
                                    "block_hash": "abcd",
                                    "group_id": 0,
                                    "start_token": 0,
                                    "end_token": 16,
                                }
                            ],
                        },
                    },
                ),
            ),
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["status"] == "completed"

    associate("a")
    associate("b")
    gap = client.post("/v1/kv/control", json=body("events", {"stream": "association"}))
    assert gap.status_code == 409
    assert gap.json()["error"]["code"] == "event_gap"
    snapshot = client.post("/v1/kv/control", json=body("associations", {})).json()
    assert [item["session_id"] for item in snapshot["items"]] == ["a", "b"]
    assert snapshot["engine_epoch"] == "epoch"
    assert snapshot["namespace"] == "namespace"
    associate("c")
    deltas = client.post(
        "/v1/kv/control",
        json=body(
            "events",
            {
                "stream": "association",
                "after": snapshot["event_sequence"],
            },
        ),
    ).json()
    assert [event["kind"] for event in deltas] == ["associated", "updated"]
    assert {event["session_id"] for event in deltas} == {"c"}
    assert service.controller.index.members(0, "abcd") == (
        ("a", "c"),
        ("b", "c"),
        ("c", "c"),
    )


@pytest.mark.parametrize(
    "invocation,method,payload,status,code",
    [
        ({"engine_id": "other"}, "associations", {}, 404, "wrong_engine"),
        ({"engine_epoch": "obsolete"}, "associations", {}, 409, "stale_epoch"),
        ({"namespace": "old"}, "associations", {}, 400, "incompatible"),
        ({"consumer": "backend"}, "associations", {}, 501, "unsupported"),
        ({"binding_id": "inference"}, "associations", {}, 400, "invalid_payload"),
        ({}, "associations", {"limit": 0}, 400, "invalid_payload"),
        ({}, "associations", {"limit": True}, 400, "invalid_payload"),
        ({}, "associations", {"after": ["s", "c"]}, 400, "invalid_cursor"),
        ({}, "associations", {"unknown": 1}, 400, "invalid_payload"),
        ({}, "events", {"stream": "operation", "after": 100}, 409, "event_gap"),
        ({}, "unknown", {}, 400, "invalid_payload"),
        ({}, "operation", {"message_id": "absent", "action_id": "a"}, 404, "missing"),
    ],
)
def test_http_control_errors_preserve_engine_state(
    control_client, invocation, method, payload, status, code
):
    client, service = control_client
    response = client.post("/v1/kv/control", json=body(method, payload, **invocation))
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    if method == "unknown":
        assert (
            service.call(method, body(method, {})["invocation"], payload)["error"][
                "code"
            ]
            == "unsupported"
        )
    assert service.controller.index.query().items == ()


def test_http_rejects_invalid_json_unknown_fields_and_oversized_body(control_client):
    client, service = control_client
    invalid = client.post(
        "/v1/kv/control", content="{", headers={"Content-Type": "application/json"}
    )
    assert invalid.status_code == 400
    unknown = client.post(
        "/v1/kv/control", json={**body("associations", {}), "extra": 1}
    )
    assert unknown.status_code == 400
    client.app.state.vllm_config.kv_control_config.max_envelope_bytes = 32
    oversized = client.post(
        "/v1/kv/control",
        content=b" " * 2081,
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    assert service.controller.index.query().items == ()


def test_http_disabled_feature_reports_unsupported(control_client):
    client, _ = control_client
    client.app.state.vllm_config.kv_control_config = None
    response = client.post("/v1/kv/control", json=body("associations", {}))
    assert response.status_code == 501


@pytest.mark.parametrize(
    "request_type,fields",
    [
        (CompletionRequest, {"prompt": "hello"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
def test_inference_protocol_preserves_the_merged_opaque_envelope(request_type, fields):
    hints = {
        "protocol_version": "backend-version",
        "message_id": "m",
        "actions": [
            {
                "action_id": "a",
                "action_type": "opaque",
                "action_version": "future",
                "payload": {"values": [None, True, 0.5, {"text": "值"}]},
            }
        ],
    }
    request = request_type(**fields, kv_hints=hints)
    assert isinstance(request.kv_hints, KvHintsEnvelope)
    assert request.model_dump()["kv_hints"] == hints
    assert request.model_json_schema()["properties"]["kv_hints"]
    for options in ({"n": 2}, {"use_beam_search": True}):
        with pytest.raises(ValidationError, match="n=1"):
            request_type(**fields, kv_hints=hints, **options)
    with pytest.raises(ValidationError, match="message_id"):
        request_type(**fields, kv_hints={**hints, "message_id": ""})
