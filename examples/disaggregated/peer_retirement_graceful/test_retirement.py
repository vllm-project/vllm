# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import asyncio
import importlib.util
import json
import logging
import queue
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import peer_lab  # noqa: E402
from peer_lab_api import AdmissionFence  # noqa: E402
from prepare_sources import checked_sources  # noqa: E402

SOURCE = ROOT / ".sources"
checked_sources(SOURCE)
spec = importlib.util.spec_from_file_location("patch_vllm", ROOT / "patch_vllm.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


def source_method(name, patched=False, filename="base_worker.py"):
    text = (SOURCE / filename).read_text()
    if patched:
        text = patcher.transform(filename, text)
    tree = ast.parse(text)
    cls_name = (
        "NixlBaseConnectorWorker"
        if filename == "base_worker.py"
        else "NixlPullConnectorWorker"
    )
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {
        "EngineId": str,
        "logger": logging.getLogger("test"),
        "time": time,
        "Future": Future,
        "NixlConnectorMetadata": object,
        "ReqMeta": object,
        "ReadSpec": SimpleNamespace,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[name]


class Native:
    def __init__(self):
        self.calls = []
        self.fail = False

    def release_dlist_handle(self, value):
        self.calls.append(("release", value))
        if self.fail:
            raise RuntimeError("native failure")

    def remove_remote_agent(self, value):
        self.calls.append(("remove", value))


class Worker:
    _cleanup_remote_engine = source_method("_cleanup_remote_engine")
    _evict_stale_engines = source_method("_evict_stale_engines", patched=True)
    _send_heartbeats = source_method("_send_heartbeats", patched=True)
    _ensure_handshake = source_method("_ensure_handshake", patched=True)

    def __init__(self):
        self.engine_id, self.tp_rank = "decode-1", 0
        self.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
        self._bidirectional_kv_xfer_enabled = False
        self._handshake_lock = threading.RLock()
        self._handshake_futures = {}
        self._ready_requests = queue.Queue()
        self._failed_recv_reqs = queue.Queue()
        self._recving_metadata, self._recving_transfers = {}, {}
        self._reqs_to_send, self._reqs_to_process = {}, set()
        for name in peer_lab.PEER_MAPS:
            setattr(self, name, {})
        self._remote_agents = {
            "old-p": {(0, 0): "old-agent"},
            "live-p": {(0, 0): "live-agent"},
        }
        self.dst_xfer_side_handles = {"old-p": {0: 11}, "live-p": {0: 12}}
        self._engine_last_active = {"old-p": 0.0, "live-p": time.perf_counter()}
        self._engine_ttl = 3600
        self.transfer_topo = SimpleNamespace(
            unregister_remote_engine=lambda e: self.removed.append(e)
        )
        self.removed = []
        self.nixl_wrapper = Native()


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("PEER_LAB_RETIRE_ENABLED", "1")
    monkeypatch.setenv("PEER_LAB_ADMIN_KEY", "x" * 40)


def operate(worker, phase="prepare", operation="op-1"):
    return peer_lab.retire(worker, "old-p", operation, "decode-1", phase)


def test_graceful_cleanup_uses_real_v026_order_and_preserves_other_peer():
    w = Worker()
    assert operate(w)["state"] == "prepared"
    assert w.nixl_wrapper.calls == []
    result = operate(w, "commit")
    assert result["state"] == "cleanup_returned"
    assert not result["native_release_verified"]
    assert not result["gpu_memory_measured"]
    assert w.nixl_wrapper.calls == [("release", 11), ("remove", "old-agent")]
    assert w._remote_agents == {"live-p": {(0, 0): "live-agent"}}
    assert w.dst_xfer_side_handles == {"live-p": {0: 12}}
    assert w.removed == ["old-p"]
    assert peer_lab.is_fenced(w, "old-p")
    assert not peer_lab.is_fenced(w, "new-p-same-address")


@pytest.mark.parametrize(
    "field",
    [
        "_handshake_futures",
        "_recving_metadata",
        "_recving_transfers",
        "_reqs_to_send",
        "_reqs_to_process",
        "_ready_requests",
        "_failed_recv_reqs",
    ],
)
def test_busy_never_fences_or_releases(field):
    w = Worker()
    value = getattr(w, field)
    if hasattr(value, "put"):
        value.put("request")
    elif isinstance(value, set):
        value.add("request")
    else:
        value["request"] = object()
    assert operate(w)["state"] == "busy"
    assert not peer_lab.is_fenced(w, "old-p")
    assert not w.nixl_wrapper.calls


def test_two_phase_requires_prepare():
    w = Worker()
    assert operate(w, "commit")["reason"] == "prepare_required"
    assert not w.nixl_wrapper.calls


def test_retries_do_not_double_release():
    w = Worker()
    operate(w)
    first = operate(w, "commit")
    first["state"] = "caller_mutation"
    assert operate(w, "commit")["state"] == "cleanup_returned"
    assert len(w.nixl_wrapper.calls) == 2
    assert operate(w, operation="op-2")["state"] == "conflict"


def test_retired_heartbeat_cannot_resurrect_peer():
    w = Worker()
    operate(w)
    operate(w, "commit")
    w._send_heartbeats(SimpleNamespace(heartbeat_by_engine={"old-p": object()}))
    assert "old-p" not in w._remote_agents
    with pytest.raises(RuntimeError, match="handshake rejected"):
        w._ensure_handshake("old-p", "127.0.0.1", 10001, 4)
    assert len(w.nixl_wrapper.calls) == 2


@pytest.mark.parametrize("cached", [True, False])
def test_notification_only_metadata_cleanup_does_not_clear_real_read(cached):
    w = Worker()
    local_blocks = [] if cached else [[1]]
    meta = SimpleNamespace(
        remote=SimpleNamespace(engine_id="old-p", block_ids=[[2]], request_id="r"),
        local_physical_block_ids=local_blocks,
    )
    info = SimpleNamespace(
        remote_tp_size=4, remote_physical_blocks_per_logical=1, remote_block_size=256
    )
    w.transfer_topo.get_engine_info = lambda eid: info
    w.transfer_topo.tp_ratio = lambda size: 1
    w._logical_to_kernel_block_ids = lambda blocks, ratio: blocks
    w.tp_mappings["old-p"] = SimpleNamespace(
        all_source_ranks=[0], source_ranks_per_group=[{0}]
    )
    w.src_xfer_handles_by_block_size = {256: 1}
    w.use_mla = True
    w._recving_metadata["r"] = meta
    calls = []

    def read(**kwargs):
        calls.append(kwargs)
        if not cached:
            w._recving_transfers["r"] = [42]

    w._read_blocks = read
    source_method("_read_blocks_for_req", patched=True, filename="pull_worker.py")(
        w, "r", meta
    )
    assert len(calls) == 1
    assert ("r" in w._recving_metadata) == (not cached)
    assert ("r" in w._recving_transfers) == (not cached)


def test_handshake_pending_blocks_retirement_until_publication_complete():
    w = Worker()
    future = Future()
    w._handshake_futures["old-p"] = future
    assert operate(w)["state"] == "busy"
    future.set_result(({}, 0.0))
    assert operate(w)["state"] == "busy"
    del w._handshake_futures["old-p"]
    assert operate(w)["state"] == "prepared"


def test_partial_failure_stays_failed_and_excludes_ttl_retry():
    w = Worker()
    operate(w)
    w.nixl_wrapper.fail = True
    assert operate(w, "commit")["state"] == "failed"
    assert operate(w, "commit")["state"] == "failed"
    w._evict_stale_engines()
    assert w.nixl_wrapper.calls == [("release", 11)]


def test_busy_between_prepare_and_commit_never_releases():
    w = Worker()
    operate(w)
    w._recving_metadata["unrelated-read"] = object()
    assert operate(w, "commit")["state"] == "busy"
    assert not w.nixl_wrapper.calls


def test_ttl_cannot_free_inflight_or_prepared_peer():
    w = Worker()
    w._recving_transfers["r"] = [1]
    w._evict_stale_engines()
    assert not w.nixl_wrapper.calls
    w._recving_transfers.clear()
    operate(w)
    w._evict_stale_engines()
    assert not w.nixl_wrapper.calls


def test_control_image_is_readonly(monkeypatch):
    monkeypatch.setenv("PEER_LAB_RETIRE_ENABLED", "0")
    w = Worker()
    assert peer_lab.snapshot(w)["peers"] == ["live-p", "old-p"]
    with pytest.raises(RuntimeError, match="disabled"):
        operate(w)


@pytest.mark.parametrize(
    "field,value",
    [("engine_id", "replacement-d"), ("_bidirectional_kv_xfer_enabled", True)],
)
def test_wrong_generation_or_direction_is_rejected(field, value):
    w = Worker()
    setattr(w, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        operate(w)


def test_partial_peer_map_is_not_absent_success():
    w = Worker()
    del w._remote_agents["old-p"]
    assert operate(w)["reason"] == "partial_peer_state"


def test_wrong_thread_cannot_cleanup():
    w = Worker()
    operate(w)
    errors = []

    def other_thread():
        try:
            operate(w, "commit")
        except RuntimeError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    assert errors and not w.nixl_wrapper.calls


@pytest.mark.parametrize("name", list(patcher.HASHES))
def test_patch_is_pinned_and_syntax_valid(name):
    original = (SOURCE / name).read_text()
    result = patcher.transform(name, original)
    ast.parse(result)
    with pytest.raises(RuntimeError, match="does not match"):
        patcher.transform(name, original + "\n")
    assert result.count("_handle_failed_transfer(req_id, None)") == original.count(
        "_handle_failed_transfer(req_id, None)"
    )


def test_fixture_source_mismatch_is_rejected_before_use(tmp_path):
    for name in patcher.HASHES:
        (tmp_path / name).write_bytes((SOURCE / name).read_bytes())
    path = tmp_path / "pull_worker.py"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Unexpected.*pull_worker.py"):
        checked_sources(tmp_path)


class Engine:
    def __init__(self):
        self.calls = []
        self.ranks = [0, 1, 2, 3]

    async def collective_rpc(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if method == "peer_lab_snapshot":
            return [
                {
                    "tp_rank": i,
                    "decode_engine_id": "decode-1",
                    "peers": ["old-p"],
                    "busy_reason": None,
                }
                for i in self.ranks
            ]
        target, op, decode, phase = kwargs["args"]
        return [
            {
                "tp_rank": i,
                "decode_engine_id": decode,
                "target_engine_id": target,
                "operation_id": op,
                "state": "prepared" if phase == "prepare" else "cleanup_returned",
            }
            for i in self.ranks
        ]


async def invoke(middleware, engine, body, path="/peer_lab/control", authorized=True):
    messages = []
    inbox = [
        {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
    ]

    async def receive():
        return inbox.pop(0) if inbox else {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [
            (
                b"authorization",
                ("Bearer " + ("x" * 40 if authorized else "bad")).encode(),
            )
        ],
        "app": SimpleNamespace(state=SimpleNamespace(engine_client=engine)),
    }
    await middleware(scope, receive, send)
    return messages[0]["status"], json.loads(messages[-1]["body"])


async def application(scope, receive, send):
    await receive()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send(
        {"type": "http.response.body", "body": b'{"ok":true}', "more_body": False}
    )


COMMAND = {
    "target_engine_id": "old-p",
    "operation_id": "op-1",
    "decode_engine_id": "decode-1",
}


def test_api_two_phase_and_retired_request_never_enters_engine():
    async def run():
        m, e = AdmissionFence(application), Engine()
        assert (await invoke(m, e, dict(COMMAND, phase="prepare")))[0] == 200
        assert (
            await invoke(
                m,
                e,
                {"kv_transfer_params": {"remote_engine_id": "new-p"}},
                "/v1/completions",
            )
        )[0] == 503
        assert (await invoke(m, e, dict(COMMAND, phase="commit")))[0] == 200
        before = len(e.calls)
        status, _ = await invoke(
            m,
            e,
            {"kv_transfer_params": {"remote_engine_id": "old-p"}},
            "/v1/completions",
        )
        assert status == 409 and len(e.calls) == before
        assert (
            await invoke(
                m,
                e,
                {"kv_transfer_params": {"remote_engine_id": "new-p"}},
                "/v1/completions",
            )
        )[0] == 200
        assert m.active == 0

    asyncio.run(run())


def test_disconnect_during_snapshot_is_rechecked():
    async def run():
        m = AdmissionFence(application)

        class RacingEngine(Engine):
            async def collective_rpc(self, method, **kwargs):
                result = await super().collective_rpc(method, **kwargs)
                m.uncertain_disconnect = True
                return result

        assert (await invoke(m, RacingEngine(), dict(COMMAND, phase="prepare")))[
            0
        ] == 409
        assert not m.fences

    asyncio.run(run())


def test_partial_commit_retains_full_admission_freeze():
    async def run():
        m, e = AdmissionFence(application), Engine()
        assert (await invoke(m, e, dict(COMMAND, phase="prepare")))[0] == 200
        e.ranks = [0, 1, 2]
        assert (await invoke(m, e, dict(COMMAND, phase="commit")))[0] == 409
        assert m.retirement == ("old-p", "op-1")
        assert (await invoke(m, e, {}, "/v1/completions"))[0] == 503
        other = dict(
            COMMAND, target_engine_id="live-p", operation_id="op-2", phase="prepare"
        )
        assert (await invoke(m, e, other))[0] == 409

    asyncio.run(run())


def test_prepare_retry_after_commit_does_not_freeze_forever():
    async def run():
        class CompletedEngine(Engine):
            async def collective_rpc(self, method, **kwargs):
                result = await super().collective_rpc(method, **kwargs)
                if method == "peer_lab_retire" and getattr(self, "completed", False):
                    for row in result:
                        row["state"] = "cleanup_returned"
                return result

        m, e = AdmissionFence(application), CompletedEngine()
        assert (await invoke(m, e, dict(COMMAND, phase="prepare")))[0] == 200
        assert (await invoke(m, e, dict(COMMAND, phase="commit")))[0] == 200
        e.completed = True
        status, result = await invoke(m, e, dict(COMMAND, phase="prepare"))
        assert status == 200 and result["state"] == "cleanup_returned"
        assert m.retirement is None
        assert (await invoke(m, e, {}, "/v1/completions"))[0] == 200

    asyncio.run(run())


@pytest.mark.parametrize("ranks", [[0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3, 4]])
def test_missing_duplicate_or_extra_rank_cannot_prepare(ranks):
    async def run():
        m, e = AdmissionFence(application), Engine()
        e.ranks = ranks
        assert (await invoke(m, e, dict(COMMAND, phase="prepare")))[0] == 409
        assert m.fences == {}
        assert all(c[0] == "peer_lab_snapshot" for c in e.calls)

    asyncio.run(run())


def test_management_requires_credential_and_zero_active():
    async def run():
        m, e = AdmissionFence(application), Engine()
        assert (await invoke(m, e, COMMAND, authorized=False))[0] == 401
        assert not e.calls
        m.active = 1
        assert (await invoke(m, e, dict(COMMAND, phase="prepare")))[0] == 409
        assert not m.fences

    asyncio.run(run())


@pytest.mark.parametrize(
    "path",
    ["/v1/responses", "/invocations", "/collective_rpc", "/v1/chat/completions/"],
)
def test_alternate_inference_or_dev_routes_cannot_bypass_fence(path):
    async def run():
        m, e = AdmissionFence(application), Engine()
        assert (await invoke(m, e, {}, path))[0] == 404
        assert not e.calls and m.active == 0

    asyncio.run(run())


def test_disconnected_response_blocks_normal_drain_experiment():
    async def incomplete(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": True})

    async def run():
        m, e = AdmissionFence(incomplete), Engine()
        await invoke(m, e, {}, "/v1/completions")
        assert m.uncertain_disconnect
        status, result = await invoke(m, e, dict(COMMAND, phase="prepare"))
        assert (
            status == 409 and result["reason"] == "aborted_request_cleanup_not_in_scope"
        )

    asyncio.run(run())


def test_new_active_request_during_snapshot_is_rechecked():
    async def run():
        m = AdmissionFence(application)

        class RacingEngine(Engine):
            async def collective_rpc(self, method, **kwargs):
                result = await super().collective_rpc(method, **kwargs)
                m.active = 1
                return result

        assert (await invoke(m, RacingEngine(), dict(COMMAND, phase="prepare")))[
            0
        ] == 409
        assert not m.fences

    asyncio.run(run())
