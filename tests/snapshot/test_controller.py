# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import stat
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.snapshot.server import (
    ListenerConfig,
    Oracle,
    SnapshotBarrierError,
    SnapshotCanaryError,
    oracle_from_request_output,
    parse_control_args,
    read_release_marker,
    run_snapshot_child,
    write_ready_atomic,
)


@pytest.mark.asyncio
async def test_snapshot_child_waits_before_http_bind():
    events: list[str] = []
    engine = object()

    @asynccontextmanager
    async def engine_context():
        events.append("engine-ready")
        yield engine

    async def run_canary(actual_engine: object) -> Oracle:
        assert actual_engine is engine
        return Oracle(token_ids=(12095,), text=" Paris")

    def write_ready(oracle: Oracle) -> None:
        assert oracle.token_ids == (12095,)
        events.append("ready-written")

    async def wait_for_release() -> None:
        events.append("barrier-wait")
        events.append("barrier-release")

    async def bind_and_serve(actual_engine: object) -> None:
        assert actual_engine is engine
        events.append("http-bound")

    await run_snapshot_child(
        engine_context=engine_context(),
        run_canary=run_canary,
        prepare_snapshot=lambda: events.append("streams-detached"),
        write_ready=write_ready,
        wait_for_release=wait_for_release,
        bind_and_serve=bind_and_serve,
    )

    assert events == [
        "engine-ready",
        "streams-detached",
        "ready-written",
        "barrier-wait",
        "barrier-release",
        "http-bound",
    ]


@pytest.mark.asyncio
async def test_empty_canary_prevents_http_bind():
    bound = False

    @asynccontextmanager
    async def engine_context():
        yield object()

    async def run_canary(_engine: object) -> Oracle:
        return Oracle(token_ids=(), text="")

    async def bind_and_serve(_engine: object) -> None:
        nonlocal bound
        bound = True

    with pytest.raises(SnapshotCanaryError, match="no token"):
        await run_snapshot_child(
            engine_context=engine_context(),
            run_canary=run_canary,
            prepare_snapshot=lambda: None,
            write_ready=lambda _oracle: None,
            wait_for_release=_return_none,
            bind_and_serve=bind_and_serve,
        )

    assert not bound


@pytest.mark.asyncio
async def test_barrier_failure_prevents_http_bind():
    bound = False

    @asynccontextmanager
    async def engine_context():
        yield object()

    async def wait_for_release() -> None:
        raise RuntimeError("malformed release marker")

    async def bind_and_serve(_engine: object) -> None:
        nonlocal bound
        bound = True

    with pytest.raises(RuntimeError, match="malformed release marker"):
        await run_snapshot_child(
            engine_context=engine_context(),
            run_canary=lambda _engine: _return_oracle(),
            prepare_snapshot=lambda: None,
            write_ready=lambda _oracle: None,
            wait_for_release=wait_for_release,
            bind_and_serve=bind_and_serve,
        )

    assert not bound


async def _return_none() -> None:
    return None


async def _return_oracle() -> Oracle:
    return Oracle(token_ids=(12095,), text=" Paris")


def test_ready_marker_is_atomic_and_private(tmp_path: Path):
    ready_path = tmp_path / "ready.json"

    write_ready_atomic(ready_path, Oracle(token_ids=(12095,), text=" Paris"))

    assert json.loads(ready_path.read_text()) == {
        "token_ids": [12095],
        "text": " Paris",
    }
    assert stat.S_IMODE(ready_path.stat().st_mode) == 0o600
    assert not (tmp_path / "ready.json.tmp").exists()


def test_release_marker_supplies_listener(tmp_path: Path):
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps({"release": True, "host": "127.0.0.1", "port": 9000})
    )

    assert read_release_marker(release_path) == ListenerConfig(
        host="127.0.0.1", port=9000
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"release": False, "host": "127.0.0.1", "port": 9000},
        {"release": True, "host": "127.0.0.1", "port": 0},
        {"release": True, "host": 4, "port": 9000},
    ],
)
def test_malformed_release_marker_is_rejected(
    tmp_path: Path, payload: dict[str, object]
):
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(payload))

    with pytest.raises(SnapshotBarrierError, match="release marker"):
        read_release_marker(release_path)


def test_request_output_becomes_exact_oracle():
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[12095], text=" Paris")]
    )

    assert oracle_from_request_output(output) == Oracle(
        token_ids=(12095,), text=" Paris"
    )


def test_child_control_flags_are_removed_from_vllm_args():
    control, remaining = parse_control_args(
        [
            "--ready-file",
            "ready.json",
            "--release-file",
            "release.json",
            "--release-timeout-s",
            "123",
            "--",
            "Qwen/Qwen3-0.6B",
            "--dtype",
            "float16",
        ]
    )

    assert control.ready_file == Path("ready.json")
    assert control.release_file == Path("release.json")
    assert control.release_timeout_s == 123
    assert remaining == ["Qwen/Qwen3-0.6B", "--dtype", "float16"]
