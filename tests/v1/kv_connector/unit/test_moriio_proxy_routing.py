# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-fair request routing in the MoRIIO toy P/D proxy.

Exercises the REAL ``flat_interleaved_dp_route`` from the toy proxy (loaded from
its ``examples/`` path with heavy deps stubbed) — no routing logic is copied
here. This is the request-distribution half of end-to-end fairness: the proxy
must hand every prefill/decode (instance, dp_rank) slot an equal share so no GPU
is starved. The connector-side read routing is covered in
``test_moriio_routing_fairness.py``.

Regression target: the previous scheme derived ``instance = req % n_instances``
and ``dp_rank = req % dp_size`` from the same counter, so when
``n_instances | dp_size`` each instance was locked to a stride-``n`` subset of
its ranks — e.g. 2 prefill instances x DP8 stranded 4 of every node's 8 GPUs.
``flat_interleaved_dp_route`` walks ONE counter over the full
``(instance, dp_rank)`` slot space, so the two selections can never alias.

Role shapes below are exactly those in the RFC (#46107) deployments:
    (1, 1) 1P/1D TP8      (2, 1) 2P/2D TP8      (4, 1) 4D TP8
    (1, 8) 1D DP8EP       (2, 8) 2P DP8EP       (3, 8) 3D DP8EP
"""

import contextlib
import ctypes
import importlib.util
import os
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import cast

import msgpack
import pytest
import zmq

_MISSING = object()

PROXY_REL = "examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py"


def _module(name, **attrs):
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _package(name):
    module = _module(name)
    module.__path__ = []
    return module


class _QuartStub:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        return lambda fn: fn

    def post(self, *a, **k):
        return self.route()


async def _make_response_stub(value):
    return value


@contextlib.contextmanager
def _proxy_import_stubs():
    """Stub the proxy's external deps so the module imports for a pure unit test.

    Only third-party/vllm imports are stubbed; the routing function under test
    is executed as-is from the real module.
    """
    common = "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"

    class _MoRIIOConstants:
        TRANSFER_PREFIX = "moriio-transfer"

    stubs = {
        "aiohttp": _module("aiohttp"),
        "msgpack": _module("msgpack"),
        "zmq": _module("zmq"),
        "quart": _module(
            "quart",
            Quart=_QuartStub,
            Request=object,
            make_response=_make_response_stub,
            request=object(),
        ),
        "vllm": _package("vllm"),
        "vllm.distributed": _package("vllm.distributed"),
        "vllm.distributed.kv_transfer": _package("vllm.distributed.kv_transfer"),
        "vllm.distributed.kv_transfer.kv_connector": _package(
            "vllm.distributed.kv_transfer.kv_connector"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1": _package(
            "vllm.distributed.kv_transfer.kv_connector.v1"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio": _package(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio"
        ),
        common: _module(common, MoRIIOConstants=_MoRIIOConstants),
    }
    saved = {}
    for name, module in stubs.items():
        if name not in sys.modules:
            saved[name] = _MISSING
            sys.modules[name] = module
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = cast(types.ModuleType, previous)


def _load_proxy_module():
    path = Path(__file__).parents[4] / PROXY_REL
    spec = importlib.util.spec_from_file_location("moriio_proxy_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _proxy_import_stubs():
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def route():
    return _load_proxy_module().flat_interleaved_dp_route


def _instances(n: int, dp_size: int):
    # Only dp_size is read by the router; tp_size carried for realism.
    return [{"dp_size": dp_size, "tp_size": 8 // dp_size} for _ in range(n)]


# (n_instances, dp_size) for every distinct P/D role shape in the RFC configs.
ROLE_SHAPES = [(1, 1), (2, 1), (4, 1), (1, 8), (2, 8), (3, 8)]
SHAPE_IDS = [f"n{n}_dp{dp}" for n, dp in ROLE_SHAPES]


def _route_n(route, instances, count):
    # Proxy uses 1-indexed request numbers (slot = (request_number - 1) % ...).
    return [route(rn, instances) for rn in range(1, count + 1)]


@pytest.mark.parametrize(("n", "dp"), ROLE_SHAPES, ids=SHAPE_IDS)
def test_full_slot_space_is_covered_uniformly(route, n, dp):
    instances = _instances(n, dp)
    period = n * dp
    # Three full cycles -> every (instance, dp_rank) slot must be hit the same
    # number of times (exactly uniform, no starved slot, no aliasing).
    hits = Counter(_route_n(route, instances, period * 3))

    expected: set[tuple[int, int | None]]
    if dp == 1:
        expected = {(inst, None) for inst in range(n)}
    else:
        expected = {(inst, r) for inst in range(n) for r in range(dp)}
    assert set(hits) == expected, f"missing slots: {expected - set(hits)}"
    assert max(hits.values()) == min(hits.values())


@pytest.mark.parametrize(("n", "dp"), ROLE_SHAPES, ids=SHAPE_IDS)
def test_instance_and_dp_rank_marginals_are_balanced(route, n, dp):
    instances = _instances(n, dp)
    routed = _route_n(route, instances, n * dp * 5)

    inst_counts = Counter(inst for inst, _ in routed)
    assert set(inst_counts) == set(range(n))
    assert max(inst_counts.values()) == min(inst_counts.values())

    dp_counts = Counter(r for _, r in routed)
    if dp == 1:
        assert set(dp_counts) == {None}
    else:
        assert set(dp_counts) == set(range(dp))
        assert max(dp_counts.values()) == min(dp_counts.values())


def test_tp_instance_forwards_no_dp_rank(route):
    # dp_size == 1 (a TP instance) must yield dp_rank None so the proxy never
    # forwards an out-of-range data-parallel rank.
    assert all(r is None for _, r in _route_n(route, _instances(2, 1), 8))


def test_two_instance_dp8_gives_every_node_all_ranks(route):
    # Direct regression for the stranded-GPU bug: with 2 prefill instances x DP8
    # each instance must receive ALL 8 dp ranks (16 distinct slots), not a
    # stride-2 subset of 4.
    routed = _route_n(route, _instances(2, 8), 2 * 8)
    per_instance: dict[int, set] = {0: set(), 1: set()}
    for inst, dp_rank in routed:
        per_instance[inst].add(dp_rank)
    assert per_instance[0] == set(range(8))
    assert per_instance[1] == set(range(8))


def test_consecutive_requests_alternate_instances(route):
    # Interleaved order spreads consecutive requests across instances rather
    # than filling one instance's ranks before moving on.
    insts = [inst for inst, _ in _route_n(route, _instances(3, 8), 3)]
    assert insts == [0, 1, 2]


@pytest.mark.skipif(sys.platform != "linux", reason="GIL probe uses libc.usleep")
def test_discovery_heartbeat_progresses_while_worker_holds_gil():
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_heartbeat import (
        MoRIIOHeartbeat,
    )

    payload = {
        "type": "P",
        "http_address": "127.0.0.1:8000",
        "zmq_address": "host:127.0.0.1,handshake:6301,notify:61005",
        "dp_size": 1,
        "tp_size": 8,
        "transfer_mode": "READ",
    }
    with zmq.Context() as context, context.socket(zmq.ROUTER) as socket:
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        heartbeat = MoRIIOHeartbeat(f"tcp://127.0.0.1:{port}", payload, 0.02)
        try:
            assert socket.poll(5000), "Discovery did not register"
            assert msgpack.unpackb(socket.recv_multipart()[1]) == payload
            # PyDLL retains the GIL throughout the native call.
            ctypes.PyDLL(None).usleep(300_000)
            received = 0
            while socket.poll(0):
                assert msgpack.unpackb(socket.recv_multipart()[1]) == payload
                received += 1
            assert received >= 3, "Discovery stalled while the worker held the GIL"
        finally:
            heartbeat.shutdown()
            heartbeat.shutdown()
        while socket.poll(0):
            socket.recv_multipart()
        assert not socket.poll(100), "Heartbeats continued after shutdown"


@pytest.mark.skipif(sys.platform != "linux", reason="Checks child exit via procfs")
def test_discovery_heartbeat_exits_when_worker_dies():
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_heartbeat

    owner_code = """
import runpy
import sys
Heartbeat = runpy.run_path(sys.argv[1])["MoRIIOHeartbeat"]
heartbeat = Heartbeat(sys.argv[2], {"type": "D"}, 0.02)
print(heartbeat._process.pid, flush=True)
sys.stdin.read()
"""
    with zmq.Context() as context, context.socket(zmq.ROUTER) as socket:
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        owner = subprocess.Popen(
            [
                sys.executable,
                "-c",
                owner_code,
                moriio_heartbeat.__file__,
                f"tcp://127.0.0.1:{port}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        child_pid = None
        child_exited = False
        try:
            assert socket.poll(5000), "Discovery did not register"
            assert owner.stdout is not None
            child_pid = int(owner.stdout.readline())
            owner.kill()
            owner.wait(timeout=5)
            stat = Path(f"/proc/{child_pid}/stat")
            deadline = time.monotonic() + 5
            while stat.exists():
                # A terminated child may await reaping by the container's init.
                try:
                    if stat.read_text().split(") ")[1].startswith("Z "):
                        break
                except FileNotFoundError:
                    break
                assert time.monotonic() < deadline, "Orphan heartbeat survived worker"
                time.sleep(0.01)
            child_exited = True
        finally:
            if owner.poll() is None:
                owner.kill()
            owner.communicate(timeout=5)
            if child_pid is not None and not child_exited:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child_pid, 9)
