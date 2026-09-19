# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute selected, unmodified vLLM methods with a fake native boundary.

AST extraction avoids importing torch/CUDA on this Mac. The source SHA is
checked before execution; this is NOT a full vLLM or CUDA integration test.
"""

from __future__ import annotations
import __future__

import ast
import hashlib
import logging
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from peer_retirement import PEER_MAPS

BASE_COMMIT = "0136df94b0d75732be6c66f620f51289c888e41d"
SOURCE_SHA256 = "5f9a4f45edd38fe73eede0e8b7be8147b02c141cea2f2efc0f352e38c5380102"
SOURCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py"
)
METHOD_NAMES = {
    "_cleanup_remote_engine",
    "_evict_stale_engines",
    "_engines_with_inflight_transfers",
}


def load_worker_methods(path=SOURCE_PATH):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source changed: re-review the experiment's pinned methods")
    parsed = ast.parse(raw, filename=str(path))
    cls = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "NixlBaseConnectorWorker"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in METHOD_NAMES
    ]
    assert {node.name for node in methods} == METHOD_NAMES
    namespace = {"time": time, "logger": logging.getLogger("vllm_cpu_experiment")}
    module = ast.Module(body=methods, type_ignores=[])
    code = compile(
        module,
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    return type("ExtractedVllmWorker", (), {n: namespace[n] for n in METHOD_NAMES})


@dataclass
class SimulatedAllocation:
    """Reference-count teaching model, NOT NVIDIA's allocation implementation."""

    owner_alive: bool = True
    imports: set = field(default_factory=set)

    def owner_exit(self):
        self.owner_alive = False

    @property
    def resident(self):
        return self.owner_alive or bool(self.imports)


class FakeNative:
    def __init__(self, identity, delayed_close=False):
        self.identity = identity
        self.delayed_close = delayed_close
        self.allocations = {}
        self.descriptors = {}
        self.pending_closes = set()
        self.events = []
        self.fail_on = None

    def register(self, engine_id, allocation, descriptors, agents):
        for handle in descriptors.values():
            self.descriptors[handle] = engine_id
        for agent in agents.values():
            self.allocations[agent] = (engine_id, allocation)
            allocation.imports.add((self.identity, agent))

    def release_dlist_handle(self, handle):
        self.events.append(("release_dlist", handle))
        if self.fail_on == ("release_dlist", handle):
            raise RuntimeError("Injected descriptor release failure")
        del self.descriptors[handle]

    def remove_remote_agent(self, agent):
        self.events.append(("remove_agent", agent))
        if self.fail_on == ("remove_agent", agent):
            raise RuntimeError("Injected remote agent removal failure")
        engine_id, _ = self.allocations[agent]
        assert engine_id not in self.descriptors.values()
        self.pending_closes.add(agent)
        if not self.delayed_close:
            self.progress()

    def progress(self):
        for agent in tuple(self.pending_closes):
            _, allocation = self.allocations.pop(agent)
            allocation.imports.remove((self.identity, agent))
            self.pending_closes.remove(agent)


def add_peer(worker, engine_id, allocation):
    agents = {(0, 0): f"{engine_id}-agent0", (0, 1): f"{engine_id}-agent1"}
    handles = {0: f"{engine_id}-dlist0", 1: f"{engine_id}-dlist1"}
    for name in PEER_MAPS:
        getattr(worker, name)[engine_id] = {"sentinel": engine_id}
    worker._remote_agents[engine_id] = agents
    worker.dst_xfer_side_handles[engine_id] = handles
    worker._engine_last_active[engine_id] = time.perf_counter()
    worker.nixl_wrapper.register(engine_id, allocation, handles, agents)
    worker._topology_peers.add(engine_id)


def make_worker(identity="d0-rank0", allocation=None, delayed_close=False):
    worker = load_worker_methods()()
    for name in PEER_MAPS:
        setattr(worker, name, {})
    worker._handshake_lock = threading.RLock()
    worker._handshake_futures = {}
    worker._ready_requests = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
    worker._recving_metadata = {}
    worker._recving_transfers = defaultdict(list)
    worker._pending_recv_notifs = {}
    worker._reqs_to_send = {}
    worker._reqs_to_process = set()
    worker._engine_ttl = 3600.0
    worker.src_xfer_handles_by_block_size = {16: "local-source-do-not-release"}
    worker._topology_peers = set()
    worker.transfer_topo = SimpleNamespace(
        unregister_remote_engine=worker._topology_peers.remove
    )
    worker.nixl_wrapper = FakeNative(identity, delayed_close=delayed_close)
    if allocation is None:
        allocation = SimulatedAllocation()
    add_peer(worker, "p-old-generation", allocation)
    return worker, allocation


def recv_meta(engine_id="p-old-generation"):
    return SimpleNamespace(remote=SimpleNamespace(engine_id=engine_id))


if __name__ == "__main__":
    from peer_retirement import PeerRetirement

    worker, allocation = make_worker(delayed_close=True)
    allocation.owner_exit()
    print("SIMULATION ONLY: no GPU, no vLLM server, no native NIXL loaded")
    print(f"P exited; simulated allocation resident: {allocation.resident}")
    result = PeerRetirement(worker).drop_peer("p-old-generation")
    print(f"D Python cleanup result: {result.state}")
    print(f"Before simulated native close: {allocation.resident}")
    worker.nixl_wrapper.progress()
    print(f"After simulated native close: {allocation.resident}")
    print(f"Real native release verified: {result.native_release_verified}")
