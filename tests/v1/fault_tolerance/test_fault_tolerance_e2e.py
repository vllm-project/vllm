# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the elastic fault-tolerance framework.

Requires nixl_ep FT hardware; gated behind ``has_nixl_ep()``.
"""

import contextlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil
import pytest
import requests

from tests.utils import RemoteOpenAIServer, multi_gpu_test
from vllm.utils.import_utils import has_nixl_ep

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V2-Lite-Chat")

# Fault-detection timeout budget:
# - CPU: Gloo DP allreduce timeout (30s) detects the dead peer.
# - nixl_ep: kernel masks the dead rank after Buffer's default timeout_ms=30000 (30s).
# - Deadline (45s): slowest fallback (30s) + margin.
CPU_DISTRIBUTED_TIMEOUT_S = 30
FAULT_DETECTION_DEADLINE_S = 45


NUM_REDUNDANT_EXPERTS = 32

# Recovery after scale_down (mask + expert redistribution + weight reload)
# is expected to finish well within this deadline; the busy-loop wrapper
# itself waits engine_recovery_timeout_sec (120s) before giving up.
SCALE_DOWN_DEADLINE_S = 20


# Patches ``gpu.dp_utils.sync_cudagraph_and_dp_padding`` to raise on ``rank`` at
# a chosen step. Gated on VLLM_FT_TEST_INJECT_FAULT.
_FAULT_INJECT_SITECUSTOMIZE = """\
import builtins
import os
import sys

_SPEC = os.environ.get("VLLM_FT_TEST_INJECT_FAULT")
_MODULE = "vllm.v1.worker.gpu.dp_utils"
_ATTR = "sync_cudagraph_and_dp_padding"

if _SPEC:
    _f = dict(kv.split("=", 1) for kv in _SPEC.split(","))
    _RANK, _STEP = int(_f["rank"]), int(_f["step"])
    _steps = [0]

    def _patch(m):
        import inspect
        _orig = getattr(m, _ATTR)
        _sig = inspect.signature(_orig)
        def _wrapped(*args, **kwargs):
            result = _orig(*args, **kwargs)
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            dp_rank = bound.arguments.get("dp_rank")
            if dp_rank == _RANK:
                _steps[0] += 1
                if _steps[0] == _STEP:
                    raise RuntimeError(
                        "FT test fault injection (rank=%d step=%d)" % (_RANK, _STEP)
                    )
            return result

        setattr(m, _ATTR, _wrapped)

    _real_import = builtins.__import__

    def _hook(name, *a, **k):
        module = _real_import(name, *a, **k)
        m = sys.modules.get(_MODULE)
        # During vLLM's circular import the module lands in sys.modules before
        # its functions are defined; hasattr guards against patching too early.
        if (
            m is not None
            and hasattr(m, _ATTR)
            and not getattr(m, "_ft_patched", False)
        ):
            m._ft_patched = True
            _patch(m)
        return module

    builtins.__import__ = _hook
"""


def _install_fault_injection(monkeypatch, tmp_path, rank: int, step: int) -> None:
    """Arrange for the DP-sync fn to raise on ``rank`` at serving ``step``.

    Writes a ``sitecustomize.py`` and prepends its dir to PYTHONPATH so every
    vLLM subprocess picks it up; the fault spec is read from the environment.
    """
    site_dir = tmp_path / "ft_inject"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(_FAULT_INJECT_SITECUSTOMIZE)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(site_dir) + (os.pathsep + existing if existing else ""),
    )
    monkeypatch.setenv("VLLM_FT_TEST_INJECT_FAULT", f"rank={rank},step={step}")


def _ft_server_args(extra_args: list[str] | None = None) -> list[str]:
    return [
        "--enforce-eager",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enable-expert-parallel",
        "--all2all-backend",
        "nixl_ep",
        "--enable-fault-tolerance",
        "--cpu-distributed-timeout-seconds",
        str(CPU_DISTRIBUTED_TIMEOUT_S),
        "--fault-tolerance-config",
        '{"engine_recovery_timeout_sec": 120}',
        *(extra_args or []),
    ]


def _ft_manager(extra_args: list[str] | None = None, dp_size: int = 2):
    """Build the shared DP+EP fault-tolerant server topology (one engine/server)."""
    from tests.v1.distributed.test_external_lb_dp import ExternalLBServerManager

    return ExternalLBServerManager(
        MODEL_NAME,
        dp_size,
        api_server_count=1,  # FT requires a single API server per engine
        base_server_args=_ft_server_args(extra_args),
        tp_size=1,
    )


def _server_for_rank(servers, rank: int):
    """Locate the server for a DP rank."""
    for server, sargs in servers:
        if "--data-parallel-rank" in sargs:
            idx = sargs.index("--data-parallel-rank")
            if int(sargs[idx + 1]) == rank:
                return server
    raise AssertionError(f"no server found for DP rank {rank}")


def _complete(client):
    """Issue the one standard completion the tests use everywhere."""
    return client.completions.create(
        model=MODEL_NAME,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
        timeout=10.0,
    )


def _in_parallel(fn, servers) -> list:
    """Run ``fn(server)`` for all servers concurrently; return results in order."""
    with ThreadPoolExecutor(max_workers=len(servers)) as ex:
        return list(ex.map(fn, servers))


def _get_ft_status(server) -> dict:
    resp = requests.get(server.url_for("fault_tolerance/status"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _assert_serving_and_healthy(servers) -> None:
    """Wait until every engine is healthy, then serve one request per server."""
    healthy = _wait_for_engines(
        list(servers), match_key="status", match_values={"healthy"}
    )
    assert all(healthy), healthy
    _in_parallel(lambda s: _complete(s.get_client()), servers)


def _apply_ft(server, instruction: str, params: dict | None = None) -> dict:
    """POST an FT instruction; assert it is accepted (202) and return the body."""
    resp = requests.post(
        server.url_for("fault_tolerance/apply"),
        json={"instruction": instruction, "params": params or {}},
        timeout=10,
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


def _kill_worker_process(server) -> None:
    """SIGKILL only the worker proc, leaving EngineCore and API server alive."""
    workers = [
        p
        for p in psutil.Process(server.proc.pid).children(recursive=True)
        if "Worker" in " ".join(p.cmdline())
    ]
    assert len(workers) == 1, f"expected 1 worker proc, found: {workers}"
    workers[0].kill()


def _wait_for_engines(
    servers: list[RemoteOpenAIServer],
    match_key: str,
    match_values: set[str],
    deadline_s: int = FAULT_DETECTION_DEADLINE_S,
) -> list[dict[str, Any] | None]:
    """Poll ``/fault_tolerance/status`` until each server's engine status matches.

    A server matches when its engine-status dict has ``match_key`` equal to
    one of ``match_values``. Returns one engine-status dict per server. Servers still
    unmatched after ``deadline_s`` get None.
    """
    results: dict[int, dict[str, Any]] = {}
    pending = dict(enumerate(servers))
    start = time.time()
    while pending and time.time() - start < deadline_s:
        for i, server in list(pending.items()):
            with contextlib.suppress(Exception):
                for engine_status in _get_ft_status(server)["engines"]:
                    if engine_status.get(match_key) in match_values:
                        results[i] = engine_status
                        del pending[i]
                        break
        if pending:
            time.sleep(1.0)
    return [results.get(i) for i in range(len(servers))]


@contextlib.contextmanager
def _driving(*servers):
    """Pump completions at each server in the background for the block's duration.

    Keeps every engine stepping into its failed component so a fault surfaces.
    Errors are expected once faulted and are ignored.
    """
    stop = threading.Event()

    def _drive(server):
        client = server.get_client()
        while not stop.is_set():
            with contextlib.suppress(Exception):
                _complete(client)
            time.sleep(0.2)

    threads = [threading.Thread(target=_drive, args=(s,), daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        yield
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)


def _wait_for_ft_apply_outcome(server, request_id: str, deadline_s: int) -> str | None:
    """Wait until ``/fault_tolerance/status`` records the FT apply outcome."""
    engine_status = _wait_for_engines(
        [server],
        match_key="last_ft_request_id",
        match_values={request_id},
        deadline_s=deadline_s,
    )[0]
    return engine_status.get("ft_error") if engine_status else None


def _servers_by_rank(servers, dp_size: int) -> dict[int, RemoteOpenAIServer]:
    """Map each DP rank to its server, so tests can index by rank not position."""
    return {r: _server_for_rank(servers, r) for r in range(dp_size)}


def _drive_to_faulted(
    servers_by_rank: dict[int, RemoteOpenAIServer],
    match_values: set[str],
) -> dict[int, dict[str, Any]]:
    """Drive every engine until each reports a matching fault status.

    Returns a ``{rank: engine_status}`` map. Asserts no rank hung (all matched
    within the deadline) so callers can assume every value is present.
    """
    ranks = sorted(servers_by_rank)
    all_servers = [servers_by_rank[r] for r in ranks]
    with _driving(*all_servers):
        statuses = _wait_for_engines(
            all_servers, match_key="status", match_values=match_values
        )
    faulted: dict[int, dict[str, Any]] = {}
    for rank, engine_status in zip(ranks, statuses):
        assert engine_status is not None, (
            f"rank {rank} did not report fault within "
            f"{FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
        )
        faulted[rank] = engine_status
    return faulted


@pytest.mark.skipif(not has_nixl_ep(), reason="Requires nixl_ep all2all backend")
@multi_gpu_test(num_gpus=2)
def test_injected_fault_retry_recovers_all_ranks(monkeypatch, tmp_path):
    """An exception injected into the inference path drives full retry recovery.

    Injecting an exception into ``sync_cudagraph_and_dp_padding`` at a chosen
    step on rank 1.

    - Rank 1 raises inside the busy loop and goes UNHEALTHY.
    - Rank 0 detects the now-absent peer via the communication timeout and also
      goes UNHEALTHY.

    Both being UNHEALTHY is the precondition for ``retry``. The fault is patched
    into the DP-sync fn from the test (via a generated ``sitecustomize``).
    """
    fault_step = int(os.getenv("FT_FAULT_STEP", "50"))
    _install_fault_injection(monkeypatch, tmp_path, rank=1, step=fault_step)

    dp_size = 2
    faulted_rank = 1
    with _ft_manager(dp_size=dp_size) as servers:
        assert len(servers) == dp_size
        servers_by_rank = _servers_by_rank(servers, dp_size)

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(list(servers_by_rank.values()))

        # 2. Drive both ranks so the injected rank accumulates execute_model steps
        #    and trips the fault; the others then time out on the DP allreduce.
        faulted = _drive_to_faulted(servers_by_rank, match_values={"unhealthy"})

        # The rank that raised carries the fault info from its own exception.
        assert faulted[faulted_rank].get("fault_info"), faulted[faulted_rank]

        # 3. retry every engine.
        for server in servers_by_rank.values():
            _apply_ft(server, "retry")

        # 4. Recovery completes: every engine returns to healthy and serves again.
        _assert_serving_and_healthy(list(servers_by_rank.values()))


@pytest.mark.skipif(not has_nixl_ep(), reason="Requires nixl_ep all2all backend")
@multi_gpu_test(num_gpus=4)
def test_scale_down_removes_dead_rank_and_recovers():
    """scale_down removes a dead DP rank; survivors keep serving with a smaller
    DP group.

    SIGKILLing rank 1's worker leaves its EngineCore DEAD while the other ranks
    detect the peer fault and go UNHEALTHY. The orchestrator then sends
    ``scale_down`` with ``removed_dp_ranks=[1]`` to every survivor.


    Also verifies that a DEAD engine rejects ``retry``: recovery is gated on
    UNHEALTHY, so trying ``retry`` on the victim records a rejection reason.
    """
    eplb_args = [
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        str(NUM_REDUNDANT_EXPERTS),
    ]
    dp_size = 4
    victim_rank = 1
    with _ft_manager(eplb_args, dp_size=dp_size) as servers:
        assert len(servers) == dp_size
        servers_by_rank = _servers_by_rank(servers, dp_size)
        victim = servers_by_rank[victim_rank]
        survivor_ranks = [r for r in servers_by_rank if r != victim_rank]
        survivors = [servers_by_rank[r] for r in survivor_ranks]

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(list(servers_by_rank.values()))

        # 2. Kill the victim's worker; drive all engines into the fault.
        _kill_worker_process(victim)
        faulted = _drive_to_faulted(servers_by_rank, match_values={"dead", "unhealthy"})

        # 3. DEAD engine rejects retry: recovery requires UNHEALTHY.
        assert faulted[victim_rank]["status"] == "dead", faulted[victim_rank]
        request_id = _apply_ft(victim, "retry")["request_id"]
        ft_error = _wait_for_ft_apply_outcome(
            victim, request_id, FAULT_DETECTION_DEADLINE_S
        )
        assert ft_error is not None, (
            "rejection was never recorded in /fault_tolerance/status"
        )
        assert "status is DEAD" in ft_error, ft_error

        # 4. scale_down sent to every survivor: remove the dead rank.
        for server in survivors:
            _apply_ft(server, "scale_down", {"removed_dp_ranks": [victim_rank]})

        # 5. Recovery completes: all survivors are healthy and serving.
        recovered = _wait_for_engines(
            survivors,
            match_key="status",
            match_values={"healthy"},
            deadline_s=SCALE_DOWN_DEADLINE_S,
        )
        for rank, engine_status in zip(survivor_ranks, recovered):
            assert engine_status is not None, (
                f"survivor {rank} did not recover within {SCALE_DOWN_DEADLINE_S}s "
                "-- expert reload or DP-group reinit likely failed"
            )
        # Verify all survivors actually serve.
        for server in survivors:
            completion = _complete(server.get_client())
            assert completion.choices[0].text
