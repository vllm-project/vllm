# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the elastic fault-tolerance framework.

Requires nixl_ep FT hardware; gated behind ``has_nixl_ep()``.
"""

import contextlib
import os
import threading
import time

import psutil
import pytest
import requests

from tests.utils import multi_gpu_test
from vllm.utils.import_utils import has_nixl_ep

MODEL_NAME = os.getenv("MODEL_NAME", "ibm-research/PowerMoE-3b")
DP_SIZE = int(os.getenv("DP_SIZE", "2"))

# Fault-detection timeout budget:
# - CPU: Gloo DP allreduce timeout (10s) detects the dead peer.
# - nixl_ep: kernel masks the dead rank after Buffer's default timeout_ms=30000 (30s).
# - Deadline (45s): slowest fallback (30s) + margin.
CPU_DISTRIBUTED_TIMEOUT_S = 10
FAULT_DETECTION_DEADLINE_S = 45


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


def _ft_server_args() -> list[str]:
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
    ]


def _ft_manager():
    """Build the shared DP+EP fault-tolerant server topology (one engine/server)."""
    from tests.v1.distributed.test_external_lb_dp import ExternalLBServerManager

    return ExternalLBServerManager(
        MODEL_NAME,
        DP_SIZE,
        api_server_count=1,  # FT requires a single API server per engine
        base_server_args=_ft_server_args(),
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


def _get_ft_status(server) -> dict:
    resp = requests.get(server.url_for("fault_tolerance/status"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _assert_serving_and_healthy(server) -> dict:
    """Serve one request and assert every engine reports healthy. Returns status."""
    _complete(server.get_client())
    status = _get_ft_status(server)
    assert all(e["status"] == "healthy" for e in status["engines"]), status
    return status


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


def _poll_status(server, deadline_s: int, predicate):
    """Poll ``/fault_tolerance/status`` until ``predicate(engine)`` is true.

    Returns the first matching engine dict, or ``None`` on timeout. Request
    errors are suppressed so a briefly-unreachable server doesn't abort the poll.
    """
    start = time.time()
    while time.time() - start < deadline_s:
        with contextlib.suppress(Exception):
            for engine in _get_ft_status(server)["engines"]:
                if predicate(engine):
                    return engine
        time.sleep(1.0)
    return None


def _wait_for_status(server, statuses, deadline_s: int = FAULT_DETECTION_DEADLINE_S):
    """Poll until an engine reports one of ``statuses`` (a set of status strings)."""
    return _poll_status(server, deadline_s, lambda e: e["status"] in statuses)


@contextlib.contextmanager
def _driving(*servers):
    """Pump completions at each server in the background for the block's duration.

    Keeps every engine stepping into its failed component so a fault surfaces,
    and lets each ``retry`` reach its cross-rank collective. Errors are expected
    once faulted and are ignored.
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
    """Wait until ``/status`` records the outcome of the given FT apply request."""
    engine = _poll_status(
        server,
        deadline_s,
        lambda engine: engine.get("last_ft_request_id") == request_id,
    )
    return engine.get("ft_error") if engine else None


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

    with _ft_manager() as servers:
        assert len(servers) == DP_SIZE
        rank0 = _server_for_rank(servers, 0)
        rank1 = _server_for_rank(servers, 1)

        # 1. Both engines healthy and serving.
        for server in (rank0, rank1):
            status = _assert_serving_and_healthy(server)
            assert status["schema_version"] == 1, status
            assert status["total_engines"] == 1, status  # one engine per server

        # 2. Drive both ranks so rank 1 accumulates execute_model steps and trips
        #    the injected fault; rank 0 then times out on the DP allreduce.
        with _driving(rank0, rank1):
            faulted = {
                rank: _wait_for_status(server, {"unhealthy"})
                for rank, server in ((0, rank0), (1, rank1))
            }

        for rank, engine in faulted.items():
            assert engine is not None, (
                f"rank {rank} did not report UNHEALTHY within "
                f"{FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
            )
        # The rank that raised carries the fault info from its own exception.
        assert faulted[1].get("fault_info"), faulted[1]

        # 3. retry both engines.
        for server in (rank0, rank1):
            _apply_ft(server, "retry")

        # 4. Recovery completes: both engines return to healthy.
        for rank, server in ((0, rank0), (1, rank1)):
            healthy = _wait_for_status(server, {"healthy"})
            assert healthy is not None, (
                f"rank {rank} did not recover to healthy within "
                f"{FAULT_DETECTION_DEADLINE_S}s"
            )

        # 5. Post-recovery inference works on both ranks.
        for server in (rank0, rank1):
            completion = _complete(server.get_client())
            assert completion.choices[0].text is not None


@pytest.mark.skipif(not has_nixl_ep(), reason="Requires nixl_ep all2all backend")
@multi_gpu_test(num_gpus=2)
def test_worker_kill_survivor_unhealthy_and_dead_rejects_retry():
    """One worker kill surfaces two status transitions at once.

    SIGKILLing only rank 1's worker leaves both EngineCores alive, so the same
    fault is seen two ways:

    - Survivor (rank 0): detects the dead peer via Gloo allreduce / nixl_ep
      kernel timeout. Its own executor is fine, so ``on_fault`` marks it
      UNHEALTHY with a ``fault_info``.
    - Victim (rank 1): detects its own executor failure and marks itself DEAD.

    Recovery is gated on UNHEALTHY: the DEAD engine accepts ``retry`` at the
    HTTP layer (202 = background dispatch) but rejects it in the engine,
    recording the reason as ``ft_error``.
    """
    with _ft_manager() as servers:
        assert len(servers) == DP_SIZE
        survivor = _server_for_rank(servers, 0)
        victim = _server_for_rank(servers, 1)

        # 1. Confirm both engines are healthy and serving.
        for server in (survivor, victim):
            _assert_serving_and_healthy(server)

        # 2. Kill only the victim's worker; both EngineCores stay alive.
        _kill_worker_process(victim)

        # 3. Drive both engines so each keeps stepping into the failed component.
        #    Both fault ~together via timeout, so sequential polling under one
        #    driving context still sees each within the deadline.
        with _driving(survivor, victim):
            survivor_faulted = _wait_for_status(survivor, {"dead", "unhealthy"})
            victim_faulted = _wait_for_status(victim, {"dead", "unhealthy"})

        assert survivor_faulted is not None, (
            "survivor did not report the peer fault within "
            f"{FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
        )
        # The survivor's own executor is fine, so it must be UNHEALTHY, not DEAD.
        assert survivor_faulted["status"] == "unhealthy", survivor_faulted
        assert survivor_faulted.get("fault_info"), survivor_faulted

        assert victim_faulted is not None, (
            "victim did not report its worker's death within "
            f"{FAULT_DETECTION_DEADLINE_S}s"
        )
        assert victim_faulted["status"] == "dead", victim_faulted

        # 4. retry is accepted at the HTTP layer (202 = background dispatch)...
        request_id = _apply_ft(victim, "retry")["request_id"]

        # 5. ...but the DEAD engine must reject it: recovery requires UNHEALTHY.
        ft_error = _wait_for_ft_apply_outcome(
            victim, request_id, FAULT_DETECTION_DEADLINE_S
        )
        assert ft_error is not None, (
            "rejection was never recorded in /fault_tolerance/status"
        )
        assert "status is DEAD" in ft_error, ft_error
