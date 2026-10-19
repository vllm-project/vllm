# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end coverage of graph-discard sleep and idle recapture."""

import asyncio
import json
import os
import time

import pytest

from tests.utils import multi_gpu_test
from vllm import LLM, SamplingParams


def _graph_snapshot(worker):
    import weakref

    import torch

    manager = worker.model_runner.cudagraph_manager
    if not hasattr(worker, "_test_graph_replays"):
        worker._test_graph_replays = 0
        worker._test_graph_objects = weakref.WeakSet()
        worker._test_recapture_streams = set()
        capture_one = worker.model_runner.capture_one_sleep_graph
        forward = worker.model_runner.model.forward
        worker._test_recapturing = False
        worker._test_host_cache_flushes = 0
        empty_host_cache = torch._C._host_emptyCache

        def observe_host_cache_flush():
            if worker._test_recapturing:
                worker._test_host_cache_flushes += 1
            return empty_host_cache()

        torch._C._host_emptyCache = observe_host_cache_flush

        def observe_forward(*args, **kwargs):
            if worker._test_recapturing:
                worker._test_recapture_streams.add(
                    torch.cuda.current_stream().cuda_stream
                )
            return forward(*args, **kwargs)

        def observe_capture():
            worker._test_recapturing = True
            try:
                return capture_one()
            finally:
                worker._test_recapturing = False

        worker.model_runner.model.forward = observe_forward
        worker.model_runner.capture_one_sleep_graph = observe_capture
        original = manager.run_fullgraph

        def replay(desc):
            worker._test_graph_replays += 1
            return original(desc)

        manager.run_fullgraph = replay
        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
        custom_ar = getattr(communicator, "ca_comm", None)
        worker._test_custom_ar_calls = 0
        if custom_ar is not None and not custom_ar.disabled:
            original_ar = custom_ar.custom_all_reduce

            def all_reduce(input_):
                result = original_ar(input_)
                if result is not None:
                    worker._test_custom_ar_calls += 1
                return result

            custom_ar.custom_all_reduce = all_reduce
        # Observe graph release before the backend changes weight/KV mappings
        # or creates CPU backups. Those operations can change driver residency.
        backend = worker.sleep_mode_backend
        suspend = backend.suspend
        worker._test_before_suspend_free = None

        def observe_suspend(level=1):
            worker._test_before_suspend_free = torch.accelerator.get_memory_info()[0]
            return suspend(level)

        backend.suspend = observe_suspend
    # Clearing graph handles alone does not prove that the Python/C++ graph
    # objects, including their generator registrations, have been destroyed.
    worker._test_graph_objects.update(manager.graphs.values())
    return {
        "pid": os.getpid(),
        "graphs": len(manager.graphs),
        "graph_objects": len(worker._test_graph_objects),
        "uses_cumem_graph_pool": worker.vllm_config.use_cumem_cudagraph_pool,
        "replays": worker._test_graph_replays,
        "custom_ar_calls": worker._test_custom_ar_calls,
        "recapture_streams": sorted(worker._test_recapture_streams),
        "host_cache_flushes": worker._test_host_cache_flushes,
        "injected_oom": getattr(worker, "_test_injected_oom", False),
        "free_bytes": torch.accelerator.get_memory_info()[0],
        "before_suspend_free_bytes": worker._test_before_suspend_free,
    }


def _run_discard_recapture_rounds(
    monkeypatch,
    tensor_parallel_size: int,
    distributed_executor_backend: str,
    model: str | None = None,
    require_custom_allreduce: bool = False,
    workload_batch_size: int | None = None,
    graph_mode: str = "FULL",
    recoverable_oom: bool = False,
    shutdown_state: str | None = None,
    terminal_capture_failure: bool = False,
    **llm_kwargs,
):
    """Exercise real graphs, partial wake and outputs through the public API."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # The local test probe is a callable RPC, never exposed by the server API.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_SLEEP_DISCARD_GRAPHS", "1")
    if model is None:
        model = os.environ.get("SLEEP_GRAPH_TEST_MODEL", "Qwen/Qwen3-0.6B")
    llm = LLM(
        model=model,
        dtype=os.environ.get("SLEEP_GRAPH_TEST_DTYPE", "bfloat16"),
        max_model_len=512 if workload_batch_size is not None else 256,
        max_num_seqs=8,
        max_num_batched_tokens=512 if workload_batch_size is not None else 256,
        gpu_memory_utilization=0.8,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        **llm_kwargs,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": graph_mode,
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    )
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "The opposite of hot is",
        "The first three positive integers are",
    ]
    params = SamplingParams(temperature=0, max_tokens=16, seed=42)
    if workload_batch_size is not None:
        prompts = [
            "Read the following notes. "
            + "A city has parks, schools, and libraries. " * (i * 3)
            + "Describe a useful improvement to the city in detail."
            for i in range(workload_batch_size)
        ]
        params = SamplingParams(temperature=0, max_tokens=96, ignore_eos=True, seed=42)

    def generate():
        tokens = [o.outputs[0].token_ids for o in llm.generate(prompts, params)]
        if workload_batch_size is not None:
            assert len(tokens) == workload_batch_size
            assert all(len(output) == 96 for output in tokens)
        return tokens

    def snapshot():
        # One result per rank; TP ranks must never diverge on graph state.
        return llm.collective_rpc(_graph_snapshot)

    try:
        reference = generate()
        initial = snapshot()
        assert all(rank["graphs"] > 0 for rank in initial) and all(reference)
        assert all(not rank["uses_cumem_graph_pool"] for rank in initial)
        rounds = int(os.environ.get("SLEEP_GRAPH_TEST_ROUNDS", "2"))
        for iteration in range(rounds):
            before = snapshot()
            started = time.perf_counter()
            llm.sleep(level=1)
            sleep_s = time.perf_counter() - started
            asleep = snapshot()
            assert all(rank["graphs"] == 0 for rank in asleep)
            assert all(rank["graph_objects"] == 0 for rank in asleep)
            assert all(
                rank["free_bytes"] > prev["free_bytes"]
                for rank, prev in zip(asleep, before, strict=True)
            )
            if shutdown_state is not None:
                if shutdown_state != "asleep":
                    llm.wake_up(tags=[shutdown_state])
                return [rank["pid"] for rank in asleep]
            llm.sleep(level=1)
            tags = ["weights", "kv_cache"]
            if iteration % 2:
                tags.reverse()
            llm.wake_up(tags=tags[:1])
            assert all(rank["graphs"] == 0 for rank in snapshot())
            started = time.perf_counter()
            llm.wake_up(tags=tags[1:])
            final_tag_wake_s = time.perf_counter() - started
            awake = snapshot()
            assert all(rank["graphs"] == 0 for rank in awake)
            if recoverable_oom and iteration == 0:
                llm.collective_rpc(_arm_recoverable_capture_oom)
            if terminal_capture_failure:
                llm.collective_rpc(_arm_terminal_capture_failure)
            started = time.perf_counter()
            assert generate() == reference
            eager_s = time.perf_counter() - started
            first = snapshot()
            assert all(
                rank["replays"] == prev["replays"]
                for rank, prev in zip(first, awake, strict=True)
            )
            deadline = time.monotonic() + 120
            if recoverable_oom and iteration == 0:
                while not all(rank["injected_oom"] for rank in first):
                    assert time.monotonic() < deadline, "OOM was not injected"
                    time.sleep(0.05)
                    first = snapshot()
                assert all(rank["graphs"] == 0 for rank in first)
                assert generate() == reference
                recovered = snapshot()
                assert all(rank["graphs"] == 0 for rank in recovered)
                assert all(
                    rank["replays"] == prev["replays"]
                    for rank, prev in zip(recovered, first, strict=True)
                )
                continue
            while any(
                rank["graphs"] < prev["graphs"]
                for rank, prev in zip(first, initial, strict=True)
            ):
                assert time.monotonic() < deadline, "Idle recapture did not finish"
                time.sleep(0.05)
                first = snapshot()
            # Ranks recapture the same descriptors in lockstep.
            assert len({rank["graphs"] for rank in first}) == 1
            started = time.perf_counter()
            assert generate() == reference
            graph_s = time.perf_counter() - started
            final = snapshot()
            assert all(len(rank["recapture_streams"]) == 1 for rank in final)
            assert all(rank["host_cache_flushes"] == 0 for rank in final)
            if require_custom_allreduce:
                assert all(rank["custom_ar_calls"] > 0 for rank in final)
            assert all(
                rank["replays"] > prev["replays"]
                for rank, prev in zip(final, first, strict=True)
            )
            assert all(
                rank["pid"] == prev["pid"]
                for rank, prev in zip(final, initial, strict=True)
            )
            print(
                "SLEEP_GRAPH_RESULT "
                + json.dumps(
                    {
                        "model": model,
                        "tensor_parallel_size": tensor_parallel_size,
                        "backend": distributed_executor_backend,
                        "iteration": iteration,
                        "timestamp": time.time(),
                        "sleep_s": sleep_s,
                        "final_tag_wake_s": final_tag_wake_s,
                        "eager_s": eager_s,
                        "graph_s": graph_s,
                        "before": before,
                        "asleep": asleep,
                        "awake": awake,
                        "recaptured": final,
                    }
                ),
                flush=True,
            )
    finally:
        llm.llm_engine.engine_core.shutdown()


@multi_gpu_test(num_gpus=1)
@pytest.mark.parametrize("graph_mode", ["FULL", "FULL_DECODE_ONLY"])
def test_graph_discard_sleep_recaptures_after_first_request(monkeypatch, graph_mode):
    _run_discard_recapture_rounds(monkeypatch, 1, "uni", graph_mode=graph_mode)


def _arm_recoverable_capture_oom(worker):
    import torch

    model = worker.model_runner.model
    forward = model.forward

    def fail_once(*args, **kwargs):
        if (
            torch.cuda.is_current_stream_capturing()
            and worker.model_runner.cudagraph_manager.graphs
        ):
            model.forward = forward
            worker._test_injected_oom = True
            raise torch.OutOfMemoryError("injected recoverable capture failure")
        return forward(*args, **kwargs)

    model.forward = fail_once


@multi_gpu_test(num_gpus=1)
def test_graph_discard_sleep_recovers_after_single_worker_oom(monkeypatch):
    monkeypatch.setenv("SLEEP_GRAPH_TEST_ROUNDS", "2")
    _run_discard_recapture_rounds(monkeypatch, 1, "uni", recoverable_oom=True)


def _arm_terminal_capture_failure(worker):
    def fail_capture():
        # Keep capture in progress while the client sends its next graph-state
        # RPC. That RPC must fail when the engine dies, not wait indefinitely.
        time.sleep(0.5)
        raise RuntimeError("injected terminal capture failure")

    worker.model_runner.capture_one_sleep_graph = fail_capture


@multi_gpu_test(num_gpus=1)
@pytest.mark.timeout(180)
def test_graph_capture_failure_releases_pending_sync_rpc(monkeypatch):
    from vllm.v1.engine.exceptions import EngineDeadError

    with pytest.raises(EngineDeadError):
        _run_discard_recapture_rounds(
            monkeypatch, 1, "uni", terminal_capture_failure=True
        )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("custom_allreduce", [False, True])
def test_graph_discard_sleep_tensor_parallel(monkeypatch, custom_allreduce):
    """TP2: graphs embed cross-rank collectives; discard/recapture must hold."""
    if custom_allreduce:
        monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0")
        monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0")
        monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
    _run_discard_recapture_rounds(
        monkeypatch, 2, "mp", require_custom_allreduce=custom_allreduce
    )


@multi_gpu_test(num_gpus=2)
def test_graph_discard_sleep_tensor_parallel_ray(monkeypatch):
    """TP2 on the ray executor: same lockstep discard/recapture contract."""
    _run_discard_recapture_rounds(monkeypatch, 2, "ray")


@multi_gpu_test(num_gpus=2)
def test_graph_discard_sleep_data_parallel(monkeypatch):
    """DP2: each engine core discards and recaptures its own rank's graphs."""
    asyncio.run(_run_dp_discard_recapture_rounds(monkeypatch))


@multi_gpu_test(num_gpus=2)
def test_graph_discard_sleep_pipeline_parallel(monkeypatch):
    """PP2: per-stage graphs embed no cross-stage collectives; lockstep holds."""
    _run_discard_recapture_rounds(monkeypatch, 1, "mp", pipeline_parallel_size=2)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("tp,pp", [(4, 1), (2, 2)])
def test_graph_discard_sleep_four_gpus(monkeypatch, tp, pp):
    _run_discard_recapture_rounds(monkeypatch, tp, "mp", pipeline_parallel_size=pp)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_graph_discard_sleep_long_mixed_inputs(monkeypatch, batch_size):
    _run_discard_recapture_rounds(monkeypatch, 2, "mp", workload_batch_size=batch_size)


@multi_gpu_test(num_gpus=1)
def test_graph_discard_sleep_with_cumem_graph_pool(monkeypatch):
    _run_discard_recapture_rounds(
        monkeypatch, 1, "uni", sleep_mode_offload_cudagraph=True
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("tp,state", [(1, "asleep"), (2, "asleep"), (2, "weights")])
def test_graph_discard_shutdown_while_asleep(monkeypatch, capfd, tp, state):
    pids = _run_discard_recapture_rounds(
        monkeypatch, tp, "uni" if tp == 1 else "mp", shutdown_state=state
    )
    _assert_sleep_graph_workers_stopped(pids)
    captured = capfd.readouterr()
    assert "CUDA Error:" not in captured.err
    assert "Fatal Python error:" not in captured.err


@multi_gpu_test(num_gpus=2)
def test_graph_discard_sleep_qwen3_5_hybrid(monkeypatch):
    """Qwen3.5 hybrid linear-attention arch, allowed via the env extension."""
    model = os.environ.get("SLEEP_GRAPH_TEST_QWEN35_MODEL")
    if model is None:
        pytest.skip("SLEEP_GRAPH_TEST_QWEN35_MODEL not set")
    monkeypatch.setenv(
        "VLLM_SLEEP_DISCARD_GRAPHS_EXTRA_MODELS", "Qwen3_5ForConditionalGeneration"
    )
    _run_discard_recapture_rounds(monkeypatch, 2, "mp", model=model)


@multi_gpu_test(num_gpus=2)
def test_graph_discard_sleep_expert_parallel(monkeypatch):
    """MoE + EP: expert all2all collectives live inside the graphs."""
    model = os.environ.get("SLEEP_GRAPH_TEST_MOE_MODEL")
    if model is None:
        pytest.skip("SLEEP_GRAPH_TEST_MOE_MODEL not set")
    _run_discard_recapture_rounds(
        monkeypatch, 2, "mp", model=model, enable_expert_parallel=True
    )


async def _run_dp_discard_recapture_rounds(monkeypatch) -> None:
    """DP2 variant: the sync LLM entry point rejects data_parallel_size > 1,
    so drive two engine cores through AsyncLLM with the mp DP backend."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_SLEEP_DISCARD_GRAPHS", "1")
    model = os.environ.get("SLEEP_GRAPH_TEST_MODEL", "Qwen/Qwen3-0.6B")
    engine_args = AsyncEngineArgs(
        model=model,
        dtype=os.environ.get("SLEEP_GRAPH_TEST_DTYPE", "bfloat16"),
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.8,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        data_parallel_size=2,
        data_parallel_backend="mp",
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "The opposite of hot is",
        "The first three positive integers are",
    ]
    params = SamplingParams(temperature=0, max_tokens=16, seed=42)

    async def generate() -> list:
        async def one(request_id: str, prompt: str):
            final = None
            async for out in engine.generate(prompt, params, request_id):
                final = out
            return final.outputs[0].token_ids

        return await asyncio.gather(
            *(one(f"req-{time.monotonic()}-{i}", p) for i, p in enumerate(prompts))
        )

    async def snapshot() -> list:
        # One result per worker; engine cores must never diverge on graphs.
        # collective_rpc targets only the first engine, so fan out explicitly.
        per_engine = await engine.engine_core.call_utility_all_async(
            "collective_rpc", _graph_snapshot
        )
        return [result for results in per_engine for result in results]

    try:
        reference = await generate()
        initial = await snapshot()
        assert all(rank["graphs"] > 0 for rank in initial) and all(reference)
        rounds = int(os.environ.get("SLEEP_GRAPH_TEST_ROUNDS", "2"))
        for iteration in range(rounds):
            started = time.perf_counter()
            await engine.sleep(level=1)
            sleep_s = time.perf_counter() - started
            asleep = await snapshot()
            assert all(rank["graphs"] == 0 for rank in asleep)
            await engine.wake_up(tags=["weights"])
            assert all(rank["graphs"] == 0 for rank in await snapshot())
            started = time.perf_counter()
            await engine.wake_up(tags=["kv_cache"])
            wake_s = time.perf_counter() - started
            assert all(rank["graphs"] == 0 for rank in await snapshot())
            started = time.perf_counter()
            assert await generate() == reference
            eager_s = time.perf_counter() - started
            deadline = time.monotonic() + 120
            current = await snapshot()
            while any(
                rank["graphs"] < prev["graphs"]
                for rank, prev in zip(current, initial, strict=True)
            ):
                assert time.monotonic() < deadline, "Idle recapture did not finish"
                await asyncio.sleep(0.05)
                current = await snapshot()
            assert len({rank["graphs"] for rank in current}) == 1
            started = time.perf_counter()
            assert await generate() == reference
            graph_s = time.perf_counter() - started
            final = await snapshot()
            assert all(
                rank["replays"] > prev["replays"]
                for rank, prev in zip(final, current, strict=True)
            )
            print(
                "SLEEP_GRAPH_RESULT "
                + json.dumps(
                    {
                        "model": model,
                        "data_parallel_size": 2,
                        "backend": "mp",
                        "iteration": iteration,
                        "timestamp": time.time(),
                        "sleep_s": sleep_s,
                        "wake_s": wake_s,
                        "eager_s": eager_s,
                        "graph_s": graph_s,
                        "initial": initial,
                        "asleep": asleep,
                        "final": final,
                    }
                )
            )
    finally:
        engine.shutdown()


def _arm_recapture_failure(worker, phase):
    import torch

    from vllm.distributed.parallel_state import get_tp_group

    if get_tp_group().rank_in_group != 1:
        return
    if phase in ("sleep", "wake"):

        def fail_transition(*args, **kwargs):
            raise RuntimeError("injected sleep graph transition failure")

        if phase == "sleep":
            worker.model_runner.cudagraph_manager.discard_full_graphs = fail_transition
        else:
            worker.sleep_mode_backend.resume = fail_transition
        return
    runner = worker.model_runner
    capture = runner.capture_one_sleep_graph

    def recapture():
        if phase == "prepare":
            raise torch.OutOfMemoryError("injected recapture preparation failure")
        forward = runner.model.forward

        def fail_forward(*args, **kwargs):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing == (phase == "capture"):
                raise torch.OutOfMemoryError("injected recapture forward failure")
            return forward(*args, **kwargs)

        runner.model.forward = fail_forward
        return capture()

    runner.capture_one_sleep_graph = recapture


def _worker_pid(worker):
    return os.getpid()


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("phase", ["prepare", "warmup", "capture", "sleep", "wake"])
@pytest.mark.timeout(180)
def test_graph_recapture_failure_terminates_all_workers(monkeypatch, phase):
    """A rank-local OOM must reach the client and release blocked peers."""
    from vllm.v1.engine.exceptions import EngineDeadError

    monkeypatch.setenv("VLLM_SLEEP_DISCARD_GRAPHS", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    rpc_timeout = int(os.environ.get("SLEEP_GRAPH_TEST_RPC_TIMEOUT_SECONDS", "5"))
    failure_budget = rpc_timeout + 30
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", str(rpc_timeout))
    llm = LLM(
        model=os.environ.get("SLEEP_GRAPH_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        # NCCL exposes a peer blocked inside a real collective after rank 1 fails.
        disable_custom_all_reduce=True,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        gpu_memory_utilization=float(
            os.environ.get("SLEEP_GRAPH_TEST_GPU_MEMORY_UTILIZATION", "0.25")
        ),
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    )
    pids = []
    try:
        params = SamplingParams(temperature=0, max_tokens=8)
        llm.generate("The capital of France is", params)
        pids = llm.collective_rpc(_worker_pid)
        if phase == "wake":
            llm.sleep()
        if phase in ("sleep", "wake"):
            llm.collective_rpc(_arm_recapture_failure, args=(phase,))
            with pytest.raises(Exception) as failure:
                (llm.sleep if phase == "sleep" else llm.wake_up)()
            assert (
                isinstance(failure.value, EngineDeadError)
                or ("injected sleep graph transition failure" in str(failure.value))
                or (
                    # A peer may be blocked before the failing rank's reply is read.
                    "RPC call to " in str(failure.value)
                    and "timed out" in str(failure.value)
                )
            )
        else:
            llm.sleep()
            llm.wake_up()
            llm.collective_rpc(_arm_recapture_failure, args=(phase,))
        started = time.monotonic()
        with pytest.raises(EngineDeadError):
            # Idle capture follows the first eager request. Repeated requests
            # allow either ordering between completion delivery and capture.
            while time.monotonic() - started < failure_budget:
                llm.generate("The opposite of hot is", params)
                time.sleep(0.1)
        assert time.monotonic() - started < failure_budget
        # Check before explicit client cleanup, which otherwise hides workers
        # left behind by the engine's failure path.
        _assert_sleep_graph_workers_stopped(pids)
    finally:
        llm.llm_engine.engine_core.shutdown()


def _assert_sleep_graph_workers_stopped(pids):
    from pathlib import Path

    deadline = time.monotonic() + 5
    while True:
        alive = []
        for pid in pids:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0]
            except FileNotFoundError:
                continue
            if state != "Z":
                alive.append(pid)
        if not alive or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    assert not alive, f"Workers survived engine failure: {alive}"
