# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for RayExecutorV2 at the executor level.
Validates executor initialization, placement group support, RPC calls,
and distributed execution with various TP/PP configurations.
"""

import gc
import os
import threading
import time
from unittest.mock import patch

import pytest
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.state import list_actors

from vllm import LLM
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

MODEL = "facebook/opt-125m"


@pytest.fixture(autouse=True)
def enable_ray_v2_backend():
    """Enable the RayExecutorV2 backend via feature flag for all tests."""
    saved = {
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": os.environ.get(
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND"
        ),
        "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        ),
    }
    os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "1"
    # The multiprocess engine forks a subprocess that inherits the Ray
    # driver connection, causing hangs. RayExecutorV2 already distributes
    # work via Ray actors, so the EngineCore can run safely in-process.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    try:
        yield
    finally:
        _cleanup_ray_resources()
        os.environ.update({k: v for k, v in saved.items() if v is not None})
        for key in (k for k, v in saved.items() if v is None):
            os.environ.pop(key, None)


def _cleanup_ray_resources():
    if not ray.is_initialized():
        return

    # Ray actor shutdown is async -- wait until all actors are dead.
    dangling_actors = []
    try:
        for _ in range(10):
            dangling_actors = [
                actor
                for actor in list_actors(filters=[("state", "=", "ALIVE")])
                if actor.class_name == "RayWorkerProc"
            ]
            if not dangling_actors:
                break
            time.sleep(1)
    except Exception:
        # Tolerate connection errors to the Ray dashboard
        pass

    assert not dangling_actors
    ray.shutdown()


def create_vllm_config(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: int = 256,
    gpu_memory_utilization: float = 0.3,
    placement_group=None,
) -> VllmConfig:
    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        distributed_executor_backend="ray",
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()

    if placement_group is not None:
        vllm_config.parallel_config.placement_group = placement_group

    return vllm_config


def ensure_ray_initialized():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


@pytest.fixture
def create_placement_group(request):
    ensure_ray_initialized()
    num_gpus = request.param
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = ray.util.placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    yield pg
    ray.util.remove_placement_group(pg)


@pytest.fixture
def executor(request):
    """Create a RayExecutorV2 and shut it down after the test."""
    executor = RayExecutorV2(vllm_config=request.param)
    yield executor
    executor.shutdown()


def assert_executor(executor, tp_size, pp_size):
    """Common assertions for executor initialization tests."""
    world_size = tp_size * pp_size
    expected_output_rank = (pp_size - 1) * tp_size

    assert executor.world_size == world_size
    assert len(executor.ray_worker_handles) == world_size
    assert len(executor.response_mqs) == world_size
    assert executor._get_output_rank() == expected_output_rank

    if pp_size > 1:
        assert executor.max_concurrent_batches == pp_size

    executor.check_health()
    assert not executor.is_failed

    ranks = sorted(h.rank for h in executor.ray_worker_handles)
    assert ranks == list(range(world_size))

    for handle in executor.ray_worker_handles:
        assert handle.node_id is not None


@pytest.mark.parametrize("tp_size, pp_size", [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_ray_v2_executor(tp_size, pp_size):
    """Validate RayExecutorV2 with various TP/PP configs."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    executor = RayExecutorV2(vllm_config=vllm_config)
    try:
        assert_executor(executor, tp_size, pp_size)
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "tp_size, pp_size, create_placement_group",
    [(2, 1, 2), (4, 1, 4), (2, 2, 4)],
    indirect=["create_placement_group"],
)
def test_ray_v2_executor_pg(tp_size, pp_size, create_placement_group):
    """Validate RayExecutorV2 with various TP/PP configs using external PG."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        placement_group=create_placement_group,
    )
    executor = RayExecutorV2(vllm_config=vllm_config)
    try:
        assert_executor(executor, tp_size, pp_size)
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "executor",
    [create_vllm_config(tensor_parallel_size=2)],
    indirect=True,
)
def test_ray_v2_executor_failure_callback(executor):
    """Validate failure callback registration."""
    callback_invoked = False

    def test_callback():
        nonlocal callback_invoked
        callback_invoked = True

    executor.register_failure_callback(test_callback)
    assert not callback_invoked

    executor.is_failed = True
    executor.register_failure_callback(test_callback)
    assert callback_invoked


@pytest.mark.parametrize(
    "executor",
    [create_vllm_config(tensor_parallel_size=2)],
    indirect=True,
)
def test_ray_v2_executor_collective_rpc(executor):
    """Validate collective RPC calls through MessageQueue."""
    executor.check_health()
    assert not executor.is_failed
    assert executor.rpc_broadcast_mq is not None


@pytest.mark.parametrize(
    "executor",
    [create_vllm_config(tensor_parallel_size=2)],
    indirect=True,
)
def test_ray_v2_executor_driver_node_rank_0(executor):
    """Validate that driver node workers get the lowest ranks."""
    driver_node = ray.get_runtime_context().get_node_id()

    for handle in executor.ray_worker_handles:
        assert handle.node_id == driver_node

    rank0_handle = next(h for h in executor.ray_worker_handles if h.rank == 0)
    assert rank0_handle.node_id == driver_node


@pytest.mark.parametrize(
    "executor",
    [create_vllm_config(tensor_parallel_size=2)],
    indirect=True,
)
def test_ray_v2_executor_worker_death(executor):
    """Validate executor detects worker death via ray.wait()."""
    callback_event = threading.Event()

    def on_failure():
        callback_event.set()

    executor.register_failure_callback(on_failure)
    assert not executor.is_failed

    # Kill one worker actor externally
    victim = executor.ray_worker_handles[1].actor
    ray.kill(victim, no_restart=True)

    # Monitor thread should detect the death and invoke callback
    assert callback_event.wait(timeout=30)
    assert executor.is_failed
    assert executor.shutting_down


def test_ray_v2_executor_shutdown():
    """Validate graceful shutdown: ray.kill() terminates all worker actors."""
    executor = RayExecutorV2(vllm_config=create_vllm_config(tensor_parallel_size=2))
    assert executor.rpc_broadcast_mq is not None
    assert len(executor.response_mqs) == executor.world_size

    actors = [h.actor for h in executor.ray_worker_handles]
    executor.shutdown()

    for actor in actors:
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.wait_for_init.remote(), timeout=5)

    assert executor.rpc_broadcast_mq is None
    assert len(executor.response_mqs) == 0


@pytest.mark.parametrize(
    "executor",
    [create_vllm_config(tensor_parallel_size=2)],
    indirect=True,
)
def test_ray_v2_run_refs_stored_for_monitoring(executor):
    """Validate worker handles store run_ref for monitoring."""
    for handle in executor.ray_worker_handles:
        assert handle.run_ref is not None
        ready, _ = ray.wait([handle.run_ref], timeout=0)
        assert len(ready) == 0, "run_ref should be pending"


@pytest.mark.parametrize("tp_size, pp_size", [(2, 1), (2, 2)])
def test_ray_v2_single_node_generation(tp_size, pp_size):
    """End-to-end LLM generation with RayExecutorV2."""

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        distributed_executor_backend="ray",
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=0.3,
    )
    try:
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
        ]
        outputs = llm.generate(prompts)

        assert len(outputs) == len(prompts)
        for output in outputs:
            assert len(output.outputs) > 0
            assert len(output.outputs[0].text) > 0
    finally:
        llm.llm_engine.model_executor.shutdown()
        del llm
        gc.collect()


@pytest.mark.parametrize("tp_size, pp_size", [(2, 1), (2, 2)])
def test_ray_v2_single_node_generation_with_pg(tp_size, pp_size):
    """E2E LLM generation with a user-provided placement group."""
    ensure_ray_initialized()
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(tp_size * pp_size)]
    pg = ray.util.placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    try:
        with patch.object(ray.util, "get_current_placement_group", return_value=pg):
            llm = LLM(
                model=MODEL,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                distributed_executor_backend="ray",
                enforce_eager=True,
                max_model_len=256,
                gpu_memory_utilization=0.3,
            )
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
        ]
        outputs = llm.generate(prompts)

        assert len(outputs) == len(prompts)
        for output in outputs:
            assert len(output.outputs) > 0
            assert len(output.outputs[0].text) > 0
    finally:
        llm.llm_engine.model_executor.shutdown()
        del llm
        gc.collect()
