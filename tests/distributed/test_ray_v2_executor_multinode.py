# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Multi-node integration tests for RayExecutorV2.
Validates executor initialization, worker placement, and end-to-end
generation across multiple nodes with various TP/PP configurations.

Requires VLLM_MULTI_NODE=1 env var and a multi-node Ray cluster.

Run:
```sh
VLLM_MULTI_NODE=1 VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
    pytest -v -s distributed/test_ray_v2_executor_multinode.py
```
"""

import gc
import os
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

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"

pytestmark = pytest.mark.skipif(
    not VLLM_MULTI_NODE, reason="Need VLLM_MULTI_NODE=1 and a multi-node cluster."
)


@pytest.fixture(autouse=True)
def enable_ray_v2_backend():
    """Enable the RayExecutorV2 backend via feature flag for all tests."""
    saved = {
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": os.environ.get(
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND"
        ),
        "RAY_RUNTIME_ENV_HOOK": os.environ.get("RAY_RUNTIME_ENV_HOOK"),
        "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        ),
    }
    os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "1"
    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    # Disable multiprocessing to avoid fork-after-Ray-init issues.
    # The RayExecutorV2 already handles distribution via Ray actors,
    # so the EngineCore can safely run in-process.
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

    # Wait briefly for async cleanup (del llm triggers GC-based shutdown)
    for _ in range(10):
        dangling_actors = [
            actor
            for actor in list_actors(filters=[("state", "=", "ALIVE")])
            if actor.class_name == "RayWorkerProc"
        ]
        if not dangling_actors:
            break
        time.sleep(1)

    for pg_id, pg_info in ray.util.placement_group_table().items():
        if pg_info["state"] == "CREATED":
            pg = PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id)))
            ray.util.remove_placement_group(pg)

    # Disconnect from Ray so forked subprocesses (EngineCore) don't inherit
    # a stale driver connection that can't create placement groups.
    ray.shutdown()


def create_vllm_config(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: int = 256,
    gpu_memory_utilization: float = 0.3,
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
    return engine_args.create_engine_config()


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


def test_ray_v2_multinode_executor_init():
    """Validate RayExecutorV2 initializes correctly across multiple nodes
    with TP=4, PP=2 (8 GPUs)."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
    )
    executor = RayExecutorV2(vllm_config=vllm_config)
    try:
        assert_executor(executor, tp_size=4, pp_size=2)

        # Verify workers span multiple nodes
        node_ids = {h.node_id for h in executor.ray_worker_handles}
        assert len(node_ids) > 1

        # Verify rank 0 exists and has a valid node_id.
        # On clusters where the driver node has GPUs, rank 0 will be there.
        # On GPU-less head nodes, rank 0 is on the first GPU node instead.
        rank0_handle = next(h for h in executor.ray_worker_handles if h.rank == 0)
        assert rank0_handle.node_id is not None
    finally:
        executor.shutdown()


def test_ray_v2_multinode_worker_placement():
    """Verify TP locality: workers in the same TP group share a node."""
    tp_size = 4
    pp_size = 2

    vllm_config = create_vllm_config(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    executor = RayExecutorV2(vllm_config=vllm_config)
    try:
        # Workers are sorted by rank; consecutive tp_size ranks form a TP group
        for pp_rank in range(pp_size):
            start_rank = pp_rank * tp_size
            tp_group_handles = [
                h
                for h in executor.ray_worker_handles
                if start_rank <= h.rank < start_rank + tp_size
            ]
            tp_group_nodes = {h.node_id for h in tp_group_handles}
            assert len(tp_group_nodes) == 1

        # Workers should be distributed across > 1 node
        all_nodes = {h.node_id for h in executor.ray_worker_handles}
        assert len(all_nodes) > 1
    finally:
        executor.shutdown()


def test_ray_v2_multinode_generation():
    """End-to-end LLM generation with TP=4, PP=2 across multiple nodes."""
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
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


@pytest.mark.parametrize("tp_size, pp_size", [(4, 2), (2, 4)])
def test_ray_v2_multinode_generation_with_pg(tp_size, pp_size):
    """E2E LLM generation with a user-provided placement group across nodes."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

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
