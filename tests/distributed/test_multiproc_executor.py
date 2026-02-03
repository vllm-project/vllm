# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for MultiprocExecutor at the executor level.
This test directly tests the executor without going through the LLM interface,
focusing on executor initialization, RPC calls, and distributed execution.
"""

import multiprocessing
import os

from tests.utils import multi_gpu_test
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import get_open_port
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

MODEL = "facebook/opt-125m"


def create_vllm_config(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: int = 256,
    gpu_memory_utilization: float = 0.3,
    distributed_executor_backend: str = "mp",
    nnodes: int = 1,
    node_rank: int = 0,
    master_port: int = 0,
) -> VllmConfig:
    """Create a VllmConfig for testing using EngineArgs."""
    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()

    # Override distributed node settings if needed
    if nnodes > 1 or node_rank > 0:
        vllm_config.parallel_config.nnodes = nnodes
        vllm_config.parallel_config.node_rank = node_rank
        vllm_config.parallel_config.master_port = master_port
    if nnodes > 1:
        vllm_config.parallel_config.disable_custom_all_reduce = True

    return vllm_config


def create_test_scheduler_output(num_requests: int = 1) -> SchedulerOutput:
    """Create a minimal SchedulerOutput for testing."""
    # This is a simplified version - in practice you'd need proper
    # SchedulerOutput construction based on the actual vLLM v1 API
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        scheduled_running_reqs=[],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
    )


def test_multiproc_executor_initialization():
    """Test that MultiprocExecutor can be initialized with proper config."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    # Create executor - this should initialize workers
    executor = MultiprocExecutor(vllm_config=vllm_config)

    # Verify executor properties
    assert executor.world_size == 1, "World size should be 1 for single GPU"
    assert executor.local_world_size == 1, "Local world size should be 1"
    assert hasattr(executor, "workers"), "Executor should have workers"
    assert len(executor.workers) == 1, "Should have 1 worker for single GPU"

    # Clean up
    executor.shutdown()


@multi_gpu_test(num_gpus=2)
def test_multiproc_executor_initialization_tensor_parallel():
    """Test MultiprocExecutor initialization with tensor parallelism."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    # Create executor
    executor = MultiprocExecutor(vllm_config=vllm_config)

    # Verify executor properties
    assert executor.world_size == 2, "World size should be 2 for TP=2"
    assert executor.local_world_size == 2, "Local world size should be 2"
    assert len(executor.workers) == 2, "Should have 2 workers for TP=2"

    # Verify output rank calculation
    output_rank = executor._get_output_rank()
    assert output_rank == 0, "Output rank should be 0 for TP=2, PP=1"

    # Clean up
    executor.shutdown()


@multi_gpu_test(num_gpus=2)
def test_multiproc_executor_collective_rpc():
    """Test collective RPC calls to all workers."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    # Create executor
    executor = MultiprocExecutor(vllm_config=vllm_config)

    try:
        # Test check_health RPC - should work without errors
        executor.check_health()

        # Test that RPC works correctly
        # Note: We're just testing that the RPC mechanism works,
        # not testing actual model execution here
        assert not executor.is_failed, "Executor should not be in failed state"

    finally:
        # Clean up
        executor.shutdown()


def test_multiproc_executor_failure_callback():
    """Test failure callback registration and invocation."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    executor = MultiprocExecutor(vllm_config=vllm_config)

    try:
        # Test callback registration
        callback_invoked = []

        def test_callback():
            callback_invoked.append(True)

        # Register callback
        executor.register_failure_callback(test_callback)

        # Callback should not be invoked yet
        assert len(callback_invoked) == 0, "Callback should not be invoked immediately"

        # Simulate failure
        executor.is_failed = True

        # Register another callback - should be invoked immediately
        executor.register_failure_callback(test_callback)
        assert len(callback_invoked) == 1, (
            "Callback should be invoked when executor is failed"
        )

    finally:
        # Clean up
        executor.shutdown()


@multi_gpu_test(num_gpus=2)
def test_multiproc_executor_worker_monitor():
    """Test that worker monitor is set up correctly."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    executor = MultiprocExecutor(vllm_config=vllm_config)

    try:
        # Verify all worker processes are alive
        for worker in executor.workers:
            assert worker.proc.is_alive(), f"Worker rank {worker.rank} should be alive"

        # Verify executor is not in failed state
        assert not executor.is_failed, "Executor should not be in failed state"

    finally:
        # Clean up
        executor.shutdown()

        # After shutdown, workers should be terminated
        import time

        time.sleep(0.5)  # Give processes time to terminate
        for worker in executor.workers:
            assert not worker.proc.is_alive(), (
                f"Worker rank {worker.rank} should terminate after shutdown"
            )


@multi_gpu_test(num_gpus=2)
def test_multiproc_executor_get_response_message_queues():
    """Test message queue retrieval for different ranks."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    executor = MultiprocExecutor(vllm_config=vllm_config)

    try:
        # Get all message queues
        all_queues = executor.get_response_mqs()
        assert len(all_queues) == 2, "Should have 2 message queues for 2 workers"

        # Get message queue for specific rank
        rank0_queue = executor.get_response_mqs(unique_reply_rank=0)
        assert len(rank0_queue) == 1, "Should have 1 message queue for rank 0"

        rank1_queue = executor.get_response_mqs(unique_reply_rank=1)
        assert len(rank1_queue) == 1, "Should have 1 message queue for rank 1"

    finally:
        # Clean up
        executor.shutdown()


def test_multiproc_executor_shutdown_cleanup():
    """Test that shutdown properly cleans up resources."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    executor = MultiprocExecutor(vllm_config=vllm_config)

    # Verify executor is set up
    assert hasattr(executor, "workers"), "Executor should have workers"
    assert len(executor.workers) > 0, "Should have at least one worker"

    # Shutdown
    executor.shutdown()

    # Verify cleanup
    import time

    time.sleep(0.5)  # Give processes time to terminate

    for worker in executor.workers:
        assert not worker.proc.is_alive(), "Worker processes should be terminated"

    # Verify shutdown event is set
    assert executor.shutdown_event.is_set(), "Shutdown event should be set"

    # Multiple shutdowns should be safe (idempotent)
    executor.shutdown()
    executor.shutdown()


@multi_gpu_test(num_gpus=4)
def test_multiproc_executor_pipeline_parallel():
    """Test MultiprocExecutor with pipeline parallelism."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )

    executor = MultiprocExecutor(vllm_config=vllm_config)

    try:
        # Verify executor properties
        assert executor.world_size == 4, "World size should be 4 for TP=2, PP=2"
        assert len(executor.workers) == 4, "Should have 4 workers"

        # Verify output rank calculation
        # For TP=2, PP=2: output should be from the last PP stage (ranks 2-3)
        # Specifically rank 2 (first rank of last PP stage)
        output_rank = executor._get_output_rank()
        assert output_rank == 2, "Output rank should be 2 (first rank of last PP stage)"

        # Verify max_concurrent_batches for pipeline parallel
        assert executor.max_concurrent_batches == 2, (
            "Max concurrent batches should equal PP size"
        )

    finally:
        # Clean up
        executor.shutdown()


def test_multiproc_executor_properties():
    """Test various executor properties and configurations."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    executor = MultiprocExecutor(vllm_config=vllm_config)

    try:
        # Test supports_pp property
        assert MultiprocExecutor.supports_pp is True, (
            "MultiprocExecutor should support pipeline parallelism"
        )

        # Test world_size calculation
        assert executor.world_size == (
            executor.parallel_config.tensor_parallel_size
            * executor.parallel_config.pipeline_parallel_size
        ), "World size should equal TP * PP"

        # Test local_world_size calculation
        assert executor.local_world_size == (
            executor.parallel_config.world_size // executor.parallel_config.nnodes
        ), "Local world size should be world_size / nnodes"

    finally:
        # Clean up
        executor.shutdown()


@multi_gpu_test(num_gpus=4)
def test_multiproc_executor_multi_node():
    """
    Test MultiprocExecutor with multi-node configuration.
    This simulates 2 nodes with TP=4:
    - Node 0 (rank 0): Uses GPUs 0,1 (CUDA_VISIBLE_DEVICES=0,1) with TP=2
    - Node 1 (rank 1): Uses GPUs 2,3 (CUDA_VISIBLE_DEVICES=2,3) with TP=2
    Total world_size = 4, nnodes = 2
    """
    port = get_open_port()
    # symm_mem does not work for simulating multi instance in single node
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"

    def run_node(node_rank: int, result_queue: multiprocessing.Queue, port: int):
        """Run a single node's executor."""
        executor = None
        try:
            # Set CUDA_VISIBLE_DEVICES for this node
            if node_rank == 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

            # Create config for this node
            vllm_config = create_vllm_config(
                tensor_parallel_size=4,  # Total TP across all nodes
                pipeline_parallel_size=1,
                nnodes=2,  # 2 nodes
                node_rank=node_rank,
                master_port=port,  # same port
            )

            # Create executor for this node
            executor = MultiprocExecutor(vllm_config=vllm_config)

            # Verify node-specific properties
            assert executor.world_size == 4, (
                f"World size should be 4 on node {node_rank}"
            )
            assert executor.local_world_size == 2, (
                f"Local world size should be 2 on node {node_rank}"
            )
            assert len(executor.workers) == 2, (
                f"Should have 2 local workers on node {node_rank}"
            )

            # Verify worker ranks are correct for this node
            expected_ranks = [node_rank * 2, node_rank * 2 + 1]
            actual_ranks = sorted([w.rank for w in executor.workers])
            assert actual_ranks == expected_ranks, (
                f"Node {node_rank} should have workers "
                f"with ranks {expected_ranks}, got {actual_ranks}"
            )
            # Verify all workers are alive
            for worker in executor.workers:
                assert worker.proc.is_alive(), (
                    f"Worker rank {worker.rank} should be alive on node {node_rank}"
                )
            # executor.gen
            # Put success result in queue BEFORE shutdown to avoid hanging
            result_queue.put({"node": node_rank, "success": True})
            import time

            time.sleep(2)
            executor.shutdown()
        except Exception as e:
            # Put failure result in queue
            result_queue.put({"node": node_rank, "success": False, "error": str(e)})
            raise e
        finally:
            if executor is not None:
                executor.shutdown()

    # Create a queue to collect results from both processes
    result_queue: multiprocessing.Queue[dict[str, int | bool]] = multiprocessing.Queue()

    # Start both node processes
    processes = []
    for node_rank in range(2):
        p = multiprocessing.Process(
            target=run_node,
            args=(node_rank, result_queue, port),
            name=f"Node{node_rank}",
        )
        p.start()
        processes.append(p)

    # Wait for both processes to complete
    all_completed = True
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=20)
            if p.is_alive():
                p.kill()
                p.join()
            all_completed = False

    # Check results from both nodes
    results: list[dict[str, int | bool]] = []
    while len(results) < 2:
        try:
            result = result_queue.get(timeout=1)
            results.append(result)
        except Exception:
            pass
    assert all_completed, "Not all processes completed successfully"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0]["success"], f"Node 0 failed: {results[0]}"
    assert results[1]["success"], f"Node 1 failed: {results[1]}"
