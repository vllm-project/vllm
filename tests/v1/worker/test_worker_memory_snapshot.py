# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import os
import tempfile
from multiprocessing.queues import Queue
from unittest.mock import patch

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment

# Global queue to track operation order across processes
_QUEUE: Queue | None = None


def track_operation(operation: str, rank: int):
    """Track when an operation happens and its rank."""
    if _QUEUE is not None:
        _QUEUE.put((operation, rank))


def make_operation_tracker(operation_name: str, original_func):
    """Create a mock function that tracks when an operation is called.

    Args:
        operation_name: Name to use when tracking this operation
        original_func: The original function to wrap

    Returns:
        A wrapper function that tracks the operation and calls the original
    """

    def wrapper(*args, **kwargs):
        rank = int(os.environ.get("RANK", "-1"))
        track_operation(operation_name, rank)
        return original_func(*args, **kwargs)

    return wrapper


def worker_process(
    rank: int,
    world_size: int,
    distributed_init_method: str,
    queue: Queue,
    error_queue: Queue,
):
    """Worker process that initializes a GPU worker with proper tracking."""
    global _QUEUE
    _QUEUE = queue

    try:
        # Set environment variables
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Create vLLM config with small model
        vllm_config = EngineArgs(
            model="facebook/opt-125m", tensor_parallel_size=2, load_format="dummy"
        ).create_engine_config()

        # Create worker
        worker = Worker(
            vllm_config=vllm_config,
            local_rank=rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

        # Get original functions before patching
        original_init_worker = init_worker_distributed_environment
        original_memory_snapshot_init = MemorySnapshot.__init__
        original_all_reduce = torch.distributed.all_reduce

        # Apply minimal patches to track operation order
        init_patch = patch(
            "vllm.v1.worker.gpu_worker.init_worker_distributed_environment",
            side_effect=make_operation_tracker(
                "init_distributed", original_init_worker
            ),
        )
        memory_patch = patch.object(
            MemorySnapshot,
            "__init__",
            make_operation_tracker("memory_snapshot", original_memory_snapshot_init),
        )
        all_reduce_patch = patch(
            "torch.distributed.all_reduce",
            side_effect=make_operation_tracker("nccl_all_reduce", original_all_reduce),
        )

        with init_patch, memory_patch, all_reduce_patch:
            # Initialize device (this is where we test the order)
            worker.init_device()

            # Load model to ensure everything works
            worker.load_model()

        # Signal success
        queue.put(("success", rank))

    except Exception as e:
        error_queue.put((rank, str(e), type(e).__name__))
        raise


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for tensor parallelism"
)
def test_init_distributed_is_called_before_memory_snapshot():
    """Test that distributed env is setup before memory snapshot.

    This test makes sure during worker initialization, the initial memory
    snapshot is taken after distributed env is setup to include all the buffers
    allocated by distributed env.
    """
    world_size = 2

    # Create a temporary file for distributed init
    with tempfile.NamedTemporaryFile(delete=False) as f:
        distributed_init_method = f"file://{f.name}"

    # Create queues for inter-process communication
    ctx = mp.get_context("spawn")
    operation_queue = ctx.Queue()
    error_queue = ctx.Queue()

    # Start worker processes
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker_process,
            args=(
                rank,
                world_size,
                distributed_init_method,
                operation_queue,
                error_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=60)  # 60 second timeout

    # Check for errors
    errors = []
    while not error_queue.empty():
        rank, error_msg, error_type = error_queue.get()
        errors.append(f"Rank {rank}: {error_type}: {error_msg}")

    if errors:
        pytest.fail("Worker processes failed:\n" + "\n".join(errors))

    # Collect all operations from the queue
    operations = []
    while not operation_queue.empty():
        operations.append(operation_queue.get())

    # Verify we got operations from both ranks
    print(f"Collected operations: {operations}")

    # Check operations for each rank
    for rank in range(world_size):
        rank_ops = [op for op, r in operations if r == rank]
        print(f"\nRank {rank} operations: {rank_ops}")

        # Raises ValueError if the operation is not found
        init_distributed = rank_ops.index("init_distributed")
        nccl_all_reduce = rank_ops.index("nccl_all_reduce")
        memory_snapshot = rank_ops.index("memory_snapshot")

        # Verify order: init_distributed should happen before memory_snapshot
        assert init_distributed < nccl_all_reduce < memory_snapshot, (
            f"Rank {rank}: init_distributed (index {init_distributed}) "
            f"must happen before nccl_all_reduce (index {nccl_all_reduce}) "
            f"and memory_snapshot (index {memory_snapshot})"
        )

    # Clean up
    os.unlink(distributed_init_method.replace("file://", ""))
