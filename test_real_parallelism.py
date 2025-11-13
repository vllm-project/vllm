#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple test demonstrating vLLM's parallel context with REAL tensor parallelism.

Run with:
    torchrun --nproc_per_node=2 test_real_parallelism.py

Or for 4 GPUs:
    torchrun --nproc_per_node=4 test_real_parallelism.py

This will spawn multiple processes (one per GPU) and actually initialize
process groups for tensor parallelism.
"""

import os

import torch
import torch.distributed as dist

from vllm.distributed import parallel_state
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_context import ParallelContext


def init_distributed():
    """Initialize distributed environment (normally vLLM does this)."""
    # This is what torchrun sets up
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}/{world_size}] Initializing distributed environment...")

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
        )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}] Using device: {device}")
    return rank, world_size, local_rank, device


def init_vllm_parallel_state(world_size):
    """Initialize vLLM's parallel state (normally vLLM does this)."""
    # For simplicity, use all GPUs for tensor parallelism
    tensor_model_parallel_size = world_size
    pipeline_model_parallel_size = 1

    # This is what vLLM does internally
    parallel_state.ensure_model_parallel_initialized(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    print(
        f"[Rank {dist.get_rank()}] Initialized TP={tensor_model_parallel_size}, "
        f"PP={pipeline_model_parallel_size}"
    )


def test_parallel_context():
    """Test that ParallelContext correctly reflects actual parallel state."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create parallel context from vLLM's parallel config
    # (In real vLLM, this is created automatically)
    ctx = ParallelContext(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
        data_parallel_size=1,
    )

    # Now these should return REAL values, not fallbacks!
    tp_rank = ctx.get_tensor_parallel_rank()
    tp_size = ctx.get_tensor_parallel_world_size()
    pp_rank = ctx.get_pipeline_parallel_rank()

    print(f"[Rank {rank}] ParallelContext reports:")
    print(f"  TP rank: {tp_rank} / {tp_size}")
    print(f"  PP rank: {pp_rank}")

    # Verify it matches actual distributed state
    assert tp_rank == rank, f"TP rank mismatch: {tp_rank} != {rank}"
    assert tp_size == world_size, f"TP size mismatch: {tp_size} != {world_size}"

    print(f"[Rank {rank}] ✓ ParallelContext is correct!")


def test_callback_with_real_parallelism():
    """Test model registration callback with real parallel context."""
    rank = dist.get_rank()

    # Define a factory that uses parallel context
    def build_parallel_model(vllm_config, parallel_context):
        tp_rank = parallel_context.get_tensor_parallel_rank()
        tp_size = parallel_context.get_tensor_parallel_world_size()

        print(f"[Rank {rank}] Factory called with:")
        print(f"  TP rank: {tp_rank}")
        print(f"  TP size: {tp_size}")

        # Verify we got REAL ranks, not fallbacks
        assert tp_rank == rank, "Factory should receive actual rank"
        assert tp_size == dist.get_world_size(), (
            "Factory should receive actual world size"
        )

        # Create a simple model (just for demonstration)
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self, rank, size):
                super().__init__()
                self.rank = rank
                self.size = size
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel(tp_rank, tp_size)
        print(
            f"[Rank {rank}] ✓ Model created with TP rank={model.rank}, "
            f"size={model.size}"
        )

        return model

    # Register with callback
    ModelRegistry.register_model("ParallelTestModel", build_parallel_model)

    # Load the model (simulating what vLLM does)
    model_cls = ModelRegistry._try_load_model_cls("ParallelTestModel")

    # Create parallel context
    ctx = ParallelContext(
        tensor_model_parallel_size=dist.get_world_size(),
        pipeline_model_parallel_size=1,
    )

    # Instantiate model (this calls our factory)
    mock_config = type("Config", (), {"hf_config": None, "parallel_config": None})()
    model = model_cls(vllm_config=mock_config, parallel_context=ctx)

    # Verify model was created with correct rank
    assert model.rank == rank, f"Model should have rank {rank}, got {model.rank}"

    print(f"[Rank {rank}] ✓ Callback registration works with real parallelism!")


def main():
    # Initialize distributed
    rank, world_size, local_rank, device = init_distributed()

    # Initialize vLLM's parallel state
    init_vllm_parallel_state(world_size)

    # Run tests
    print(f"\n[Rank {rank}] ========== Test 1: ParallelContext ==========")
    test_parallel_context()

    print(f"\n[Rank {rank}] ========== Test 2: Callback with Parallelism ==========")
    test_callback_with_real_parallelism()

    # Synchronize before finishing
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("✓ All tests passed on all ranks!")
        print("=" * 60)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
