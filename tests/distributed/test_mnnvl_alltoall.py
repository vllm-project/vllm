# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Direct unit and component tests for MNNVL AllToAll operations.
requires container ran w/ docker run ... --cap-add=SYS_PTRACE ...
Run `pytest tests/distributed/test_mnnvl_alltoall.py`.
"""

import os
import subprocess

import pytest
import torch
import torch.distributed as dist

from vllm.distributed import get_ep_group
from vllm.utils.flashinfer import has_flashinfer_all2all

from ..utils import init_test_distributed_environment

# Skip all tests if FlashInfer alltoall is not available
pytestmark = pytest.mark.skipif(
    not has_flashinfer_all2all(),
    reason="FlashInfer alltoall not available",
)


def has_sys_ptrace_capability() -> bool:
    """
    Check if the process has SYS_PTRACE capability.

    MNNVL (Multi-Node NVLink) requires SYS_PTRACE to share memory file descriptors
    between processes using pidfd_getfd() system call.

    Returns:
        True if SYS_PTRACE is available, False otherwise.
    """
    try:
        # Try to check capabilities using capsh if available
        result = subprocess.run(
            ["capsh", "--print"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "cap_sys_ptrace" in result.stdout.lower():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Alternative check: try to read /proc/self/status
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    # SYS_PTRACE is capability bit 19 (0x80000 = 1 << 19)
                    cap_eff = int(line.split()[1], 16)
                    # Check if bit 19 is set
                    return bool(cap_eff & (1 << 19))
    except Exception:
        pass

    # If we can't determine, assume it's not available in container environments
    # Check if we're in a container; if not, assume it's available
    return not (
        os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    )


def test_flashinfer_all2all_import():
    """Test that we can import FlashInfer alltoall components."""
    try:
        from flashinfer.comm import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig
        from flashinfer.comm.trtllm_alltoall import MnnvlMoe
        from vllm.distributed.device_communicators.all2all import (
            FlashInferAllToAllManager,
        )
        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        assert Mapping is not None
        assert MnnvlConfig is not None
        assert MnnvlMoe is not None
        assert FlashInferAllToAllManager is not None
        assert CustomCommunicator is not None
    except ImportError as e:
        pytest.fail(f"Failed to import FlashInfer alltoall components: {e}")


def run_multi_gpu_test(rank: int, world_size: int, port: str, test_func):
    """Helper to run a test function in a multi-GPU distributed environment."""
    # Remove CUDA_VISIBLE_DEVICES to allow access to all GPUs
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Set device for this rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Initialize distributed environment
    # Use world_size for tp to create multi-process setup
    init_test_distributed_environment(world_size, 1, rank, port)

    # Verify multi-GPU setup
    assert torch.distributed.is_initialized()
    assert torch.distributed.get_world_size() == world_size
    assert torch.distributed.get_rank() == rank

    print(f"\n[Rank {rank}] GPU: {torch.cuda.current_device()}, "
          f"World size: {torch.distributed.get_world_size()}")

    # Run the actual test
    test_func(rank, world_size)

    # Synchronize before cleanup
    torch.distributed.barrier()


def manager_initialization_worker(rank: int, world_size: int):
    """Worker function for testing FlashInferAllToAllManager initialization."""
    from vllm.distributed.device_communicators.all2all import (
        FlashInferAllToAllManager,
    )

    # Get CPU group from EP
    cpu_group = get_ep_group().cpu_group

    # Create manager
    manager = FlashInferAllToAllManager(cpu_group)

    # Verify multi-GPU properties
    print(
        f"[Rank {rank}] Manager rank: {manager.rank}, "
        f"world_size: {manager.world_size}"
    )
    assert manager is not None
    assert manager.rank == rank
    assert manager.world_size == world_size
    assert not manager.initialized

    # Test workspace initialization - should work with world_size > 1
    manager.initialize(
        world_size=world_size,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
    )

    assert manager.initialized
    assert manager.workspace_tensor is not None
    assert manager.prepare_workspace_tensor is not None
    assert manager.mapping is not None

    print(f"[Rank {rank}] Manager initialized successfully")

    # Synchronize before cleanup
    torch.distributed.barrier()

    # Test cleanup
    manager.cleanup()
    assert not manager.initialized
    assert manager.workspace_tensor is None
    assert manager.prepare_workspace_tensor is None

    print(f"[Rank {rank}] Manager cleanup successful")


def workspace_reinitialization_worker(rank: int, world_size: int):
    """Worker function for testing workspace reinitialization."""
    from vllm.distributed.device_communicators.all2all import (
        FlashInferAllToAllManager,
    )

    cpu_group = get_ep_group().cpu_group
    manager = FlashInferAllToAllManager(cpu_group)

    # Initialize
    manager.initialize(
        world_size=world_size,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
    )
    assert manager.initialized
    print(f"[Rank {rank}] First initialization complete")

    torch.distributed.barrier()

    # Cleanup
    manager.cleanup()
    assert not manager.initialized
    print(f"[Rank {rank}] Cleanup complete")

    torch.distributed.barrier()

    # Re-initialize
    manager.initialize(
        world_size=world_size,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
    )
    assert manager.initialized
    print(f"[Rank {rank}] Re-initialization complete")

    torch.distributed.barrier()

    manager.cleanup()


def ensure_initialized_worker(rank: int, world_size: int):
    """Worker function for testing ensure_alltoall_workspace_initialized."""
    from vllm.distributed.device_communicators.all2all import (
        FlashInferAllToAllManager,
    )

    cpu_group = get_ep_group().cpu_group
    manager = FlashInferAllToAllManager(cpu_group)

    # Should not be initialized yet
    assert not manager.initialized

    # Call ensure - should initialize with world_size > 1
    result = manager.ensure_alltoall_workspace_initialized()
    assert result
    assert manager.initialized
    print(f"[Rank {rank}] ensure_initialized returned True, manager initialized")

    torch.distributed.barrier()

    # Call again - should return True without re-initializing
    result = manager.ensure_alltoall_workspace_initialized()
    assert result
    assert manager.initialized
    print(f"[Rank {rank}] ensure_initialized (2nd call) returned True")

    torch.distributed.barrier()

    manager.cleanup()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs")
@pytest.mark.skipif(
    not has_sys_ptrace_capability(),
    reason=(
        "SYS_PTRACE capability required for MNNVL. "
        "Run container with: docker run --cap-add=SYS_PTRACE"
    ),
)
@pytest.mark.parametrize("world_size", [2])
def test_flashinfer_alltoall_manager_initialization(world_size: int):
    """
    Test FlashInferAllToAllManager initialization with multiple GPUs.

    This test spawns multiple processes (one per GPU) to test actual multi-GPU
    AllToAll operations. Requires SYS_PTRACE capability for MNNVL memory sharing.
    """
    import torch.multiprocessing as mp

    # Use spawn method for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    port = "12355"

    # Launch multiple processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_multi_gpu_test,
            args=(rank, world_size, port, manager_initialization_worker)
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()
        assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs")
@pytest.mark.skipif(
    not has_sys_ptrace_capability(),
    reason=(
        "SYS_PTRACE capability required for MNNVL. "
        "Run container with: docker run --cap-add=SYS_PTRACE"
    ),
)
@pytest.mark.parametrize("world_size", [2])
def test_flashinfer_alltoall_workspace_reinitialization(world_size: int):
    """
    Test that workspace can be reinitialized with multiple GPUs.

    This test spawns multiple processes to test workspace reinitialization.
    Requires SYS_PTRACE capability for MNNVL memory sharing.
    """
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    port = "12356"

    # Launch multiple processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_multi_gpu_test,
            args=(rank, world_size, port, workspace_reinitialization_worker)
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()
        assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs")
@pytest.mark.skipif(
    not has_sys_ptrace_capability(),
    reason=(
        "SYS_PTRACE capability required for MNNVL. "
        "Run container with: docker run --cap-add=SYS_PTRACE"
    ),
)
@pytest.mark.parametrize("world_size", [2])
def test_flashinfer_alltoall_ensure_initialized(world_size: int):
    """
    Test ensure_alltoall_workspace_initialized with multiple GPUs.

    This test spawns multiple processes to test the ensure_initialized method.
    Requires SYS_PTRACE capability for MNNVL memory sharing.
    """
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    port = "12357"

    # Launch multiple processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_multi_gpu_test,
            args=(rank, world_size, port, ensure_initialized_worker)
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()
        assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"


def test_custom_communicator():
    """Test CustomCommunicator wrapper for FlashInfer."""
    if not has_flashinfer_all2all():
        pytest.skip("FlashInfer alltoall not available")

    from vllm.distributed.device_communicators.mnnvl_compat import (
        CustomCommunicator,
    )

    class MockGroup:
        def rank(self):
            return 0

        def size(self):
            return 2

    mock_group = MockGroup()
    comm = CustomCommunicator(mock_group)

    # Test basic methods
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 2

    # Test unimplemented methods raise NotImplementedError
    with pytest.raises(NotImplementedError):
        comm.bcast(None, 0)

    with pytest.raises(NotImplementedError):
        comm.barrier()

    # Test Split returns self (as per implementation)
    split_comm = comm.Split(0, 0)
    assert split_comm is comm


if __name__ == "__main__":
    # Check SYS_PTRACE capability first
    print("=" * 70)
    print("MNNVL AllToAll Test Configuration")
    print("=" * 70)

    print(f"\nGPUs available: {torch.cuda.device_count()}")
    print(f"FlashInfer AllToAll available: {has_flashinfer_all2all()}")

    if has_sys_ptrace_capability():
        print("✓ SYS_PTRACE capability: DETECTED")
        print("  Multi-GPU tests will run")
    else:
        print("⚠ SYS_PTRACE capability: NOT DETECTED")
        print("  Multi-GPU tests will be skipped")
        print("  To enable: docker run --cap-add=SYS_PTRACE ...")

    print("\n" + "=" * 70)
    print("Running Standalone Tests")
    print("=" * 70 + "\n")

    # Run basic import test
    test_flashinfer_all2all_import()
    print("✓ Import test passed")

    # Run communicator test
    test_custom_communicator()
    print("✓ Custom communicator test passed")

    print("\n" + "=" * 70)
    print("All standalone tests passed!")
    print("=" * 70)
    print("\nTo run full multi-GPU test suite:")
    print("  pytest tests/distributed/test_mnnvl_alltoall.py -v")
