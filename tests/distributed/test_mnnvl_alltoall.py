# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MNNVL AllToAll operations.

Requires: docker run ... --cap-add=SYS_PTRACE ...
Run: pytest tests/distributed/test_mnnvl_alltoall.py -v
"""

import os
import traceback

import pytest
import torch
import torch.multiprocessing as mp

from vllm.distributed import get_ep_group
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.network_utils import get_open_port

from ..utils import init_test_distributed_environment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_sys_ptrace() -> bool:
    """Check for SYS_PTRACE capability (bit 19 in CapEff)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    return bool(int(line.split()[1], 16) & (1 << 19))
    except Exception:
        pass
    return False


def _spawn_workers(worker_fn, world_size, *, dp_size=None):
    """Spawn one process per GPU, run worker_fn, assert all succeed.

    Uses an mp.Queue to propagate worker tracebacks back to the parent
    so pytest shows the actual failure, not just an exit code.
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    port = str(get_open_port())
    # Allocate a second port for DP master when dp_size is set, so the
    # distributed init port and DP port can't collide even under xdist.
    dp_port = str(get_open_port()) if dp_size is not None else None
    err_queue: mp.Queue = mp.Queue()
    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=_run_worker,
            args=(rank, world_size, port, worker_fn, dp_size, dp_port,
                  err_queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Collect any errors from workers before asserting.
    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())
    err_queue.close()
    err_queue.join_thread()
    if errors:
        pytest.fail("Worker(s) failed:\n" + "\n---\n".join(errors))


def _run_worker(rank, world_size, port, worker_fn, dp_size, dp_port,
                err_queue):
    """Per-process setup: device, distributed env, then call worker_fn.

    Args:
        dp_size: If set, initialize with tp=1 and data_parallel_size=dp_size.
                 Otherwise use tp=world_size (default for EP-based tests).
        dp_port: Separate port for the DP master (only used when dp_size is set).
        err_queue: Queue for propagating tracebacks to the parent process.
    """
    try:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        torch.accelerator.set_device_index(rank)
        if dp_size is not None:
            _init_dp_environment(world_size, rank, port, dp_size, dp_port)
        else:
            init_test_distributed_environment(world_size, 1, rank, port)
        worker_fn(rank, world_size)
        torch.distributed.barrier()
    except Exception:
        err_queue.put(f"[Rank {rank}]\n{traceback.format_exc()}")
        # Don't re-raise: the parent reads errors from err_queue.
        # A non-zero exit from the re-raise would be redundant.
        import sys
        sys.exit(1)


def _init_dp_environment(world_size, rank, port, dp_size, dp_port):
    """Initialize distributed env with data parallelism.

    Sets up tp=1, pp=1, dp=dp_size. Each process is one DP rank
    with local rank 0 within its (trivial) tp*pp group.

    Args:
        port: Port for torch.distributed init.
        dp_port: Separate port for the DP master group init.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=dp_size,
        data_parallel_rank=rank,
        # Pre-populate port list so __post_init__ doesn't auto-generate
        # random ports. All DP ranks must agree on the same port.
        _data_parallel_master_port_list=[int(dp_port)],
    )
    with set_current_vllm_config(vllm_config):
        # rank=0 here because each DP rank has a single (tp=1,pp=1) process,
        # so the local rank within the tp*pp group is always 0.
        # init_distributed_environment will offset by data_parallel_rank.
        init_distributed_environment(
            world_size=1,  # tp * pp = 1
            rank=0,
            distributed_init_method=f"tcp://localhost:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(1, 1)


def _make_forward_context(rank, world_size, num_tokens_per_rank):
    """Create a forward context with mock DP metadata for AgRs tests.

    Returns a context manager suitable for ``with`` statements.
    The real DPMetadata (with sp_local_sizes etc.) is created internally
    by set_forward_context from num_tokens_across_dp; the attn_metadata
    placeholder just satisfies the "attn_metadata is not None" guard.
    """
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig
    from vllm.forward_context import set_forward_context

    class _AttnMeta:
        """Minimal placeholder so set_forward_context's
        ``attn_metadata is not None`` guard (forward_context.py:334)
        is satisfied. The real DPMetadata is built from num_tokens_across_dp."""
        dp_metadata = None

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=world_size,
        is_moe_model=True,
        data_parallel_rank=rank,
    )
    return set_forward_context(
        _AttnMeta(),
        vllm_config,
        num_tokens=num_tokens_per_rank,
        num_tokens_across_dp=torch.tensor(
            [num_tokens_per_rank] * world_size, dtype=torch.int
        ),
    )


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

requires_multi_gpu = pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="Need >= 2 GPUs"
)
requires_two_sided = pytest.mark.skipif(
    not has_flashinfer_nvlink_two_sided(),
    reason="FlashInfer NVLink two-sided not available",
)
requires_one_sided = pytest.mark.skipif(
    not has_flashinfer_nvlink_one_sided(),
    reason="FlashInfer NVLink one-sided not available",
)
requires_ptrace = pytest.mark.skipif(
    not _has_sys_ptrace(),
    reason="SYS_PTRACE required (docker run --cap-add=SYS_PTRACE)",
)

# Skip entire module if neither backend is available
pytestmark = pytest.mark.skipif(
    not (has_flashinfer_nvlink_two_sided() or has_flashinfer_nvlink_one_sided()),
    reason="FlashInfer NVLink backends not available",
)


# ---------------------------------------------------------------------------
# Test 1: Two-sided manager lifecycle (init, cleanup, reinit, ensure_init)
# ---------------------------------------------------------------------------
#
# Uses EP group (get_ep_group) because the two-sided manager is constructed
# with an EP-scoped communicator in production. With tp=world_size the EP
# group spans all ranks, giving us a multi-rank group for testing.
# ---------------------------------------------------------------------------


def _two_sided_lifecycle_worker(rank, world_size):
    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkTwoSidedManager,
    )

    cpu_group = get_ep_group().cpu_group
    num_gpus = torch.accelerator.device_count()
    manager = FlashInferNVLinkTwoSidedManager(cpu_group)

    # Not initialized yet
    assert not manager.initialized
    assert manager.rank == rank
    assert manager.world_size == world_size

    # Initialize
    manager.initialize(world_size=world_size, rank=rank, gpus_per_node=num_gpus)
    assert manager.initialized
    assert manager.workspace_tensor is not None
    assert manager.prepare_workspace_tensor is not None
    assert manager.mapping is not None

    torch.distributed.barrier()

    # Cleanup
    manager.cleanup()
    assert not manager.initialized
    assert manager.workspace_tensor is None
    assert manager.prepare_workspace_tensor is None

    torch.distributed.barrier()

    # Reinitialize
    manager.initialize(world_size=world_size, rank=rank, gpus_per_node=num_gpus)
    assert manager.initialized

    torch.distributed.barrier()

    # ensure_alltoall_workspace_initialized is idempotent when already init'd
    assert manager.ensure_alltoall_workspace_initialized()
    assert manager.initialized

    manager.cleanup()
    assert not manager.initialized


@requires_multi_gpu
@requires_two_sided
@requires_ptrace
@pytest.mark.parametrize("world_size", [2])
def test_two_sided_manager_lifecycle(world_size):
    """Test init, cleanup, reinit, and ensure_initialized idempotency."""
    _spawn_workers(_two_sided_lifecycle_worker, world_size)


# ---------------------------------------------------------------------------
# Test 2: One-sided manager lifecycle (init, cleanup, reinit)
# ---------------------------------------------------------------------------
#
# Uses DP group (get_dp_group) because the one-sided manager's initialize()
# internally calls get_dp_group() to set up the MnnvlConfig communicator.
# We therefore need a real DP group with world_size > 1, which requires
# dp_size=world_size via _init_dp_environment.
# ---------------------------------------------------------------------------


def _one_sided_lifecycle_worker(rank, world_size):
    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkOneSidedManager,
    )
    from vllm.distributed.parallel_state import get_dp_group

    cpu_group = get_dp_group().cpu_group
    manager = FlashInferNVLinkOneSidedManager(cpu_group)

    assert not manager.initialized
    assert manager.rank == rank
    assert manager.world_size == world_size

    init_kwargs = dict(
        max_num_tokens=1024, top_k=2,
        num_experts=world_size * 8, hidden_size=4096,
    )

    # Initialize
    manager.initialize(**init_kwargs)
    assert manager.initialized
    assert manager.moe_alltoall is not None
    assert manager.mapping is not None

    torch.distributed.barrier()

    # Cleanup
    manager.cleanup()
    assert not manager.initialized
    assert manager.moe_alltoall is None

    torch.distributed.barrier()

    # Reinitialize with different token count
    manager.initialize(**{**init_kwargs, "max_num_tokens": 2048})
    assert manager.initialized

    torch.distributed.barrier()
    manager.cleanup()


@requires_multi_gpu
@requires_one_sided
@requires_ptrace
@pytest.mark.parametrize("world_size", [2])
def test_one_sided_manager_lifecycle(world_size):
    """Test init, cleanup, and reinit with different params."""
    _spawn_workers(
        _one_sided_lifecycle_worker, world_size, dp_size=world_size,
    )


# ---------------------------------------------------------------------------
# Test 3: AgRs dispatch/combine with value validation
# ---------------------------------------------------------------------------


def _agrs_dispatch_combine_worker(rank, world_size):
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
    from vllm.forward_context import get_forward_context

    cpu_group = get_ep_group().cpu_group
    device = torch.device(f"cuda:{rank}")

    hidden_size = 64
    tokens_per_rank = 16
    experts_per_token = 2
    num_experts = world_size * 4
    total_tokens = world_size * tokens_per_rank

    # Deterministic per-rank data: rank r has value (r + 1)
    hidden = torch.full(
        (tokens_per_rank, hidden_size),
        float(rank + 1), device=device, dtype=torch.float32,
    )
    router = torch.full(
        (tokens_per_rank, num_experts),
        float(rank + 1) * 10, device=device, dtype=torch.float32,
    )
    weights = torch.full(
        (tokens_per_rank, experts_per_token),
        float(rank + 1) * 100, device=device, dtype=torch.float32,
    )
    ids = torch.full(
        (tokens_per_rank, experts_per_token),
        rank, device=device, dtype=torch.long,
    )

    with _make_forward_context(rank, world_size, tokens_per_rank):
        manager = AgRsAll2AllManager(cpu_group)
        dp_metadata = get_forward_context().dp_metadata

        with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
            # -- dispatch_router_logits --
            d_hidden, d_router = manager.dispatch_router_logits(
                hidden.clone(), router.clone(), is_sequence_parallel=True,
            )
            assert d_hidden.shape == (total_tokens, hidden_size)
            assert d_router.shape == (total_tokens, num_experts)

            for r in range(world_size):
                s = r * tokens_per_rank
                e = (r + 1) * tokens_per_rank
                torch.testing.assert_close(
                    d_hidden[s:e],
                    torch.full_like(d_hidden[s:e], float(r + 1)),
                )
                torch.testing.assert_close(
                    d_router[s:e],
                    torch.full_like(d_router[s:e], float(r + 1) * 10),
                )

            # -- dispatch --
            d_hidden2, d_weights, d_ids = manager.dispatch(
                hidden.clone(), weights.clone(), ids.clone(),
                is_sequence_parallel=True,
            )
            assert d_hidden2.shape == (total_tokens, hidden_size)
            assert d_weights.shape == (total_tokens, experts_per_token)
            assert d_ids.shape == (total_tokens, experts_per_token)

            for r in range(world_size):
                s = r * tokens_per_rank
                e = (r + 1) * tokens_per_rank
                torch.testing.assert_close(
                    d_weights[s:e],
                    torch.full_like(d_weights[s:e], float(r + 1) * 100),
                )
                assert (d_ids[s:e] == r).all()

            # -- combine (reduce-scatter) --
            # Each token i has value i in all columns; after reduce-scatter
            # each rank gets its slice, summed across ranks.
            expert_out = (
                torch.arange(total_tokens, device=device, dtype=torch.float32)
                .unsqueeze(1)
                .expand(total_tokens, hidden_size)
                .contiguous()
            )

            combined = manager.combine(expert_out, is_sequence_parallel=True)
            assert combined.shape == (tokens_per_rank, hidden_size)

            for i in range(tokens_per_rank):
                expected_val = float(rank * tokens_per_rank + i) * world_size
                torch.testing.assert_close(
                    combined[i],
                    torch.full_like(combined[i], expected_val),
                )

            torch.distributed.barrier()


@requires_multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_agrs_dispatch_combine(world_size):
    """Validate dispatch gathers all-rank data and combine reduces correctly."""
    _spawn_workers(_agrs_dispatch_combine_worker, world_size)
