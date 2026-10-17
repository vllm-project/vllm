# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MNNVL AllToAll operations.

Requires: docker run ... --cap-add=SYS_PTRACE ...
Run: pytest tests/distributed/test_mnnvl_alltoall.py -v
"""

import os
import traceback
from contextlib import contextmanager
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

from vllm.distributed import get_ep_group
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.import_utils import has_deep_ep_v2
from vllm.utils.network_utils import get_open_port

from ..utils import init_test_distributed_environment

DEVICE = current_platform.device_type

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
            args=(rank, world_size, port, worker_fn, dp_size, dp_port, err_queue),
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
        combined = "\n---\n".join(errors)
        if "NCCL GIN" in combined:
            pytest.skip("NCCL GIN not available on this system")
        pytest.fail("Worker(s) failed:\n" + combined)
    assert all(p.exitcode == 0 for p in procs), (
        f"Worker exited without reporting a traceback: {[p.exitcode for p in procs]}"
    )


def _run_worker(rank, world_size, port, worker_fn, dp_size, dp_port, err_queue):
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
        if has_flashinfer_nvlink_one_sided():
            from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll

            # Release cached tensor deleters while their JIT library is loaded,
            # rather than during Python's extension-module finalization.
            torch.accelerator.synchronize()
            MoeAlltoAll._WORKSPACE_CACHE.clear()
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


@contextmanager
def _make_forward_context(
    rank, world_size, num_tokens_per_rank, num_tokens_across_dp=None
):
    """Keep the vLLM config and mock-DP forward context active together.

    Returns a context manager suitable for ``with`` statements.
    The real DPMetadata (with sp_local_sizes etc.) is created internally
    by set_forward_context from num_tokens_across_dp; the attn_metadata
    placeholder just satisfies the "attn_metadata is not None" guard.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
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
    if num_tokens_across_dp is None:
        num_tokens_across_dp = [num_tokens_per_rank] * world_size
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(
            _AttnMeta(),
            vllm_config,
            num_tokens=num_tokens_per_rank,
            num_tokens_across_dp=torch.tensor(num_tokens_across_dp, dtype=torch.int),
        ),
    ):
        yield


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
requires_deep_ep_v2 = pytest.mark.skipif(
    not has_deep_ep_v2(),
    reason="DeepEP v2 (ElasticBuffer) not available or NCCL < 2.30.4",
)

# NOTE: No module-level pytestmark here. The FlashInfer lifecycle tests have
# their own @requires_two_sided / @requires_one_sided decorators, and
# test_args_dispatch_combine uses only standard torch.distributed ops and
# should run even when FlashInfer NVLink backends are not installed.


# ---------------------------------------------------------------------------
# Test 1: Two-sided manager lifecycle (init, cleanup, reinit, ensure_init)
# ---------------------------------------------------------------------------
#
# Tests FlashInferNVLinkTwoSidedManager which wraps FlashInfer's MnnvlMoe.
# initialize() allocates MNNVL shared workspaces via MnnvlMoe.get_moe_workspaces,
# which uses pidfd_getfd() to share memory file descriptors across processes —
# hence the SYS_PTRACE requirement.
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
# With TP=PCP=PP=1, the DP and EP groups have the same ranks.
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
        max_num_tokens=1024,
        top_k=2,
        num_experts=world_size * 8,
        hidden_size=4096,
    )

    # Model construction can set CUDA as the default device. Bootstrap
    # metadata must still be allocated on the CPU.
    with torch.device(f"{DEVICE}:{rank}"):
        manager.initialize(**init_kwargs)
        assert torch.get_default_device() == torch.device(f"{DEVICE}:{rank}")
    assert manager.initialized
    assert manager.get_communication().params.hidden_size == 4096

    torch.distributed.barrier()

    # Cleanup
    manager.cleanup()
    assert not manager.initialized
    assert manager.communication is None

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
        _one_sided_lifecycle_worker,
        world_size,
        dp_size=world_size,
    )


# ---------------------------------------------------------------------------
# Test 2b: One-sided manager grows workspace across heterogeneous MoE layers
# ---------------------------------------------------------------------------
#
# Share the largest token, dispatch-row and combine-row capacities across layers.
# ---------------------------------------------------------------------------


def _one_sided_workspace_grow_worker(rank, world_size):
    from flashinfer.fused_moe import QuantFormat

    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkOneSidedManager,
    )
    from vllm.distributed.parallel_state import get_dp_group

    cpu_group = get_dp_group().cpu_group
    manager = FlashInferNVLinkOneSidedManager(cpu_group)

    base_kwargs = dict(
        max_num_tokens=1024,
        top_k=2,
        num_experts=world_size * 8,
        hidden_size=4096,
    )
    nvfp4_kwargs = dict(dispatch_format=QuantFormat.NVFP4)
    bf16_kwargs = dict(dispatch_format=QuantFormat.BF16)

    # A quantized-only layer reserves its actual dispatch row size.
    manager.initialize(**base_kwargs, **nvfp4_kwargs)
    original = manager.get_communication()
    assert original.params.dispatch_format == QuantFormat.NVFP4
    assert original.params.dispatch_bytes_per_token == 4096 // 2 + 4096 // 16
    assert original.config.extra_payload_bytes_per_token == 0
    manager.initialize(**base_kwargs, **bf16_kwargs)
    bf16 = manager.get_communication()
    assert bf16 is not original
    assert bf16.params.dispatch_bytes_per_token == 8192

    # Token growth for a smaller format must retain earlier BF16 capacity.
    manager.initialize(**{**base_kwargs, "max_num_tokens": 2048}, **nvfp4_kwargs)
    grown = manager.get_communication()
    assert grown is not bf16
    assert grown.params.max_tokens_per_rank == 2048
    assert grown.params.dispatch_bytes_per_token == 8192

    # A BF16 row plus scales needs additional dispatch capacity.
    manager.initialize(**base_kwargs, extra_payload_bytes_per_token=256)
    with_scales = manager.get_communication()
    assert with_scales is not grown
    assert with_scales.params.max_tokens_per_rank == 2048
    assert with_scales.config.extra_payload_bytes_per_token == 256
    manager.initialize(**base_kwargs, **nvfp4_kwargs)
    assert manager.get_communication() is with_scales

    # A smaller hidden size shares the existing instance, without shrinking it.
    manager.initialize(
        **{**base_kwargs, "hidden_size": 2048},
    )
    assert manager.get_communication() is with_scales

    # A wider NVFP4 layer needs more combine capacity, but fewer dispatch bytes
    # than the narrower BF16 layer. Preserve both requirements independently.
    manager.initialize(**{**base_kwargs, "hidden_size": 8192}, **nvfp4_kwargs)
    wider = manager.get_communication()
    assert wider is not with_scales
    assert wider.params.hidden_size == 8192
    assert wider.params.max_tokens_per_rank == 2048
    assert (
        wider.params.dispatch_bytes_per_token
        + wider.config.extra_payload_bytes_per_token
    ) == 8192 + 256

    # Smaller BF16 rows can increase dispatch capacity without changing the
    # largest hidden size or the combine capacity.
    manager.initialize(**{**base_kwargs, "hidden_size": 6144}, **bf16_kwargs)
    shared = manager.get_communication()
    assert shared.params.hidden_size == 8192
    assert shared.params.max_tokens_per_rank == 2048
    assert (
        shared.params.dispatch_bytes_per_token
        + shared.config.extra_payload_bytes_per_token
    ) == 6144 * 2

    # Alternate real payload widths/formats on the same workspace. The packed
    # byte case exercises transport; repeat_interleave is synthetic expert work.
    for hidden, packed in [(6144, False), (8192, True), (2048, False)]:
        manager.initialize(
            **{**base_kwargs, "hidden_size": hidden},
            **(nvfp4_kwargs if packed else bf16_kwargs),
        )
        assert manager.get_communication() is shared
        tokens = rank + 3
        device = torch.device(f"{DEVICE}:{rank}")
        width = hidden // 2 if packed else hidden
        x = torch.arange(tokens * width, device=device).view(tokens, width) % 13
        x = (x + rank).to(torch.uint8 if packed else torch.bfloat16)
        top_k = base_kwargs["top_k"]
        ids = torch.arange(top_k, dtype=torch.int32, device=device).repeat(tokens, 1)
        weights = torch.ones(tokens, top_k, device=device)
        received = shared.dispatch(x, ids, weights, max_tokens_per_rank=world_size + 2)
        local = (received.topk_ids >= rank * 8) & (received.topk_ids < (rank + 1) * 8)
        values = received.hidden_states.to(torch.bfloat16)
        expected = x.to(torch.bfloat16)
        if packed:
            values = values.repeat_interleave(2, dim=-1)
            expected = expected.repeat_interleave(2, dim=-1)
        contribution = torch.where(
            local.any(-1, keepdim=True),
            values * local.sum(-1, keepdim=True),
            0,
        )
        output = shared.combine(contribution.contiguous())
        torch.testing.assert_close(output, expected * top_k, rtol=0, atol=0)
    manager.cleanup()


@requires_multi_gpu
@requires_one_sided
@requires_ptrace
@pytest.mark.parametrize("world_size", [2])
def test_one_sided_manager_workspace_grow(world_size):
    """Share one workspace across hidden sizes without losing capacity."""
    _spawn_workers(
        _one_sided_workspace_grow_worker,
        world_size,
        dp_size=world_size,
    )


# ---------------------------------------------------------------------------
# Test 3: AgRs dispatch/combine with value validation
# ---------------------------------------------------------------------------
#
# Tests AgRsAll2AllManager which uses only standard torch.distributed
# all_gatherv / reduce_scatterv — no FlashInfer or MNNVL dependency.
# This test validates the reference all-to-all implementation that other
# backends are compared against.
# ---------------------------------------------------------------------------


class _FakeCombineGroup:
    def __init__(self, combine_input):
        self.combine_input = combine_input
        self.allocation_calls = 0
        self.combined_input = None

    def allocate_combine_input(self, shape, dtype, device, **kwargs):
        self.allocation_calls += 1
        assert self.combine_input is not None
        assert self.combine_input.shape == shape
        assert self.combine_input.dtype == dtype
        assert self.combine_input.device == device
        return self.combine_input

    def combine_into_output(self, hidden_states, output, **kwargs):
        self.combined_input = hidden_states
        output.copy_(hidden_states)
        return output


def test_args_cpu_workspace_allocation(monkeypatch):
    from types import SimpleNamespace

    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    kernel = object.__new__(mk.FusedMoEKernelModularImpl)
    kernel.fused_experts = SimpleNamespace(
        workspace_dtype=lambda dtype: dtype,
        workspace_shapes=lambda *args: ((0,), (0,), (2, 3)),
        finalize_weight_and_reduce_impl=lambda: None,
    )
    kernel.prepare_finalize = SimpleNamespace(
        allocate_fused_expert_output=lambda *args: None,
    )
    monkeypatch.setattr(mk.current_platform, "is_cpu", lambda: True)

    workspace13, workspace2, output, specialized = kernel._allocate_buffers(
        torch.float32,
        torch.device("cpu"),
        2,
        2,
        3,
        3,
        1,
        1,
        1,
        None,
        MoEActivation.SILU,
    )
    assert workspace13.numel() == workspace2.numel() == 0
    assert output.shape == (2, 3)
    assert output.device.type == "cpu"
    assert not specialized


def test_args_finalize_noop_reuses_fused_expert_output(monkeypatch):
    import vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep as pf
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )

    fused_expert_output = torch.arange(6, dtype=torch.float32).view(2, 3)
    output = torch.empty_like(fused_expert_output)
    group = _FakeCombineGroup(combine_input=None)
    monkeypatch.setattr(pf, "get_ep_group", lambda: group)

    impl = pf.MoEPrepareAndFinalizeNaiveDPEPModular()
    impl.finalize(
        output,
        fused_expert_output,
        torch.ones((2, 1)),
        torch.zeros((2, 1), dtype=torch.long),
        False,
        TopKWeightAndReduceNoOP(),
    )

    assert group.allocation_calls == 0
    assert group.combined_input is fused_expert_output
    torch.testing.assert_close(output, fused_expert_output)


def test_args_finalize_contiguous_writes_combine_input(monkeypatch):
    import vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep as pf
    import vllm.model_executor.layers.fused_moe.topk_weight_and_reduce as wr

    fused_expert_output = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    original = fused_expert_output.clone()
    topk_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
    combine_input = torch.empty((2, 3), dtype=torch.float32)
    output = torch.empty_like(combine_input)
    group = _FakeCombineGroup(combine_input)
    monkeypatch.setattr(pf, "get_ep_group", lambda: group)
    monkeypatch.setattr(
        wr.ops,
        "moe_sum",
        lambda hidden_states, out: out.copy_(hidden_states.sum(dim=1)),
    )

    impl = pf.MoEPrepareAndFinalizeNaiveDPEPModular()
    impl.finalize(
        output,
        fused_expert_output,
        topk_weights,
        torch.zeros((2, 2), dtype=torch.long),
        False,
        wr.TopKWeightAndReduceContiguous(),
    )

    expected = (original * topk_weights.unsqueeze(-1)).sum(dim=1)
    assert group.allocation_calls == 1
    assert group.combined_input is combine_input
    torch.testing.assert_close(output, expected)


def _args_dispatch_combine_worker(rank, world_size):
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
    from vllm.forward_context import get_forward_context

    cpu_group = get_ep_group().cpu_group
    device = torch.device(f"{DEVICE}:{rank}")

    hidden_size = 64
    tokens_per_rank = 16
    experts_per_token = 2
    num_experts = world_size * 4
    total_tokens = world_size * tokens_per_rank

    # Deterministic per-rank data: rank r has value (r + 1)
    hidden = torch.full(
        (tokens_per_rank, hidden_size),
        float(rank + 1),
        device=device,
        dtype=torch.float32,
    )
    router = torch.full(
        (tokens_per_rank, num_experts),
        float(rank + 1) * 10,
        device=device,
        dtype=torch.float32,
    )
    weights = torch.full(
        (tokens_per_rank, experts_per_token),
        float(rank + 1) * 100,
        device=device,
        dtype=torch.float32,
    )
    ids = torch.full(
        (tokens_per_rank, experts_per_token),
        rank,
        device=device,
        dtype=torch.long,
    )

    with _make_forward_context(rank, world_size, tokens_per_rank):
        manager = AgRsAll2AllManager(cpu_group)
        dp_metadata = get_forward_context().dp_metadata

        with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
            # -- dispatch_router_logits --
            d_hidden, d_router = manager.dispatch_router_logits(
                hidden.clone(),
                router.clone(),
                is_sequence_parallel=True,
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
                hidden.clone(),
                weights.clone(),
                ids.clone(),
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
            expert_values = (
                torch.arange(total_tokens, device=device, dtype=torch.float32)
                .unsqueeze(1)
                .expand(total_tokens, hidden_size)
                .contiguous()
            )
            expert_out = manager.allocate_combine_input(
                expert_values.shape,
                expert_values.dtype,
                expert_values.device,
                is_sequence_parallel=True,
            )
            if expert_out is None:
                expert_out = expert_values
            else:
                from vllm.distributed.device_communicators.pynccl_allocator import (
                    is_symmetric_memory_tensor,
                )

                assert is_symmetric_memory_tensor(expert_out)
                expert_out.copy_(expert_values)

            combine_output = torch.empty(
                (tokens_per_rank, hidden_size),
                device=device,
                dtype=expert_out.dtype,
            )
            combined = manager.combine_into_output(
                expert_out,
                combine_output,
                is_sequence_parallel=True,
            )
            assert combined.data_ptr() == combine_output.data_ptr()
            assert combined.shape == (tokens_per_rank, hidden_size)

            for i in range(tokens_per_rank):
                expected_val = float(rank * tokens_per_rank + i) * world_size
                torch.testing.assert_close(
                    combined[i],
                    torch.full_like(combined[i], expected_val),
                )

            torch.distributed.barrier()

    uneven_sizes = [tokens_per_rank + r for r in range(world_size)]
    with _make_forward_context(
        rank,
        world_size,
        uneven_sizes[rank],
        num_tokens_across_dp=uneven_sizes,
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
            assert (
                manager.allocate_combine_input(
                    (sum(uneven_sizes), hidden_size),
                    torch.float32,
                    device,
                    is_sequence_parallel=True,
                )
                is None
            )

            total_uneven_tokens = sum(uneven_sizes)
            uneven_input = (
                torch.arange(
                    total_uneven_tokens,
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(1)
                .expand(total_uneven_tokens, hidden_size)
                .contiguous()
            )
            uneven_input.add_(rank)
            uneven_output = torch.empty(
                (uneven_sizes[rank], hidden_size),
                dtype=torch.float32,
                device=device,
            )
            combined = get_ep_group().reduce_scatterv_into_output(
                uneven_input,
                uneven_output,
                dim=0,
                sizes=uneven_sizes,
            )
            assert combined.data_ptr() == uneven_output.data_ptr()

            start = sum(uneven_sizes[:rank])
            expected = torch.arange(
                start,
                start + uneven_sizes[rank],
                dtype=torch.float32,
                device=device,
            ) * world_size + sum(range(world_size))
            expected = expected.unsqueeze(1).expand_as(combined)
            torch.testing.assert_close(combined, expected)


@requires_multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_args_dispatch_combine(world_size):
    """Validate dispatch gathers all-rank data and combine reduces correctly."""
    _spawn_workers(_args_dispatch_combine_worker, world_size)


# ---------------------------------------------------------------------------
# Test 4: FlashInfer two-sided dispatch/combine data communication
# ---------------------------------------------------------------------------
#
# Tests actual data flow through the FlashInfer NVLink two-sided backend
# by calling flashinfer_alltoall_dispatch (with defer_input_quant=True to
# skip quantization) and flashinfer_alltoall_combine, then verifying exact
# round-trip values. Dispatch sends each token once per distinct expert
# rank, and combine performs an unweighted sum, so:
#   dispatch(hidden) → identity → combine = hidden * num_distinct_ranks(i)
# ---------------------------------------------------------------------------


def _two_sided_data_worker(rank, world_size):
    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkTwoSidedManager,
    )
    from vllm.distributed.parallel_state import get_dp_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
        FusedMoEQuantDesc,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided import (  # noqa: E501
        flashinfer_alltoall_combine,
        flashinfer_alltoall_dispatch,
    )

    # Use DP group because MnnvlMoe workspace allocation calls get_dp_group()
    # internally and requires dp_size == ep_size.
    cpu_group = get_dp_group().cpu_group
    device = torch.device(f"{DEVICE}:{rank}")
    num_gpus = torch.accelerator.device_count()

    hidden_size = 128
    tokens_per_rank = 32
    experts_per_token = 2
    num_experts = world_size * 4

    # Initialize the FlashInfer two-sided manager
    manager = FlashInferNVLinkTwoSidedManager(cpu_group)
    manager.initialize(world_size=world_size, rank=rank, gpus_per_node=num_gpus)
    assert manager.initialized

    torch.distributed.barrier()

    # Create deterministic per-rank test data
    torch.manual_seed(rank + 42)
    hidden = torch.randn(
        tokens_per_rank,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    # Assign each token to experts spread across ranks so tokens move between GPUs
    topk_ids = torch.randint(
        0,
        num_experts,
        (tokens_per_rank, experts_per_token),
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.rand(
        tokens_per_rank,
        experts_per_token,
        device=device,
        dtype=torch.float32,
    )

    # Unquantized config: quant_dtype=None means moe_kernel_quantize_input is a no-op
    no_quant = FusedMoEQuantDesc()
    quant_config = FusedMoEQuantConfig(
        _a1=no_quant,
        _a2=no_quant,
        _w1=no_quant,
        _w2=no_quant,
    )
    assert quant_config.quant_dtype is None  # sanity: no quantization

    with _make_forward_context(rank, world_size, tokens_per_rank):
        dp_metadata = get_forward_context().dp_metadata

        with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
            local_sizes = dp_metadata.get_chunk_sizes_across_dp_rank()

            # --- FlashInfer two-sided dispatch ---
            alltoall_info, fi_topk_ids, fi_topk_weights, fi_hidden, fi_scale = (
                flashinfer_alltoall_dispatch(
                    manager,
                    local_sizes,
                    hidden.clone(),
                    None,  # no global scale
                    topk_ids.clone(),
                    topk_weights.clone(),
                    experts_per_token,
                    num_experts,
                    quant_config,
                    defer_input_quant=True,
                )
            )
            assert fi_scale is None  # deferred quant: no scale produced
            assert fi_hidden is not None
            assert fi_hidden.shape[1] == hidden_size
            assert fi_hidden.numel() > 0

            # --- Round-trip exact verification ---
            # The all-to-all sends each token once per *distinct* expert
            # rank. Combine performs an unweighted sum of the per-rank
            # contributions. With identity expert (feeding dispatched
            # hidden straight back):
            #   result[i] = hidden[i] * num_distinct_expert_ranks(i)
            combined = flashinfer_alltoall_combine(
                manager,
                fi_hidden,
                top_k=experts_per_token,
                token_count=tokens_per_rank,
                alltoall_info=alltoall_info,
            )
            assert combined.shape == (tokens_per_rank, hidden_size)

            experts_per_rank = num_experts // world_size
            expert_ranks = topk_ids // experts_per_rank  # (tokens, top_k)
            num_distinct = torch.tensor(
                [len(set(row.tolist())) for row in expert_ranks],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(1)  # (tokens, 1)
            expected = (hidden.float() * num_distinct).to(hidden.dtype)
            torch.testing.assert_close(combined, expected)

            # --- Linearity check with scaled expert output ---
            # Scaling the expert output by a constant should scale the
            # combined result by the same constant.
            scale = 3.0
            combined_scaled = flashinfer_alltoall_combine(
                manager,
                fi_hidden * scale,
                top_k=experts_per_token,
                token_count=tokens_per_rank,
                alltoall_info=alltoall_info,
            )
            expected_scaled = (hidden.float() * num_distinct * scale).to(hidden.dtype)
            torch.testing.assert_close(combined_scaled, expected_scaled)

            torch.distributed.barrier()

    manager.cleanup()


@requires_multi_gpu
@requires_two_sided
@requires_ptrace
@pytest.mark.parametrize("world_size", [2])
def test_two_sided_dispatch_combine(world_size):
    """Test FlashInfer two-sided dispatch/combine with exact value verification."""
    _spawn_workers(_two_sided_data_worker, world_size, dp_size=world_size)


# ---------------------------------------------------------------------------
# Test 5: FlashInfer one-sided dispatch/combine data communication
# ---------------------------------------------------------------------------
#
# Tests actual data flow through the FlashInfer NVLink one-sided backend
# by calling the MoEEpCommunication wrapper directly
# with synthetic payloads, then verifying shapes and round-trip consistency.
# ---------------------------------------------------------------------------


def _one_sided_data_worker(rank, world_size, *, padded_mxfp8):
    from flashinfer.fused_moe import QuantFormat

    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkOneSidedManager,
    )
    from vllm.distributed.parallel_state import get_dp_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        flashinfer_one_sided_dispatch_layout,
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    cpu_group = get_dp_group().cpu_group
    device = torch.device(f"{DEVICE}:{rank}")

    hidden_size = 160 if padded_mxfp8 else 256
    x_width = hidden_size if padded_mxfp8 else hidden_size // 2
    scale_width = 8 if padded_mxfp8 else hidden_size // 16
    quant_config = FusedMoEQuantConfig.make("mxfp8" if padded_mxfp8 else "nvfp4")
    quant_config.mx_alignment = 128 if padded_mxfp8 else 0
    layout = flashinfer_one_sided_dispatch_layout(hidden_size, quant_config)
    assert layout.dispatch_format == (
        QuantFormat.MXFP8 if padded_mxfp8 else QuantFormat.NVFP4
    )
    tokens_per_rank = 32
    experts_per_token = 2
    num_experts = world_size * 8

    # Initialize the one-sided manager
    manager = FlashInferNVLinkOneSidedManager(cpu_group)
    manager.initialize(
        max_num_tokens=tokens_per_rank,
        top_k=experts_per_token,
        num_experts=num_experts,
        hidden_size=hidden_size,
        dispatch_format=layout.dispatch_format,
        extra_payload_bytes_per_token=layout.extra_payload_bytes_per_token,
    )
    assert manager.initialized
    communication = manager.get_communication()

    with _make_forward_context(rank, world_size, tokens_per_rank):
        dp_metadata = get_forward_context().dp_metadata

        with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
            local_sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
            runtime_max_tokens = max(local_sizes)

            # Exercise raw packed values and scales, including MXFP8 padding.
            torch.manual_seed(rank + 42)
            x = torch.randint(
                0,
                256,
                (tokens_per_rank, x_width),
                device=device,
                dtype=torch.uint8,
            )
            x_sf = torch.randint(
                0,
                256,
                (tokens_per_rank, scale_width),
                device=device,
                dtype=torch.uint8,
            )
            topk_ids = torch.randint(
                0,
                num_experts,
                (tokens_per_rank, experts_per_token),
                device=device,
                dtype=torch.int32,
            )
            topk_weights = torch.rand(
                tokens_per_rank,
                experts_per_token,
                device=device,
                dtype=torch.float32,
            )

            expected = (
                x.float().mean(-1, keepdim=True) + x_sf.float().mean(-1, keepdim=True)
            ) * (topk_weights * (topk_ids + 1)).sum(-1, keepdim=True)
            expected = expected.expand(-1, hidden_size)
            experts_per_rank = num_experts // world_size
            for scale in (1.0, 3.0):
                # Receive slots may differ each round; always compute from
                # this dispatch's payloads before combining.
                received = communication.dispatch(
                    x,
                    topk_ids,
                    topk_weights,
                    hidden_states_scale=x_sf,
                    max_tokens_per_rank=runtime_max_tokens,
                )
                recv_ids = received.topk_ids
                local = (recv_ids >= rank * experts_per_rank) & (
                    recv_ids < (rank + 1) * experts_per_rank
                )
                assert received.topk_weights is not None
                assert received.hidden_states_scale is not None
                factor = torch.where(
                    local, received.topk_weights * (recv_ids + 1), 0
                ).sum(-1, keepdim=True)
                # Consume both packed activation and scale bytes so corrupted
                # payloads cannot pass by only having the correct shape.
                value = received.hidden_states.float().mean(-1, keepdim=True)
                value += received.hidden_states_scale.float().mean(-1, keepdim=True)
                expert_output = torch.where(
                    local.any(-1, keepdim=True), value * factor * scale, 0
                )
                expert_output = (
                    expert_output.expand(-1, hidden_size)
                    .to(torch.bfloat16)
                    .contiguous()
                )
                combined = communication.combine(expert_output)
                torch.testing.assert_close(
                    combined.float(), expected * scale, rtol=2e-2, atol=5e-2
                )

            torch.distributed.barrier()

    manager.cleanup()


@requires_multi_gpu
@requires_one_sided
@requires_ptrace
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("padded_mxfp8", [False, True])
def test_one_sided_dispatch_combine(world_size, padded_mxfp8):
    """Test FlashInfer one-sided dispatch/combine with actual data flow."""
    _spawn_workers(
        partial(_one_sided_data_worker, padded_mxfp8=padded_mxfp8),
        world_size,
        dp_size=world_size,
    )


def _one_sided_prepare_finalize_worker(rank, world_size, *, empty_rank, cuda_graph):
    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkOneSidedManager,
    )
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided import (  # noqa: E501
        FlashInferNVLinkOneSidedPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )

    ep_group = get_ep_group()
    manager = FlashInferNVLinkOneSidedManager(ep_group.cpu_group)
    device_communicator = ep_group.device_communicator
    assert device_communicator is not None
    previous_manager = device_communicator.all2all_manager
    device_communicator.all2all_manager = manager
    hidden = 256
    local_experts = 8
    token_counts = [0 if empty_rank else 3, 7]
    num_tokens = token_counts[rank]
    adapter = FlashInferNVLinkOneSidedPrepareAndFinalize(
        max_num_tokens=16,
        top_k=2,
        num_experts=local_experts * world_size,
        hidden_size=hidden,
    )
    quant_config = FusedMoEQuantConfig.make()
    reduce_impl = TopKWeightAndReduceNoOP()
    torch.manual_seed(42 + rank)
    x = torch.randn(num_tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    token = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32)
    # Alternate between two experts on one rank and experts on different ranks.
    ids = torch.stack((token % local_experts, (token + 1) % local_experts), dim=1)
    ids[:, 1] += (token % 2) * local_experts
    weights = torch.empty(num_tokens, 2, device=DEVICE, dtype=torch.float32)
    weights[:, 0] = 0.25
    weights[:, 1] = 0.75
    output = torch.empty_like(x)

    def round_trip():
        received, scales, _, recv_ids, recv_weights = adapter.prepare(
            x, weights, ids, local_experts * world_size, None, False, quant_config
        )
        assert scales is None
        assert recv_ids is not None and recv_weights is not None
        local = (recv_ids >= rank * local_experts) & (
            recv_ids < (rank + 1) * local_experts
        )
        factor = torch.where(local, recv_weights * (recv_ids + 1), 0).sum(
            dim=-1, keepdim=True
        )
        expert_output = torch.where(
            local.any(dim=-1, keepdim=True), received.float() * factor, 0
        ).to(torch.bfloat16)
        adapter.finalize(output, expert_output, weights, ids, False, reduce_impl)

    def check_output():
        expected = x.float() * (weights * (ids + 1)).sum(dim=-1, keepdim=True)
        torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=5e-2)

    try:
        with _make_forward_context(
            rank, world_size, num_tokens, num_tokens_across_dp=token_counts
        ):
            metadata = get_forward_context().dp_metadata
            assert metadata is not None
            with metadata.sp_local_sizes(sequence_parallel_size=1):
                for _ in range(3):
                    round_trip()
                check_output()
                graph = None
                if cuda_graph:
                    torch.accelerator.synchronize()
                    torch.distributed.barrier(group=ep_group.cpu_group)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        round_trip()
                for _ in range(3):
                    x.uniform_(-1, 1)
                    weights.copy_(weights.flip(-1))
                    if graph is None:
                        round_trip()
                    else:
                        graph.replay()
                    check_output()
                manager.checkpoint_prepare()
                manager.checkpoint_restore()
                if graph is None:
                    round_trip()
                else:
                    graph.replay()
                check_output()
    finally:
        torch.accelerator.synchronize()
        device_communicator.all2all_manager = previous_manager
        manager.cleanup()


@requires_multi_gpu
@requires_one_sided
@requires_ptrace
@pytest.mark.parametrize("empty_rank", [False, True])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_one_sided_prepare_finalize(empty_rank, cuda_graph):
    """Check the vLLM adapter's weighted routing, padding and output on two GPUs."""
    _spawn_workers(
        partial(
            _one_sided_prepare_finalize_worker,
            empty_rank=empty_rank,
            cuda_graph=cuda_graph,
        ),
        2,
        dp_size=2,
    )


def _one_sided_low_precision_worker(rank, world_size):
    from flashinfer.fused_moe import QuantFormat

    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkOneSidedManager,
    )
    from vllm.distributed.parallel_state import get_dp_group

    device = torch.device(f"cuda:{rank}")
    hidden_size = 256
    tokens_per_rank = 32
    experts_per_token = 2
    num_experts = world_size * 8

    manager = FlashInferNVLinkOneSidedManager(get_dp_group().cpu_group)
    manager.initialize(
        max_num_tokens=tokens_per_rank,
        top_k=experts_per_token,
        num_experts=num_experts,
        hidden_size=hidden_size,
        dispatch_format=QuantFormat.NVFP4,
    )
    communication = manager.get_communication()
    assert communication.config.use_low_precision_combine

    torch.manual_seed(rank + 42)
    x = torch.randint(
        0, 256, (tokens_per_rank, hidden_size // 2), device=device, dtype=torch.uint8
    )
    x_sf = torch.randint(
        0, 256, (tokens_per_rank, hidden_size // 16), device=device, dtype=torch.uint8
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (tokens_per_rank, experts_per_token),
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.rand(
        tokens_per_rank, experts_per_token, device=device, dtype=torch.float32
    )

    communication.dispatch(
        x,
        topk_ids,
        topk_weights,
        hidden_states_scale=x_sf,
        max_tokens_per_rank=tokens_per_rank,
    )

    expert_output = torch.ones(
        world_size * tokens_per_rank, hidden_size, device=device, dtype=torch.bfloat16
    )
    output = torch.empty(
        tokens_per_rank, hidden_size, device=device, dtype=torch.bfloat16
    )
    communication.combine(expert_output, output=output)

    # Integers 1..top_k are exact in fp8_e4m3, so the fp8 transport must match
    # bf16 exactly: one contribution per distinct expert-owning rank.
    experts_per_rank = num_experts // world_size
    expert_ranks = topk_ids // experts_per_rank
    num_distinct = torch.tensor(
        [len(set(row.tolist())) for row in expert_ranks],
        device=device,
        dtype=torch.bfloat16,
    ).unsqueeze(1)
    torch.testing.assert_close(output, num_distinct.expand_as(output))

    torch.distributed.barrier()
    manager.cleanup()


@requires_multi_gpu
@requires_one_sided
@requires_ptrace
@pytest.mark.parametrize("world_size", [2])
def test_one_sided_combine_low_precision(world_size, monkeypatch):
    """Exercise the real fp8 combine transport, not a fake kernel.

    A workspace/dtype mismatch corrupts the combine buffer silently rather
    than raising, so this asserts the combined values are exactly right.
    """
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE", "1")
    _spawn_workers(_one_sided_low_precision_worker, world_size, dp_size=world_size)


# ---------------------------------------------------------------------------
# Test 6: DeepEP v2 (ElasticBuffer) manager lifecycle
# ---------------------------------------------------------------------------
#
# Tests DeepEPV2All2AllManager which wraps DeepEP's ElasticBuffer API using
# the NCCL GIN backend. Requires DeepEP >= 2.0 and NCCL >= 2.30.4.
#
# Uses EP group because the DeepEP v2 manager is constructed with an
# EP-scoped communicator in production. With tp=world_size the EP group
# spans all ranks.
# ---------------------------------------------------------------------------


def _deepep_v2_lifecycle_worker(rank, world_size):
    from vllm.distributed.device_communicators.all2all import (
        DeepEPV2All2AllManager,
    )

    ep_group = get_ep_group()
    manager = DeepEPV2All2AllManager(
        ep_group.cpu_group,
        device_group=ep_group.device_group,
    )

    assert manager.rank == rank
    assert manager.world_size == world_size
    assert manager._num_sms is None

    hidden_size = 7168
    num_experts = world_size * 32
    num_topk = 8
    max_tokens = 256

    handle_kwargs = dict(
        num_max_tokens_per_rank=max_tokens,
        hidden=hidden_size,
        num_topk=num_topk,
        num_experts=num_experts,
        use_fp8_dispatch=False,
    )

    handle = manager.get_handle(handle_kwargs)
    assert handle is not None
    assert manager._num_sms is not None
    assert manager._num_sms > 0

    torch.distributed.barrier()

    # get_handle again with same args should return cached handle
    handle2 = manager.get_handle(dict(handle_kwargs))
    assert handle2 is handle

    torch.distributed.barrier()

    # Destroy clears the cache
    manager.destroy()
    assert len(manager.handle_cache._cache) == 0

    torch.distributed.barrier()

    # Re-create after destroy
    handle3 = manager.get_handle(dict(handle_kwargs))
    assert handle3 is not None

    torch.distributed.barrier()
    manager.destroy()


@requires_multi_gpu
@requires_deep_ep_v2
@pytest.mark.parametrize("world_size", [2])
def test_deepep_v2_manager_lifecycle(world_size):
    """Test DeepEP v2 ElasticBuffer manager init, caching, and destroy."""
    _spawn_workers(_deepep_v2_lifecycle_worker, world_size)
