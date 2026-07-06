# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU EP communicator tests.

CPU AgRs dispatch/combine follows the runtime group selection: non-sequence-
parallel paths use the DP group, which may take the single-node SHM fastpath,
while sequence-parallel paths use the EP group.
"""

import os

import pytest
import torch
import torch.distributed as dist

from tests.distributed.cpu_mp_test_utils import (
    report_worker_failure,
    spawn_workers,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils.network_utils import get_open_port

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)

import vllm._custom_ops  # noqa: E402,F401  # populates torch.ops._C for HAS_CPU_SHM

HIDDEN_SIZE = 8
NUM_EXPERTS = 6
TOPK = 2
HAS_CPU_SHM = hasattr(torch.ops._C, "init_shm_manager") and (
    current_platform.get_cpu_architecture()
    in (CpuArchEnum.X86, CpuArchEnum.ARM, CpuArchEnum.POWERPC)
)


def _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port):
    """Init vLLM distributed env for TP=tp_size, DP=dp_size."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        _data_parallel_master_port_list=[int(dp_port)],
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=tp_size,
            rank=tp_rank,
            distributed_init_method=f"tcp://localhost:{port}",
            local_rank=rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(tp_size, 1, backend="gloo")


def _make_forward_context(dp_rank, dp_size, num_tokens, num_tokens_across_dp):
    """Create a forward context with explicit DP token counts."""
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig
    from vllm.forward_context import set_forward_context

    class _AttnMeta:
        dp_metadata = None

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=dp_size,
        is_moe_model=True,
        data_parallel_rank=dp_rank,
    )
    return set_forward_context(
        _AttnMeta(),
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=torch.tensor(num_tokens_across_dp, dtype=torch.int),
    )


def _filled(rows: int, cols: int, value: float, dtype=torch.float32) -> torch.Tensor:
    return torch.full((rows, cols), value, dtype=dtype)


def _concat_rank_values(sizes, cols, value_scale, dtype=torch.float32):
    chunks = [
        _filled(size, cols, value_scale * (rank + 1), dtype)
        for rank, size in enumerate(sizes)
    ]
    return torch.cat(chunks, dim=0)


def _concat_rank_ids(sizes):
    chunks = [
        torch.full((size, TOPK), rank, dtype=torch.long)
        for rank, size in enumerate(sizes)
    ]
    return torch.cat(chunks, dim=0)


def _expected_combined(rank, sizes, cols):
    total_rows = sum(sizes)
    start = sum(sizes[:rank])
    end = start + sizes[rank]
    rows = (
        torch.arange(total_rows, dtype=torch.float32)
        .unsqueeze(1)
        .expand(
            total_rows,
            cols,
        )
    )
    tag_sum = sum(float((r + 1) * 1000) for r in range(len(sizes)))
    return rows[start:end] * len(sizes) + tag_sum


def _sp_local_sizes(dp_token_counts, tp_size):
    sizes = []
    for dp_tokens in dp_token_counts:
        local_rows = (dp_tokens + tp_size - 1) // tp_size
        sizes.extend([local_rows] * tp_size)
    return sizes


def _ragged_dispatch_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    sizes,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_comm_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_ep_group
        from vllm.forward_context import get_forward_context

        local_rows = sizes[rank]
        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
        weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
        ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)

        expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(sizes)

        with _make_forward_context(rank, dp_size, local_rows, sizes):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                gathered_hidden, gathered_router = (
                    get_ep_group().dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                    )
                )
                torch.testing.assert_close(gathered_hidden, expected_hidden)
                torch.testing.assert_close(gathered_router, expected_router)

                gathered_hidden2, gathered_weights, gathered_ids = (
                    get_ep_group().dispatch(
                        hidden.clone(),
                        weights.clone(),
                        ids.clone(),
                    )
                )
                torch.testing.assert_close(gathered_hidden2, expected_hidden)
                torch.testing.assert_close(gathered_weights, expected_weights)
                torch.testing.assert_close(gathered_ids, expected_ids)

                total_rows = sum(sizes)
                expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                    1
                ).expand(total_rows, HIDDEN_SIZE).contiguous() + float(
                    (rank + 1) * 1000
                )
                combined = get_ep_group().combine(expert_out)
                torch.testing.assert_close(
                    combined,
                    _expected_combined(rank, sizes, HIDDEN_SIZE),
                )

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _ragged_dispatch_padding_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    sizes,
    err_q,
):
    """VLLM_MOE_SKIP_PADDING gather propagation: an is_padding mask set in
    this rank's forward context before dispatch_router_logits() must arrive,
    gathered and thresholded, in forward_context.is_padding on every rank."""
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_comm_pad_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_ep_group
        from vllm.forward_context import get_forward_context

        local_rows = sizes[rank]
        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
        # Rank 0's row(s) are real; every other rank is all-padding (mirrors
        # the conc=1 dummy-token pattern: only 1 of N DP ranks has a real
        # token).
        is_padding = torch.full((local_rows,), rank != 0, dtype=torch.bool)

        with _make_forward_context(rank, dp_size, local_rows, sizes):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            get_forward_context().is_padding = is_padding

            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                get_ep_group().dispatch_router_logits(
                    hidden.clone(),
                    router.clone(),
                )

            expected_mask = torch.cat(
                [
                    torch.full((size,), r != 0, dtype=torch.bool)
                    for r, size in enumerate(sizes)
                ]
            )
            gathered_mask = get_forward_context().is_padding
            torch.testing.assert_close(gathered_mask, expected_mask)

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _sequence_parallel_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    dp_token_counts,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_sp_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_ep_group
        from vllm.forward_context import get_forward_context

        local_sizes = _sp_local_sizes(dp_token_counts, tp_size)
        dp_rank = rank // tp_size
        local_rows = local_sizes[rank]

        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
        weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
        ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)

        expected_hidden = _concat_rank_values(local_sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(local_sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(local_sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(local_sizes)

        with _make_forward_context(
            dp_rank,
            dp_size,
            dp_token_counts[dp_rank],
            dp_token_counts,
        ):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            with dp_metadata.sp_local_sizes(sequence_parallel_size=tp_size):
                gathered_hidden, gathered_router = (
                    get_ep_group().dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                        is_sequence_parallel=True,
                    )
                )
                torch.testing.assert_close(gathered_hidden, expected_hidden)
                torch.testing.assert_close(gathered_router, expected_router)

                gathered_hidden2, gathered_weights, gathered_ids = (
                    get_ep_group().dispatch(
                        hidden.clone(),
                        weights.clone(),
                        ids.clone(),
                        is_sequence_parallel=True,
                    )
                )
                torch.testing.assert_close(gathered_hidden2, expected_hidden)
                torch.testing.assert_close(gathered_weights, expected_weights)
                torch.testing.assert_close(gathered_ids, expected_ids)

                total_rows = sum(local_sizes)
                expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                    1
                ).expand(total_rows, HIDDEN_SIZE).contiguous() + float(
                    (rank + 1) * 1000
                )
                combined = get_ep_group().combine(
                    expert_out,
                    is_sequence_parallel=True,
                )
                torch.testing.assert_close(
                    combined,
                    _expected_combined(rank, local_sizes, HIDDEN_SIZE),
                )

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


@torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
def _run_compiled_fastpath(rank, sizes):
    from vllm.distributed.parallel_state import get_ep_group

    device_communicator = get_ep_group().device_communicator
    assert device_communicator is not None
    dispatch_op = device_communicator._ep_dispatch_rl_op
    combine_op = device_communicator._ep_combine_op

    def fastpath_step(hidden_states, router_logits):
        gathered_hidden, _, _ = dispatch_op(hidden_states, router_logits)
        return combine_op(gathered_hidden) + hidden_states

    compiled = torch.compile(fastpath_step, fullgraph=True, backend="eager")
    local_rows = sizes[rank]
    hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
    router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
    output = compiled(hidden, router)
    torch.testing.assert_close(output, hidden * (len(sizes) + 1))


def _compile_fastpath_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    size_patterns,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_compile_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
        torch._dynamo.reset()

        for sizes in size_patterns:
            with _make_forward_context(rank, dp_size, sizes[rank], sizes):
                _run_compiled_fastpath(rank, sizes)

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _get_dp_shm_communicator():
    from vllm.distributed.device_communicators.cpu_communicator import (
        CpuCommunicator,
        _CPUSHMDistributed,
    )
    from vllm.distributed.parallel_state import get_dp_group

    dp_group = get_dp_group()
    communicator = dp_group.device_communicator
    assert isinstance(communicator, CpuCommunicator)
    assert isinstance(communicator.dist_module, _CPUSHMDistributed)
    return dp_group, communicator


def _ragged_shm_reuse_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    sizes,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_ragged_reuse_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, communicator = _get_dp_shm_communicator()
        hidden = _filled(sizes[rank], HIDDEN_SIZE, float(rank + 1))
        expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)

        first = dp_group.all_gatherv(hidden.clone(), sizes=sizes)
        torch.testing.assert_close(first, expected_hidden)
        first_caps = (
            communicator._get_ragged_pad_capacity(hidden),
            communicator._get_ragged_shm_gather_capacity(hidden),
        )

        second = dp_group.all_gatherv(hidden.clone(), sizes=sizes)
        torch.testing.assert_close(second, expected_hidden)
        second_caps = (
            communicator._get_ragged_pad_capacity(hidden),
            communicator._get_ragged_shm_gather_capacity(hidden),
        )

        assert first_caps == (max(sizes), max(sizes))
        assert second_caps == first_caps

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _ragged_shm_grow_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    size_patterns,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_ragged_grow_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        small_sizes, large_sizes = size_patterns
        dp_group, communicator = _get_dp_shm_communicator()

        small_hidden = _filled(small_sizes[rank], HIDDEN_SIZE, float(rank + 1))
        small_result = dp_group.all_gatherv(small_hidden.clone(), sizes=small_sizes)
        torch.testing.assert_close(
            small_result,
            _concat_rank_values(small_sizes, HIDDEN_SIZE, 1.0),
        )
        small_caps = (
            communicator._get_ragged_pad_capacity(small_hidden),
            communicator._get_ragged_shm_gather_capacity(small_hidden),
        )

        large_hidden = _filled(large_sizes[rank], HIDDEN_SIZE, float(rank + 1))
        large_result = dp_group.all_gatherv(large_hidden.clone(), sizes=large_sizes)
        torch.testing.assert_close(
            large_result,
            _concat_rank_values(large_sizes, HIDDEN_SIZE, 1.0),
        )
        large_caps = (
            communicator._get_ragged_pad_capacity(large_hidden),
            communicator._get_ragged_shm_gather_capacity(large_hidden),
        )

        large_result_repeat = dp_group.all_gatherv(
            large_hidden.clone(),
            sizes=large_sizes,
        )
        torch.testing.assert_close(
            large_result_repeat,
            _concat_rank_values(large_sizes, HIDDEN_SIZE, 1.0),
        )
        repeat_caps = (
            communicator._get_ragged_pad_capacity(large_hidden),
            communicator._get_ragged_shm_gather_capacity(large_hidden),
        )

        assert small_caps == (max(small_sizes), max(small_sizes))
        assert large_caps == (max(large_sizes), max(large_sizes))
        assert large_caps[0] > small_caps[0]
        assert large_caps[1] > small_caps[1]
        assert repeat_caps == large_caps

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _uniform_sizes_shm_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    sizes,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_uniform_sizes_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, communicator = _get_dp_shm_communicator()
        hidden = _filled(sizes[rank], HIDDEN_SIZE, float(rank + 1))
        expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)

        explicit_sizes = dp_group.all_gatherv(hidden.clone(), sizes=sizes)
        implicit_sizes = dp_group.all_gatherv(hidden.clone(), sizes=None)

        torch.testing.assert_close(explicit_sizes, expected_hidden)
        torch.testing.assert_close(implicit_sizes, expected_hidden)
        torch.testing.assert_close(explicit_sizes, implicit_sizes)
        assert communicator._get_ragged_pad_capacity(hidden) == 0
        assert communicator._get_ragged_shm_gather_capacity(hidden) == 0

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _all_zero_gatherv_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_zero_gather_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_dp_group

        sizes = [0] * dp_size
        hidden = torch.empty((0, HIDDEN_SIZE), dtype=torch.float32)
        gathered = get_dp_group().all_gatherv(hidden.clone(), sizes=sizes)

        assert gathered.shape == (0, HIDDEN_SIZE)
        assert gathered.numel() == 0

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _implicit_uneven_reduce_scatterv_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        os.environ.setdefault(
            "VLLM_DIST_IDENT", f"test_cpu_dp_uneven_reduce_scatterv_{port}"
        )
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_dp_group

        hidden = _filled(3, HIDDEN_SIZE, float(rank + 1))
        with pytest.raises(
            AssertionError,
            match="Implicit reduce_scatterv requires the scatter dimension",
        ):
            get_dp_group().reduce_scatterv(hidden, dim=0, sizes=None)

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _dp_shm_group_name_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.parallel import ParallelConfig
        from vllm.distributed.parallel_state import (
            ensure_model_parallel_initialized,
            get_dp_group,
            init_distributed_environment,
        )
        from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident

        dp_rank = rank // tp_size
        tp_rank = rank % tp_size
        distributed_init_method = f"tcp://127.0.0.1:{port}"

        vllm_config = VllmConfig()
        vllm_config.parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            data_parallel_master_ip="127.0.0.1",
            _data_parallel_master_port_list=[int(dp_port)],
        )
        os.environ["VLLM_DIST_IDENT"] = _get_cpushm_dist_ident(
            vllm_config.parallel_config,
            distributed_init_method,
        )

        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=tp_size,
                rank=tp_rank,
                distributed_init_method=distributed_init_method,
                local_rank=rank,
                backend="gloo",
            )
            ensure_model_parallel_initialized(tp_size, 1, backend="gloo")

            dp_group = get_dp_group()
            communicator = dp_group.device_communicator
            assert communicator is not None
            assert communicator._all_group_ranks_share_shm_group_name()

            dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


def _dp_ar_shm_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    """Mirror `_run_ar`'s 4xdp_size int32 coordination all-reduce and confirm
    the SHM fp32 round-trip matches the gloo reference exactly."""
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_ar_shm_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, _ = _get_dp_shm_communicator()

        # Build the same 4 x dp_size contribution tensor `_run_ar` builds,
        # writing only this rank's column with a distinct per-rank pattern.
        tensor = torch.zeros(4, dp_size, dtype=torch.int32)
        tensor[0][rank] = 10 + rank  # orig_num_tokens
        tensor[1][rank] = 100 + rank  # padded_num_tokens
        tensor[2][rank] = 1 if rank % 2 == 0 else 0  # should_ubatch flag
        tensor[3][rank] = rank  # cudagraph_mode

        # Gloo reference (SUM over the DP group's cpu_group).
        ref = tensor.clone()
        dist.all_reduce(ref, group=dp_group.cpu_group)

        # SHM fp32 round-trip (matches the shipped `_run_ar` path).
        tensor_fp32 = tensor.to(torch.float32)
        dp_group.all_reduce(tensor_fp32)
        shm_result = tensor_fp32.round().to(torch.int32)

        torch.testing.assert_close(shm_result, ref)

        # Spot-check the semantic reads `_run_ar`'s callers rely on.
        assert bool(torch.all(shm_result[2] == 1).item()) == bool(
            torch.all(ref[2] == 1).item()
        )
        assert int(shm_result[3].min().item()) == int(ref[3].min().item())
        assert int(shm_result[1].max().item()) == int(ref[1].max().item())

        dist.barrier()
    except Exception as err:
        report_worker_failure(rank, err_q, err)


@pytest.mark.distributed
@pytest.mark.parametrize("sizes", [[2, 1], [0, 3]], ids=["ragged", "zero-rank"])
def test_cpu_ep_dispatch_combine_ragged(sizes):
    spawn_workers(
        _ragged_dispatch_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=sizes,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("sizes", [[1, 2], [1, 0, 3]], ids=["dp2", "dp3-zero-rank"])
def test_cpu_ep_dispatch_propagates_padding_mask(sizes):
    spawn_workers(
        _ragged_dispatch_padding_worker,
        world_size=len(sizes),
        tp_size=1,
        dp_size=len(sizes),
        params=sizes,
    )


@pytest.mark.distributed
def test_cpu_ep_sequence_parallel_uses_ep_group():
    spawn_workers(
        _sequence_parallel_worker,
        world_size=4,
        tp_size=2,
        dp_size=2,
        params=[3, 1],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
def test_cpu_ep_compile_ragged_fastpath():
    spawn_workers(
        _compile_fastpath_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[[2, 1], [0, 3]],
    )


@pytest.mark.distributed
def test_cpu_dp_group_ranks_share_shm_group_name():
    spawn_workers(
        _dp_shm_group_name_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
        distributed_init_ports=[get_open_port(), get_open_port()],
        dp_port=get_open_port(),
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_all_gatherv_ragged_reuses_shm_buffers():
    spawn_workers(
        _ragged_shm_reuse_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[2, 1],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_all_gatherv_ragged_shm_buffers_grow_on_demand():
    spawn_workers(
        _ragged_shm_grow_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[[1, 0], [3, 1]],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_all_gatherv_uniform_sizes_matches_direct_shm_gather():
    spawn_workers(
        _uniform_sizes_shm_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[2, 2],
    )


@pytest.mark.distributed
def test_cpu_dp_all_gatherv_all_zero_rows():
    spawn_workers(
        _all_zero_gatherv_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
    )


@pytest.mark.distributed
def test_cpu_dp_reduce_scatterv_implicit_sizes_require_even_split():
    spawn_workers(
        _implicit_uneven_reduce_scatterv_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_token_count_all_reduce_shm_matches_gloo():
    """The per-step DP token-count all-reduce (`_run_ar`) routed through SHM
    must match the gloo reference for a DP=6 (EP) topology."""
    spawn_workers(
        _dp_ar_shm_worker,
        world_size=6,
        tp_size=1,
        dp_size=6,
        params=None,
    )
