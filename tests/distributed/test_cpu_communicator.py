# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU communicator tests for DP SHM and EP dispatch paths."""

import os
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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


def test_cpu_shm_group_name_eligibility():
    from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator

    assert CpuCommunicator._is_cpushm_group_name("tp:0")
    assert CpuCommunicator._is_cpushm_group_name("pp:0")
    assert CpuCommunicator._is_cpushm_group_name("dp:0")
    assert CpuCommunicator._is_cpushm_group_name("ep:0")
    assert not CpuCommunicator._is_cpushm_group_name("eplb:0")
    assert not CpuCommunicator._is_cpushm_group_name("anonymous:0")


def _ensure_spawn_start_method():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")


def _report_worker_failure(rank: int, err_q: mp.Queue, err: Exception) -> None:
    err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
    raise SystemExit(1) from err


def _collect_worker_failures(procs, err_q: mp.Queue) -> list[str]:
    exit_errors = []
    for rank, proc in enumerate(procs):
        proc.join()
        if proc.exitcode != 0:
            exit_errors.append(f"[Rank {rank}] worker exited with code {proc.exitcode}")

    errors = []
    while not err_q.empty():
        errors.append(err_q.get_nowait())
    err_q.close()
    err_q.join_thread()
    return errors + exit_errors


def _run_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    worker_fn,
    params,
    err_q,
):
    try:
        worker_fn(rank, world_size, tp_size, dp_size, port, dp_port, params, err_q)
    except Exception as err:
        err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
        raise SystemExit(1) from err


def _spawn_workers(
    worker_fn,
    world_size,
    tp_size,
    dp_size,
    params,
    *,
    distributed_init_ports=None,
    dp_port=None,
):
    _ensure_spawn_start_method()

    if distributed_init_ports is None:
        shared_init_port = get_open_port()
        distributed_init_ports = [shared_init_port] * world_size
    elif len(distributed_init_ports) != world_size:
        raise ValueError("distributed_init_ports must provide one port per worker rank")

    if dp_port is None:
        dp_port = get_open_port()

    err_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(world_size):
        proc = mp.Process(
            target=_run_worker,
            args=(
                rank,
                world_size,
                tp_size,
                dp_size,
                distributed_init_ports[rank],
                dp_port,
                worker_fn,
                params,
                err_q,
            ),
        )
        proc.start()
        procs.append(proc)

    failures = _collect_worker_failures(procs, err_q)
    if failures:
        pytest.fail("Worker(s) failed:\n" + "\n---\n".join(failures))


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
        extra = _filled(local_rows, 1, float((rank + 1) * 1000))

        expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(sizes)
        expected_extra = _concat_rank_values(sizes, 1, 1000.0)

        with _make_forward_context(rank, dp_size, local_rows, sizes):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                gathered_hidden, gathered_router, gathered_extras = (
                    get_ep_group().dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                        extra_tensors=[extra.clone()],
                    )
                )
                torch.testing.assert_close(gathered_hidden, expected_hidden)
                torch.testing.assert_close(gathered_router, expected_router)
                assert len(gathered_extras) == 1
                torch.testing.assert_close(gathered_extras[0], expected_extra)

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
        _report_worker_failure(rank, err_q, err)


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

        from vllm.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
            _CPUSHMDistributed,
        )
        from vllm.distributed.parallel_state import get_ep_group, get_tp_group
        from vllm.forward_context import get_forward_context

        expected_local_sizes = _sp_local_sizes(dp_token_counts, tp_size)
        dp_rank = rank // tp_size
        expected_tp_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))

        assert get_tp_group().ranks == expected_tp_ranks
        ep_group = get_ep_group()
        assert ep_group.ranks == list(range(world_size))

        ep_communicator = ep_group.device_communicator
        assert isinstance(ep_communicator, CpuCommunicator)
        if HAS_CPU_SHM:
            assert isinstance(ep_communicator.dist_module, _CPUSHMDistributed)

        local_rows = expected_local_sizes[rank]

        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
        weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
        ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)
        extra = _filled(local_rows, 1, float((rank + 1) * 1000))

        expected_hidden = _concat_rank_values(expected_local_sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(expected_local_sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(expected_local_sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(expected_local_sizes)
        expected_extra = _concat_rank_values(expected_local_sizes, 1, 1000.0)

        with _make_forward_context(
            dp_rank,
            dp_size,
            dp_token_counts[dp_rank],
            dp_token_counts,
        ):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            with dp_metadata.sp_local_sizes(sequence_parallel_size=tp_size) as sizes:
                assert sizes == expected_local_sizes
                gathered_hidden, gathered_router, gathered_extras = (
                    ep_group.dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                        is_sequence_parallel=True,
                        extra_tensors=[extra.clone()],
                    )
                )
                torch.testing.assert_close(gathered_hidden, expected_hidden)
                torch.testing.assert_close(gathered_router, expected_router)
                assert len(gathered_extras) == 1
                torch.testing.assert_close(gathered_extras[0], expected_extra)

                gathered_hidden2, gathered_weights, gathered_ids = ep_group.dispatch(
                    hidden.clone(),
                    weights.clone(),
                    ids.clone(),
                    is_sequence_parallel=True,
                )
                torch.testing.assert_close(gathered_hidden2, expected_hidden)
                torch.testing.assert_close(gathered_weights, expected_weights)
                torch.testing.assert_close(gathered_ids, expected_ids)

                total_rows = sum(expected_local_sizes)
                expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                    1
                ).expand(total_rows, HIDDEN_SIZE).contiguous() + float(
                    (rank + 1) * 1000
                )
                combined = ep_group.combine(
                    expert_out,
                    is_sequence_parallel=True,
                )
                torch.testing.assert_close(
                    combined,
                    _expected_combined(rank, expected_local_sizes, HIDDEN_SIZE),
                )

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


@torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
def _run_compiled_fastpath(rank, sizes):
    from vllm.distributed.parallel_state import get_ep_group

    device_communicator = get_ep_group().device_communicator
    assert device_communicator is not None
    dispatch_op = device_communicator._ep_dispatch_rl_op
    combine_op = device_communicator._ep_combine_op

    def fastpath_step(hidden_states, router_logits):
        gathered_hidden, _ = dispatch_op(hidden_states, router_logits)
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
        _report_worker_failure(rank, err_q, err)


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


def _get_tp_shm_communicator():
    from vllm.distributed.device_communicators.cpu_communicator import (
        CpuCommunicator,
        _CPUSHMDistributed,
    )
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    communicator = tp_group.device_communicator
    assert isinstance(communicator, CpuCommunicator)
    assert communicator.unique_name.startswith("tp:")
    assert isinstance(communicator.dist_module, _CPUSHMDistributed)
    return tp_group, communicator


def _get_ragged_buffer_capacity(communicator, tensor, buffers, *, per_rank=False):
    buffer = buffers.get(communicator._ragged_buffer_key(tensor))
    if buffer is None:
        return 0
    if per_rank:
        return buffer.shape[0] // communicator.world_size
    return buffer.shape[0]


def _get_ragged_capacities(communicator, tensor):
    return (
        _get_ragged_buffer_capacity(
            communicator,
            tensor,
            communicator._ragged_pad_buffers,
        ),
        _get_ragged_buffer_capacity(
            communicator,
            tensor,
            communicator._ragged_shm_gather_buffers,
            per_rank=True,
        ),
    )


def _ragged_shm_buffers_worker(
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
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_ragged_buffers_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        small_sizes, large_sizes = size_patterns
        dp_group, communicator = _get_dp_shm_communicator()

        small_hidden = _filled(small_sizes[rank], HIDDEN_SIZE, float(rank + 1))
        small_result = dp_group.all_gatherv(small_hidden.clone(), sizes=small_sizes)
        torch.testing.assert_close(
            small_result,
            _concat_rank_values(small_sizes, HIDDEN_SIZE, 1.0),
        )
        small_caps = _get_ragged_capacities(communicator, small_hidden)
        small_result_repeat = dp_group.all_gatherv(
            small_hidden.clone(),
            sizes=small_sizes,
        )
        torch.testing.assert_close(
            small_result_repeat,
            _concat_rank_values(small_sizes, HIDDEN_SIZE, 1.0),
        )
        small_repeat_caps = _get_ragged_capacities(communicator, small_hidden)

        large_hidden = _filled(large_sizes[rank], HIDDEN_SIZE, float(rank + 1))
        large_result = dp_group.all_gatherv(large_hidden.clone(), sizes=large_sizes)
        torch.testing.assert_close(
            large_result,
            _concat_rank_values(large_sizes, HIDDEN_SIZE, 1.0),
        )
        large_caps = _get_ragged_capacities(communicator, large_hidden)

        large_result_repeat = dp_group.all_gatherv(
            large_hidden.clone(),
            sizes=large_sizes,
        )
        torch.testing.assert_close(
            large_result_repeat,
            _concat_rank_values(large_sizes, HIDDEN_SIZE, 1.0),
        )
        repeat_caps = _get_ragged_capacities(communicator, large_hidden)

        assert small_caps == (max(small_sizes), max(small_sizes))
        assert small_repeat_caps == small_caps
        assert large_caps == (max(large_sizes), max(large_sizes))
        assert large_caps[0] > small_caps[0]
        assert large_caps[1] > small_caps[1]
        assert repeat_caps == large_caps

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


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
        assert _get_ragged_capacities(communicator, hidden) == (0, 0)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


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
        _report_worker_failure(rank, err_q, err)


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
        _report_worker_failure(rank, err_q, err)


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
        _report_worker_failure(rank, err_q, err)


def _moe_parallel_config_worker(
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
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_moe_config_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.config.parallel import ParallelConfig
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            get_tp_group,
        )
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )

        dp_rank = rank // tp_size
        tp_rank = rank % tp_size
        expected_tp_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        expected_dp_ranks = list(range(tp_rank, world_size, tp_size))

        assert get_tp_group().ranks == expected_tp_ranks
        assert get_dp_group().ranks == expected_dp_ranks
        assert get_dp_group().rank_in_group == dp_rank
        assert get_ep_group().ranks == list(range(world_size))

        no_ep_parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            enable_expert_parallel=False,
        )
        no_ep_moe_config = FusedMoEParallelConfig.make(
            tp_size_=tp_size,
            pcp_size_=1,
            dp_size_=dp_size,
            sp_size_=1,
            vllm_parallel_config=no_ep_parallel_config,
        )
        assert no_ep_moe_config.tp_size == world_size
        assert no_ep_moe_config.tp_rank == rank
        assert no_ep_moe_config.dp_size == dp_size
        assert no_ep_moe_config.dp_rank == dp_rank
        assert no_ep_moe_config.ep_size == 1
        assert no_ep_moe_config.ep_rank == 0
        assert not no_ep_moe_config.use_ep

        ep_parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            enable_expert_parallel=True,
        )
        ep_moe_config = FusedMoEParallelConfig.make(
            tp_size_=tp_size,
            pcp_size_=1,
            dp_size_=dp_size,
            sp_size_=1,
            vllm_parallel_config=ep_parallel_config,
        )
        assert ep_moe_config.tp_size == 1
        assert ep_moe_config.tp_rank == 0
        assert ep_moe_config.dp_size == dp_size
        assert ep_moe_config.dp_rank == dp_rank
        assert ep_moe_config.ep_size == world_size
        assert ep_moe_config.ep_rank == rank
        assert ep_moe_config.use_ep

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _tp_shm_all_reduce_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    """Confirm the TP all-reduce path uses SHM and matches gloo."""
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_tp_shm_all_reduce_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        tp_group, _ = _get_tp_shm_communicator()
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) + float(rank)

        ref = tensor.clone()
        dist.all_reduce(ref, group=tp_group.cpu_group)

        orig_all_reduce = dist.all_reduce

        def forbidden_all_reduce(*args, **kwargs):
            raise AssertionError("TP SHM all_reduce fell back to torch.distributed")

        dist.all_reduce = forbidden_all_reduce
        try:
            shm_result = tp_group.all_reduce(tensor.clone())
        finally:
            dist.all_reduce = orig_all_reduce

        torch.testing.assert_close(shm_result, ref)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _dp_shm_all_reduce_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    """Confirm the DP SHM all-reduce matches the gloo reference exactly."""
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_shm_all_reduce_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, _ = _get_dp_shm_communicator()
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) + float(rank)

        # Gloo reference (SUM over the DP group's cpu_group).
        ref = tensor.clone()
        dist.all_reduce(ref, group=dp_group.cpu_group)

        shm_result = dp_group.all_reduce(tensor.clone())

        torch.testing.assert_close(shm_result, ref)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


@pytest.mark.distributed
@pytest.mark.parametrize("sizes", [[2, 1], [0, 3]], ids=["ragged", "zero-rank"])
def test_cpu_ep_dispatch_combine_ragged(sizes):
    _spawn_workers(
        _ragged_dispatch_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=sizes,
    )


@pytest.mark.distributed
def test_cpu_ep_sequence_parallel_uses_ep_group():
    _spawn_workers(
        _sequence_parallel_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=[3, 1, 5],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
def test_cpu_ep_compile_ragged_fastpath():
    _spawn_workers(
        _compile_fastpath_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[[2, 1], [0, 3]],
    )


@pytest.mark.distributed
def test_cpu_dp_group_ranks_share_shm_group_name():
    _spawn_workers(
        _dp_shm_group_name_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
        distributed_init_ports=[get_open_port(), get_open_port()],
        dp_port=get_open_port(),
    )


@pytest.mark.distributed
def test_cpu_moe_tp2_dp3_parallel_config():
    _spawn_workers(
        _moe_parallel_config_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_all_gatherv_ragged_shm_buffers_reuse_and_grow():
    _spawn_workers(
        _ragged_shm_buffers_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[[1, 0], [3, 1]],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_all_gatherv_uniform_sizes_matches_direct_shm_gather():
    _spawn_workers(
        _uniform_sizes_shm_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[2, 2],
    )


@pytest.mark.distributed
def test_cpu_dp_all_gatherv_all_zero_rows():
    _spawn_workers(
        _all_zero_gatherv_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
    )


@pytest.mark.distributed
def test_cpu_dp_reduce_scatterv_implicit_sizes_require_even_split():
    _spawn_workers(
        _implicit_uneven_reduce_scatterv_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_shm_all_reduce_matches_gloo():
    _spawn_workers(
        _dp_shm_all_reduce_worker,
        world_size=6,
        tp_size=1,
        dp_size=6,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_tp_shm_all_reduce_matches_gloo_without_fallback():
    _spawn_workers(
        _tp_shm_all_reduce_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=None,
    )
