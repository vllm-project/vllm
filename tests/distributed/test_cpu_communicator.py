# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU communicator tests for Gloo DP/EP and SHM TP/PP paths."""

import os

import pytest
import torch
import torch.distributed as dist

from tests.distributed.cpu_test_utils import (
    init_cpu_distributed_environment as _init_tp_dp_environment,
)
from tests.distributed.cpu_test_utils import (
    make_cpu_forward_context as _make_forward_context,
)
from tests.distributed.cpu_test_utils import (
    spawn_cpu_workers as _spawn_workers,
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
):
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

        with dp_metadata.sp_local_sizes(
            sequence_parallel_size=1, pcp_size=1, use_ep=False
        ):
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

            hidden_none, router_none = get_ep_group().dispatch_router_logits(
                hidden.clone(), router.clone()
            )
            torch.testing.assert_close(hidden_none, expected_hidden)
            torch.testing.assert_close(router_none, expected_router)

            hidden_empty, router_empty, empty_extras = (
                get_ep_group().dispatch_router_logits(
                    hidden.clone(), router.clone(), extra_tensors=[]
                )
            )
            torch.testing.assert_close(hidden_empty, expected_hidden)
            torch.testing.assert_close(router_empty, expected_router)
            assert empty_extras == []

            gathered_hidden2, gathered_weights, gathered_ids = get_ep_group().dispatch(
                hidden.clone(),
                weights.clone(),
                ids.clone(),
            )
            torch.testing.assert_close(gathered_hidden2, expected_hidden)
            torch.testing.assert_close(gathered_weights, expected_weights)
            torch.testing.assert_close(gathered_ids, expected_ids)

            hidden_with_empty, weights_with_empty, ids_with_empty, empty_extras = (
                get_ep_group().dispatch(
                    hidden.clone(),
                    weights.clone(),
                    ids.clone(),
                    extra_tensors=[],
                )
            )
            torch.testing.assert_close(hidden_with_empty, expected_hidden)
            torch.testing.assert_close(weights_with_empty, expected_weights)
            torch.testing.assert_close(ids_with_empty, expected_ids)
            assert empty_extras == []

            total_rows = sum(sizes)
            expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                1
            ).expand(total_rows, HIDDEN_SIZE).contiguous() + float((rank + 1) * 1000)
            combined = get_ep_group().combine(expert_out)
            torch.testing.assert_close(
                combined,
                _expected_combined(rank, sizes, HIDDEN_SIZE),
            )

    dist.barrier()


def _sequence_parallel_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    dp_token_counts,
):
    os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_sp_{port}")
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

    from vllm.distributed.device_communicators.cpu_communicator import (
        CpuCommunicator,
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
    assert ep_communicator.dist_module is dist

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

        with dp_metadata.sp_local_sizes(
            sequence_parallel_size=tp_size, pcp_size=1, use_ep=True
        ) as sizes:
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
            ).expand(total_rows, HIDDEN_SIZE).contiguous() + float((rank + 1) * 1000)
            combined = ep_group.combine(
                expert_out,
                is_sequence_parallel=True,
            )
            torch.testing.assert_close(
                combined,
                _expected_combined(rank, expected_local_sizes, HIDDEN_SIZE),
            )
            assert dp_metadata.local_sizes is sizes

        assert dp_metadata.local_sizes is None

    dist.barrier()


def _compile_fastpath(dp_metadata, ep_group, backend):
    def fastpath_step(hidden_states, router_logits, topk_weights, topk_ids, extra):
        with dp_metadata.sp_local_sizes(
            sequence_parallel_size=1, pcp_size=1, use_ep=False
        ) as local_sizes:
            gathered_hidden, gathered_router, gathered_extras = (
                ep_group.dispatch_router_logits(
                    hidden_states,
                    router_logits,
                    is_sequence_parallel=False,
                    extra_tensors=[extra],
                )
            )
            dispatched_hidden, dispatched_weights, dispatched_ids, _ = (
                ep_group.dispatch(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    is_sequence_parallel=False,
                    extra_tensors=[],
                )
            )
            output = (
                ep_group.combine(gathered_hidden)
                + ep_group.combine(dispatched_hidden)
                + ep_group.combine(gathered_extras[0])
            )
            assert dp_metadata.local_sizes is local_sizes
            return (
                gathered_hidden,
                gathered_router,
                gathered_extras[0],
                dispatched_hidden,
                dispatched_weights,
                dispatched_ids,
                output,
            )

    return torch.compile(
        fastpath_step,
        fullgraph=True,
        backend=backend,
        dynamic=backend == "inductor",
    )


def _run_compiled_fastpath(rank, sizes, compiled):
    from vllm.forward_context import get_forward_context

    dp_metadata = get_forward_context().dp_metadata
    assert dp_metadata is not None

    local_rows = sizes[rank]
    hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
    router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
    weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
    ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)
    extra = _filled(local_rows, 1, float((rank + 1) * 1000))

    (
        gathered_hidden,
        gathered_router,
        gathered_extra,
        dispatched_hidden,
        dispatched_weights,
        dispatched_ids,
        output,
    ) = compiled(hidden, router, weights, ids, extra)
    expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)
    expected_router = _concat_rank_values(sizes, NUM_EXPERTS, 10.0)
    expected_weights = _concat_rank_values(sizes, TOPK, 100.0)
    expected_ids = _concat_rank_ids(sizes)
    expected_extra = _concat_rank_values(sizes, 1, 1000.0)
    torch.testing.assert_close(gathered_hidden, expected_hidden)
    torch.testing.assert_close(gathered_router, expected_router)
    torch.testing.assert_close(gathered_extra, expected_extra)
    torch.testing.assert_close(dispatched_hidden, expected_hidden)
    torch.testing.assert_close(dispatched_weights, expected_weights)
    torch.testing.assert_close(dispatched_ids, expected_ids)
    torch.testing.assert_close(
        output,
        hidden * (2 * len(sizes)) + extra * len(sizes),
    )


def _compile_fastpath_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
):
    os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_compile_{port}")
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
    torch._dynamo.reset()

    from vllm.distributed.parallel_state import get_ep_group
    from vllm.forward_context import get_forward_context

    backend, size_patterns = params
    first_sizes = size_patterns[0]
    with _make_forward_context(rank, dp_size, first_sizes[rank], first_sizes):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        assert dp_metadata.local_sizes is None
        compiled = _compile_fastpath(dp_metadata, get_ep_group(), backend)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            for sizes in size_patterns:
                dp_metadata.num_tokens_across_dp_cpu.copy_(
                    torch.tensor(sizes, dtype=torch.int)
                )
                _run_compiled_fastpath(rank, sizes, compiled)
                assert dp_metadata.local_sizes is None

    dist.barrier()


def _get_dp_gloo_communicator():
    from vllm.distributed.device_communicators.cpu_communicator import (
        CpuCommunicator,
    )
    from vllm.distributed.parallel_state import get_dp_group

    dp_group = get_dp_group()
    communicator = dp_group.device_communicator
    assert isinstance(communicator, CpuCommunicator)
    assert communicator.dist_module is dist
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


def _ragged_output_lifetime_worker(
    rank, world_size, tp_size, dp_size, port, dp_port, size_patterns
):
    os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_ragged_buffers_{port}")
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
    dp_group, _ = _get_dp_gloo_communicator()
    small_sizes, large_sizes = size_patterns
    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for sizes, scale in [(small_sizes, 1.0), (large_sizes, 2.0), (small_sizes, -1.0)]:
        hidden = _filled(sizes[rank], HIDDEN_SIZE, scale * (rank + 1))
        original = hidden.clone()
        result = dp_group.all_gatherv(hidden, sizes=sizes)
        expected = _concat_rank_values(sizes, HIDDEN_SIZE, scale)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(hidden, original)
        # Repeated gathers must not overwrite already-returned outputs.
        for old_result, old_expected in results:
            torch.testing.assert_close(old_result, old_expected)
        results.append((result, expected))
        assert (
            result.untyped_storage().nbytes() == result.numel() * result.element_size()
        )
    dist.barrier()


def _default_collectives_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
):
    os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_collective_defaults_{port}")
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

    from vllm.distributed.parallel_state import get_dp_group

    dp_group = get_dp_group()
    uniform = _filled(2, HIDDEN_SIZE, float(rank + 1))
    original = uniform.clone()
    for sizes in ([2, 2], None):
        torch.testing.assert_close(
            dp_group.all_gatherv(uniform, sizes=sizes),
            _concat_rank_values([2, 2], HIDDEN_SIZE, 1.0),
        )
    torch.testing.assert_close(uniform, original)
    _get_dp_gloo_communicator()

    empty = torch.empty((0, HIDDEN_SIZE), dtype=torch.float32)
    for sizes in ([0, 0], None):
        assert dp_group.all_gatherv(empty, sizes=sizes).shape == empty.shape

    with pytest.raises(
        ValueError,
        match="Implicit reduce_scatterv requires the scatter dimension",
    ):
        dp_group.reduce_scatterv(_filled(2, 3, float(rank + 1)), sizes=None)

    scatter_input = torch.arange(18, dtype=torch.float32).view(3, 6) + rank * 100
    scatter_original = scatter_input.clone()
    sizes = [2, 4]
    reduced = scatter_input.clone()
    dist.all_reduce(reduced, group=dp_group.cpu_group)
    start = sum(sizes[:rank])
    expected = reduced.narrow(-1, start, sizes[rank])

    scattered_default = dp_group.reduce_scatterv(scatter_input, sizes=sizes)
    scattered_positive = dp_group.reduce_scatterv(
        scatter_input.clone(), dim=1, sizes=sizes
    )
    torch.testing.assert_close(scattered_default, expected)
    torch.testing.assert_close(scattered_positive, expected)
    torch.testing.assert_close(scatter_input, scatter_original)

    empty_input = torch.empty((3, 0), dtype=torch.float32)
    empty_output = dp_group.reduce_scatterv(empty_input, sizes=[0, 0])
    assert empty_output.shape == (3, 0)

    dist.barrier()


def _dp_gloo_ragged_gather_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    sizes,
):
    os.environ["VLLM_DIST_IDENT"] = f"test_cpu_dp_gloo_fallback_{rank}"
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

    from vllm.distributed.parallel_state import get_dp_group

    dp_group = get_dp_group()
    communicator = dp_group.device_communicator
    assert communicator is not None
    assert communicator.dist_module is dist

    hidden = _filled(sizes[rank], HIDDEN_SIZE, float(rank + 1))
    original = hidden.clone()
    gathered = dp_group.all_gatherv(hidden, sizes=sizes)
    torch.testing.assert_close(
        gathered,
        _concat_rank_values(sizes, HIDDEN_SIZE, 1.0),
    )
    torch.testing.assert_close(hidden, original)

    dist.barrier()


def _shm_all_reduce_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
):
    """Confirm TP all-reduce uses SHM and matches Gloo."""
    os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_shm_all_reduce_{port}")
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

    group, _ = _get_tp_shm_communicator()
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) + float(rank)

    ref = tensor.clone()
    dist.all_reduce(ref, group=group.cpu_group)

    orig_all_reduce = dist.all_reduce

    def forbidden_all_reduce(*args, **kwargs):
        raise AssertionError("SHM all_reduce fell back to torch.distributed")

    dist.all_reduce = forbidden_all_reduce
    try:
        shm_result = group.all_reduce(tensor.clone())
    finally:
        dist.all_reduce = orig_all_reduce

    torch.testing.assert_close(shm_result, ref)

    dist.barrier()


def _transport_worker(rank, world_size, tp_size, dp_size, port, dp_port, params):
    os.environ["VLLM_DIST_IDENT"] = f"test_cpu_transport_{port}"
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

    from vllm.distributed.device_communicators.cpu_communicator import (
        _CPUSHMDistributed,
    )
    from vllm.distributed.parallel_state import (
        GroupCoordinator,
        get_dp_group,
        get_ep_group,
        get_tp_group,
    )

    for group in (get_dp_group(), get_ep_group()):
        assert group.device_communicator.dist_module is dist
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) + rank
        expected = tensor.clone()
        dist.all_reduce(expected, group=group.cpu_group)
        torch.testing.assert_close(group.all_reduce(tensor), expected, rtol=0, atol=0)

    if tp_size == 1:
        return

    pp_group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_device_communicator=True,
        group_name="pp",
    )
    for group in (get_tp_group(), pp_group):
        if HAS_CPU_SHM:
            assert isinstance(group.device_communicator.dist_module, _CPUSHMDistributed)
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) + rank
        expected = tensor.clone()
        dist.all_reduce(expected, group=group.cpu_group)
        torch.testing.assert_close(group.all_reduce(tensor), expected, rtol=0, atol=0)
    pp_group.destroy()


def _rank_rows(rows, width, rank, dtype):
    return (torch.arange(rows * width).reshape(width, rows).T + rank * 32).to(dtype)


def _gather_contract_worker(rank, world_size, tp_size, dp_size, port, dp_port, params):
    os.environ["VLLM_DIST_IDENT"] = f"test_cpu_gather_contract_{port}"
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
    group, _ = _get_dp_gloo_communicator()
    local_rank = group.rank_in_group
    retained = []
    patterns = [[3] + [1] * (dp_size - 1), [0] + [3] * (dp_size - 1)]
    patterns += [[0] * dp_size, [2] * dp_size]
    for sizes in patterns:
        tensors = [
            _rank_rows(sizes[local_rank], width, rank, dtype)
            for width, dtype in (
                (3, torch.float32),
                (2, torch.bfloat16),
                (1, torch.int64),
            )
        ]
        originals = [tensor.clone() for tensor in tensors]
        outputs = group.all_gatherv(tensors, sizes=sizes)
        for tensor, original, output in zip(tensors, originals, outputs):
            expected = torch.cat(
                [
                    _rank_rows(rows, tensor.shape[1], member, tensor.dtype)
                    for rows, member in zip(sizes, group.ranks)
                ]
            )
            assert output.dtype == tensor.dtype
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(tensor, original, rtol=0, atol=0)
            assert (
                output.untyped_storage().nbytes()
                == output.numel() * output.element_size()
            )
            retained.append((output, expected))
        if sizes == [2] * dp_size:
            torch.testing.assert_close(group.all_gatherv(tensors[0]), outputs[0])
        empty_features = torch.empty((sizes[local_rank], 0), dtype=torch.float32)
        assert group.all_gatherv(empty_features, sizes=sizes).shape == (sum(sizes), 0)
        assert group.all_gatherv([], sizes=sizes) == []
    for output, expected in retained:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _scatter_contract_worker(rank, world_size, tp_size, dp_size, port, dp_port, params):
    os.environ["VLLM_DIST_IDENT"] = f"test_cpu_scatter_contract_{port}"
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
    group, _ = _get_dp_gloo_communicator()
    retained = []
    for sizes in ([3, 1], [0, 4], [0, 0], [2, 2], None):
        splits = [2, 2] if sizes is None else sizes
        for dim in (0, 1, -1):
            for dtype in (torch.float32, torch.bfloat16):
                shape = (sum(splits), 3) if dim == 0 else (3, sum(splits))
                tensor = torch.arange(shape[0] * shape[1]).reshape(shape[::-1]).T
                tensor = tensor.to(dtype) + rank * 32
                original = tensor.clone()
                reduced = tensor.clone()
                dist.all_reduce(reduced, group=group.cpu_group)
                expected = reduced.narrow(
                    dim, sum(splits[: group.rank_in_group]), splits[group.rank_in_group]
                ).clone()
                output = group.reduce_scatterv(tensor, dim=dim, sizes=sizes)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(tensor, original, rtol=0, atol=0)
                assert output.is_contiguous()
                assert (
                    output.untyped_storage().nbytes()
                    == output.numel() * output.element_size()
                )
                if output.numel():
                    assert (
                        output.untyped_storage().data_ptr()
                        != tensor.untyped_storage().data_ptr()
                    )
                into_output = torch.empty_like(expected)
                assert (
                    group.reduce_scatterv_into_output(
                        tensor, into_output, dim=dim, sizes=sizes
                    )
                    is into_output
                )
                torch.testing.assert_close(into_output, expected, rtol=0, atol=0)
                retained.append((output, expected))
    for output, expected in retained:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _validation_worker(rank, world_size, tp_size, dp_size, port, dp_port, case):
    os.environ["VLLM_DIST_IDENT"] = f"test_cpu_validation_{port}"
    _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
    group, _ = _get_dp_gloo_communicator()
    tensor = torch.ones((2, 3))
    cases = {
        "later_list_rows": lambda: group.all_gatherv(
            [tensor, tensor[:1]], sizes=[2, 2]
        ),
        "gather_length": lambda: group.all_gatherv(tensor, sizes=[2]),
        "gather_negative": lambda: group.all_gatherv(tensor, sizes=[-1, -1]),
        "gather_fractional": lambda: group.all_gatherv(tensor, sizes=[2.0, 2.0]),
        "gather_bool": lambda: group.all_gatherv(tensor[:1], sizes=[True, True]),
        "gather_local_rows": lambda: group.all_gatherv(tensor, sizes=[1, 1]),
        "gather_dim": lambda: group.all_gatherv(tensor, dim=1, sizes=[2, 2]),
        "scatter_length": lambda: group.reduce_scatterv(tensor, dim=0, sizes=[2]),
        "scatter_negative": lambda: group.reduce_scatterv(tensor, dim=0, sizes=[-1, 3]),
        "scatter_fractional": lambda: group.reduce_scatterv(
            tensor, dim=0, sizes=[0.5, 1.5]
        ),
        "scatter_bool": lambda: group.reduce_scatterv(
            tensor, dim=0, sizes=[True, True]
        ),
        "scatter_total": lambda: group.reduce_scatterv(tensor, dim=0, sizes=[2, 2]),
        "scatter_dim": lambda: group.reduce_scatterv(tensor, dim=2),
        "scatter_indivisible": lambda: group.reduce_scatterv(tensor, dim=1),
    }
    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile,
        pytest.raises(
            (AssertionError, ValueError, TypeError, NotImplementedError)
        ) as error,
    ):
        cases[case]()
    collectives = [
        event.key for event in profile.key_averages() if event.key.startswith("c10d::")
    ]
    assert collectives == [], (
        f"Malformed arguments issued data collectives: {collectives}"
    )
    expected_error = NotImplementedError if case == "gather_dim" else ValueError
    assert isinstance(error.value, expected_error)


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
        params=("eager", [[2, 1], [0, 3]]),
    )


@pytest.mark.distributed
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
@pytest.mark.skipif(
    os.environ.get("VLLM_CPU_EP_COMPILE_TEST") != "1",
    reason="CPU EP Inductor compile test is opt-in",
)
def test_cpu_ep_compile_ragged_fastpath_inductor():
    _spawn_workers(
        _compile_fastpath_worker,
        world_size=3,
        tp_size=1,
        dp_size=3,
        params=(
            "inductor",
            [[5, 5, 5], [5, 1, 3], [1, 3, 5], [3, 5, 1]],
        ),
    )


@pytest.mark.distributed
def test_cpu_dp_gloo_with_separate_worker_rendezvous():
    _spawn_workers(
        _dp_gloo_ragged_gather_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[2, 1],
        distributed_init_ports=[get_open_port(), get_open_port()],
        dp_port=get_open_port(),
    )


@pytest.mark.distributed
def test_cpu_dp_all_gatherv_retains_outputs_across_changed_sizes():
    _spawn_workers(
        _ragged_output_lifetime_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[[1, 0], [3, 1]],
    )


@pytest.mark.distributed
def test_cpu_dp_collective_defaults_empty_and_input_preservation():
    _spawn_workers(
        _default_collectives_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
    )


@pytest.mark.distributed
def test_cpu_dp_gloo_ragged_all_gatherv():
    _spawn_workers(
        _dp_gloo_ragged_gather_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[2, 1],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_shm_all_reduce_matches_gloo_without_fallback():
    _spawn_workers(
        _shm_all_reduce_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=None,
    )


@pytest.mark.distributed
def test_cpu_dp_ep_uses_gloo_preserves_tp_pp_shm():
    """DP/EP use Gloo while available TP/PP SHM keeps exact sums."""
    _spawn_workers(_transport_worker, world_size=4, tp_size=2, dp_size=2, params=None)


@pytest.mark.distributed
def test_cpu_dp_all_reduce_matches_gloo():
    """Six DP ranks retain the former SHM case's collective-value regression."""
    _spawn_workers(_transport_worker, world_size=6, tp_size=1, dp_size=6, params=None)


@pytest.mark.distributed
@pytest.mark.parametrize("dp_size", [2, 3])
def test_cpu_all_gatherv_subgroup_contract(dp_size):
    """Mixed noncontiguous tensors retain subgroup order, dtype and older outputs."""
    _spawn_workers(
        _gather_contract_worker,
        world_size=2 * dp_size,
        tp_size=2,
        dp_size=dp_size,
        params=None,
    )


@pytest.mark.distributed
def test_cpu_reduce_scatterv_subgroup_contract():
    """SUM shards use subgroup rank and own compact storage for every scatter dim."""
    _spawn_workers(
        _scatter_contract_worker, world_size=4, tp_size=2, dp_size=2, params=None
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    "case",
    [
        "later_list_rows",
        "gather_length",
        "gather_negative",
        "gather_fractional",
        "gather_bool",
        "gather_local_rows",
        "gather_dim",
        "scatter_length",
        "scatter_negative",
        "scatter_fractional",
        "scatter_bool",
        "scatter_total",
        "scatter_dim",
        "scatter_indivisible",
    ],
)
def test_cpu_collectives_validate_before_data_collectives(case):
    """Reject malformed arguments before issuing a real Gloo data collective."""
    _spawn_workers(_validation_worker, world_size=2, tp_size=1, dp_size=2, params=case)
