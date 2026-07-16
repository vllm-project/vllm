# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU FP8 MoE expert-parallel integration coverage."""

import math
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

import vllm._custom_ops as ops  # noqa: E402
from vllm.model_executor.models.utils import sequence_parallel_chunk  # noqa: E402

if not hasattr(torch.ops._C, "fused_experts_cpu"):
    pytest.skip("fused_experts_cpu op not available", allow_module_level=True)

HAS_CPU_SHM = hasattr(torch.ops._C, "init_shm_manager") and (
    current_platform.get_cpu_architecture()
    in (CpuArchEnum.X86, CpuArchEnum.ARM, CpuArchEnum.POWERPC)
)


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
            target=worker_fn,
            args=(
                rank,
                world_size,
                tp_size,
                dp_size,
                distributed_init_ports[rank],
                dp_port,
                params,
                err_q,
            ),
        )
        proc.start()
        procs.append(proc)

    failures = _collect_worker_failures(procs, err_q)
    if failures:
        pytest.fail("Worker(s) failed:\n" + "\n---\n".join(failures))


# ---------------------------------------------------------------------------
# Shared helpers (mirrors tests/kernels/moe/test_cpu_quant_fused_moe.py)
# ---------------------------------------------------------------------------

BLOCK_SIZE = [128, 128]
_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_SCALE = _FP8_INFO.max
FACTOR_FOR_SCALE = 1e-3


def _prepack_experts(w: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.convert_weight_packed(w)


def _make_fp8_moe_weights(E, N, K, block_size):
    block_n, block_k = block_size
    w1 = (
        (torch.randn(E, 2 * N, K) * FP8_SCALE)
        .clamp(-FP8_SCALE, FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        (torch.randn(E, K, N) * FP8_SCALE)
        .clamp(-FP8_SCALE, FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )
    w1_s = (
        torch.randn(E, math.ceil(2 * N / block_n), math.ceil(K / block_k))
        * FACTOR_FOR_SCALE
    )
    w2_s = (
        torch.randn(E, math.ceil(K / block_n), math.ceil(N / block_k))
        * FACTOR_FOR_SCALE
    )
    return w1, w2, w1_s, w2_s


def _get_local_expert_slice(
    expert_map: torch.Tensor,
    local_num_experts: int,
) -> slice:
    owned = (expert_map != -1).nonzero(as_tuple=False)
    expert_lo = int(owned[0].item())
    return slice(expert_lo, expert_lo + local_num_experts)


def _run_single_rank_reference(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, E):
    """Single-process FP8 MoE forward over all E experts (no EP masking)."""
    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)
    return ops.fused_experts_cpu(
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        False,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )


def _sequence_parallel_all_gather(
    tensor: torch.Tensor,
    tp_group,
    num_tokens: int,
) -> torch.Tensor:
    return tp_group.all_gather(tensor, dim=0)[:num_tokens]


# ---------------------------------------------------------------------------
# Distributed environment setup
# ---------------------------------------------------------------------------


def _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port):
    """Init vLLM distributed env for TP=tp_size, DP=dp_size.

    global_rank = dp_rank * tp_size + tp_rank
    Each DP replica contributes tp_size workers; init_distributed_environment
    offsets rank by dp_rank*tp_size internally.
    """
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
        enable_expert_parallel=True,
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


def _make_forward_context(
    dp_rank,
    dp_size,
    num_tokens_per_dp_rank,
    num_tokens_across_dp=None,
):
    """Forward context with DP metadata for EP group dispatch."""
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
        num_tokens=num_tokens_per_dp_rank,
        num_tokens_across_dp=torch.tensor(
            num_tokens_across_dp
            if num_tokens_across_dp is not None
            else [num_tokens_per_dp_rank] * dp_size,
            dtype=torch.int,
        ),
    )


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _moe_ep_worker(rank, world_size, tp_size, dp_size, port, dp_port, params, err_q):
    try:
        # Required by CpuCommunicator's SHM-group-name check on TP groups.
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_moe_ep_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
            _CPUSHMDistributed,
        )
        from vllm.distributed.parallel_state import (
            get_ep_group,
            get_tp_group,
        )
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            determine_expert_map,
        )

        M_per_dp, N, K, E, topk, seed, is_sequence_parallel = params
        torch.manual_seed(seed)

        dp_rank = rank // tp_size
        tp_rank = rank % tp_size
        ep_rank = dp_rank * tp_size + tp_rank  # flattened ep rank

        # Build shared inputs (same across all ranks via fixed seed)
        # M_per_dp tokens per DP replica; total M = dp_size * M_per_dp
        total_M = dp_size * M_per_dp
        a_all = torch.randn(total_M, K, dtype=torch.bfloat16) / math.sqrt(K)
        w1, w2, w1_s, w2_s = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)
        score = torch.softmax(
            torch.randn(total_M, E, dtype=torch.bfloat16), dim=-1, dtype=torch.float32
        )
        topk_weight_all, topk_ids_all = torch.topk(score, topk)
        topk_ids_all = topk_ids_all.to(torch.int32)

        # Single-rank reference (no EP, all experts)
        ref = _run_single_rank_reference(
            a_all, w1, w2, w1_s, w2_s, topk_weight_all, topk_ids_all, E
        )

        # This rank's DP slice
        lo, hi = dp_rank * M_per_dp, (dp_rank + 1) * M_per_dp
        a_local = a_all[lo:hi].clone()
        topk_weight_local = topk_weight_all[lo:hi]
        topk_ids_local_global = topk_ids_all[lo:hi]

        if is_sequence_parallel:
            a_local = sequence_parallel_chunk(a_local)
            topk_weight_local = sequence_parallel_chunk(topk_weight_local)
            topk_ids_local_global = sequence_parallel_chunk(topk_ids_local_global)

        # Expert placement for this EP rank
        ep_size = dp_size * tp_size
        local_num_experts, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=E,
        )
        assert expert_map is not None

        local_experts = _get_local_expert_slice(expert_map, local_num_experts)
        w1_local = w1[local_experts].contiguous()
        w2_local = w2[local_experts].contiguous()
        w1_s_local = w1_s[local_experts].contiguous()
        w2_s_local = w2_s[local_experts].contiguous()

        with _make_forward_context(dp_rank, dp_size, M_per_dp):
            ep_group = get_ep_group()
            ep_communicator = ep_group.device_communicator
            assert isinstance(ep_communicator, CpuCommunicator)
            if HAS_CPU_SHM:
                assert isinstance(ep_communicator.dist_module, _CPUSHMDistributed)
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            sp_size = tp_size if is_sequence_parallel else 1
            with dp_metadata.sp_local_sizes(sequence_parallel_size=sp_size):
                # Non-SP dispatch gathers each TP lane's DP group. SP dispatch
                # gathers the full EP group after sequence_parallel_chunk().
                a_gathered, tw_gathered, tid_gathered = ep_group.dispatch(
                    a_local.clone(),
                    topk_weight_local.clone(),
                    topk_ids_local_global.clone().long(),
                    is_sequence_parallel=is_sequence_parallel,
                )

            # Expert compute: skip non-local selections (the shipped path).
            pw1, pw2 = _prepack_experts(w1_local), _prepack_experts(w2_local)
            expert_out = ops.fused_experts_cpu_local_skip(
                a_gathered.clone(),
                pw1,
                pw2,
                tw_gathered,
                tid_gathered,
                expert_map,
                ops.CPUQuantMethod.FP8_W8A16,
                w1_s_local,
                w2_s_local,
                None,  # w1_zero
                None,  # w2_zero
                BLOCK_SIZE,
                None,  # w1_bias
                None,  # w2_bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni
            )

            with dp_metadata.sp_local_sizes(sequence_parallel_size=sp_size):
                # Non-SP combines to the DP-local slice per TP lane. SP combines
                # to the TP-local token chunk, then the TP all-gather below
                # restores the DP-local slice.
                combined = ep_group.combine(
                    expert_out,
                    is_sequence_parallel=is_sequence_parallel,
                )

        tp_group = get_tp_group()
        if is_sequence_parallel:
            combined = _sequence_parallel_all_gather(
                combined,
                tp_group,
                M_per_dp,
            )
        else:
            # TP all-reduce merges complementary expert shards for each DP replica.
            combined = tp_group.all_reduce(combined)

        # Gather results from all ranks at rank 0 for verification.
        # Each rank holds its DP-slice result [M_per_dp, K].
        # We need to reconstruct the full [total_M, K] and compare to ref.
        # Use TP group rank 0 per DP replica to avoid double-counting:
        # TP partners now hold identical combined tensors post-all-reduce.
        # new_group is a collective — all ranks must call it.
        tp0_ranks = [r for r in range(world_size) if r % tp_size == 0]
        tp0_group = dist.new_group(ranks=tp0_ranks, backend="gloo")

        result_full = None
        if tp_rank == 0:
            # Only TP-rank-0 workers participate in gathering
            all_slices = [torch.zeros_like(combined) for _ in range(dp_size)]
            dist.all_gather(all_slices, combined, group=tp0_group)
            result_full = torch.cat(all_slices, dim=0)

        if rank == 0:
            assert result_full is not None
            torch.testing.assert_close(
                ref.bfloat16(), result_full, atol=1e-2, rtol=1e-2
            )

        dist.barrier()

    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _make_monolithic_ep_step(
    backend,
    packed_w1,
    packed_w2,
    w1_scale,
    w2_scale,
    expert_map,
    topk,
    total_sp_tokens,
    local_sp_tokens,
    num_real_tokens,
):
    from vllm.distributed.parallel_state import get_ep_group, get_tp_group
    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts

    def step(hidden_states, router_logits):
        hidden_states, router_logits = get_ep_group().dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel=True,
        )
        torch._check(hidden_states.shape[0] == total_sp_tokens)
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=False,
            top_k=topk,
            renormalize=False,
        )
        expert_out = ops.fused_experts_cpu_local_skip(
            hidden_states,
            packed_w1,
            packed_w2,
            topk_weights,
            topk_ids,
            expert_map,
            ops.CPUQuantMethod.FP8_W8A16,
            w1_scale,
            w2_scale,
            None,
            None,
            BLOCK_SIZE,
            None,
            None,
            None,
            None,
            True,
        )
        combined = get_ep_group().combine(
            expert_out,
            is_sequence_parallel=True,
        )
        torch._check(combined.shape[0] == local_sp_tokens)
        return get_tp_group().all_gather(combined, dim=0)[:num_real_tokens]

    if backend == "none":
        return step
    return torch.compile(step, fullgraph=True, backend=backend)


def _run_all_experts_reference(
    hidden_states,
    router_logits,
    packed_w1,
    packed_w2,
    w1_scale,
    w2_scale,
    topk,
):
    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        use_grouped_topk=False,
        top_k=topk,
        renormalize=False,
    )
    return ops.fused_experts_cpu(
        hidden_states,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        False,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_scale,
        w2_scale,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )


@torch._dynamo.config.patch(
    capture_dynamic_output_shape_ops=True,
    error_on_recompile=True,
)
def _run_monolithic_ep_step(
    backend,
    hidden_inputs,
    router_inputs,
    packed_w1,
    packed_w2,
    w1_scale,
    w2_scale,
    expert_map,
    topk,
    total_sp_tokens,
    local_sp_tokens,
    num_real_tokens,
):
    step = _make_monolithic_ep_step(
        backend,
        packed_w1,
        packed_w2,
        w1_scale,
        w2_scale,
        expert_map,
        topk,
        total_sp_tokens,
        local_sp_tokens,
        num_real_tokens,
    )
    return [
        step(hidden_states, router_logits)
        for hidden_states, router_logits in zip(hidden_inputs, router_inputs)
    ]


def _moe_ep_monolithic_compile_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    backend,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_moe_ep_compile_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
        torch._dynamo.reset()

        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            determine_expert_map,
        )

        token_counts = [5, 4, 3]
        hidden_size = 256
        intermediate_size = 128
        num_experts = 12
        topk = 2
        dp_rank = rank // tp_size
        num_real_tokens = token_counts[dp_rank]
        local_sp_tokens = math.ceil(num_real_tokens / tp_size)
        total_sp_tokens = sum(
            math.ceil(num_tokens / tp_size) * tp_size for num_tokens in token_counts
        )

        torch.manual_seed(0)
        hidden_variants = [
            torch.randn(
                sum(token_counts),
                hidden_size,
                dtype=torch.bfloat16,
            )
            / math.sqrt(hidden_size)
            for _ in range(2)
        ]
        router_variants = [
            torch.randn(
                sum(token_counts),
                num_experts,
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        ]
        w1, w2, w1_s, w2_s = _make_fp8_moe_weights(
            num_experts,
            intermediate_size,
            hidden_size,
            BLOCK_SIZE,
        )
        packed_w1 = _prepack_experts(w1)
        packed_w2 = _prepack_experts(w2)

        start = sum(token_counts[:dp_rank])
        end = start + num_real_tokens
        hidden_dp = [hidden[start:end].clone() for hidden in hidden_variants]
        router_dp = [router[start:end].clone() for router in router_variants]
        references = [
            _run_all_experts_reference(
                hidden,
                router,
                packed_w1,
                packed_w2,
                w1_s,
                w2_s,
                topk,
            )
            for hidden, router in zip(hidden_dp, router_dp)
        ]
        hidden_local = [sequence_parallel_chunk(hidden) for hidden in hidden_dp]
        router_local = [sequence_parallel_chunk(router) for router in router_dp]
        assert all(tensor.shape[0] == local_sp_tokens for tensor in hidden_local)

        local_num_experts, expert_map, _ = determine_expert_map(
            ep_size=world_size,
            ep_rank=rank,
            global_num_experts=num_experts,
        )
        assert expert_map is not None
        local_experts = _get_local_expert_slice(expert_map, local_num_experts)

        with _make_forward_context(
            dp_rank,
            dp_size,
            num_real_tokens,
            token_counts,
        ):
            outputs = _run_monolithic_ep_step(
                backend,
                hidden_local,
                router_local,
                _prepack_experts(w1[local_experts].contiguous()),
                _prepack_experts(w2[local_experts].contiguous()),
                w1_s[local_experts].contiguous(),
                w2_s[local_experts].contiguous(),
                expert_map,
                topk,
                total_sp_tokens,
                local_sp_tokens,
                num_real_tokens,
            )

        for output, reference in zip(outputs, references):
            assert output.shape[0] == token_counts[dp_rank]
            torch.testing.assert_close(
                output,
                reference.bfloat16(),
                atol=1e-2,
                rtol=1e-2,
            )

        output_lengths = [None] * world_size
        dist.all_gather_object(output_lengths, outputs[-1].shape[0])
        assert output_lengths == [5, 5, 4, 4, 3, 3]
        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# (M_per_dp, N, K, E, topk, seed)
_PARAMS = [
    (8, 256, 512, 10, 2, 0, False),
]

_SP_PARAMS = [
    (7, 256, 512, 10, 2, 0, True),
]


@pytest.mark.distributed
@pytest.mark.parametrize("params", _PARAMS, ids=["E10-nondiv"])
def test_cpu_moe_ep_dp6_tp1(params):
    """TP=1, DP=6: pure expert parallelism baseline (EP=6)."""
    _spawn_workers(_moe_ep_worker, world_size=6, tp_size=1, dp_size=6, params=params)


@pytest.mark.distributed
@pytest.mark.parametrize("params", _PARAMS, ids=["E10-nondiv"])
def test_cpu_moe_ep_tp2_dp3(params):
    """TP=2, DP=3: DP-lane AgRs plus final TP all-reduce (EP=6)."""
    _spawn_workers(_moe_ep_worker, world_size=6, tp_size=2, dp_size=3, params=params)


@pytest.mark.distributed
@pytest.mark.parametrize("params", _SP_PARAMS, ids=["E10-nondiv-sp"])
def test_cpu_moe_ep_tp2_dp3_sequence_parallel(params):
    """TP=2, DP=3: full EP-group AgRs plus TP all-gather restore."""
    _spawn_workers(_moe_ep_worker, world_size=6, tp_size=2, dp_size=3, params=params)


@pytest.mark.distributed
@pytest.mark.skipif(
    os.environ.get("VLLM_CPU_EP_MONOLITHIC_COMPILE_TEST") != "1",
    reason="six-rank CPU EP compile regression is opt-in",
)
@pytest.mark.parametrize("backend", ["none", "eager", "inductor"])
def test_cpu_moe_ep_monolithic_compile_tp2_dp3(backend):
    """Synthetic SP alignment rows are discarded after CPU EP combine."""
    _spawn_workers(
        _moe_ep_monolithic_compile_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=backend,
    )
