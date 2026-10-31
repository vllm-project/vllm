# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU quantized MoE expert-parallel integration coverage."""

import math
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


def _run_single_rank_reference(
    a,
    w1,
    w2,
    w1_s,
    w2_s,
    topk_weight,
    topk_ids,
    E,
    *,
    quant_method=ops.CPUQuantMethod.FP8_W8A16,
    w1_bias=None,
    w2_bias=None,
    alpha=None,
    limit=None,
    act_method=ops.CPUActMethod.SILU_AND_MUL,
):
    """Single-process MoE forward over all experts (no EP masking)."""
    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)
    if quant_method == ops.CPUQuantMethod.MXFP4:
        w1_s = torch.ops._C.convert_scale_packed(w1_s)
        w2_s = torch.ops._C.convert_scale_packed(w2_s)
    a_in = a.clone()
    output = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        output,
        a_in,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        quant_method,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE if quant_method == ops.CPUQuantMethod.FP8_W8A16 else None,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        alpha=alpha,
        limit=limit,
        is_vnni=True,
        act_method=act_method,
    )
    return output


def _moe_ep_worker(rank, world_size, tp_size, dp_size, port, dp_port, params):
    # Required by CpuCommunicator's SHM-group-name check on TP groups.
    os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_moe_ep_{port}")
    _init_tp_dp_environment(
        rank, tp_size, dp_size, port, dp_port, enable_expert_parallel=True
    )

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
        with dp_metadata.sp_local_sizes(
            sequence_parallel_size=sp_size, pcp_size=1, use_ep=True
        ):
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
        expert_out = torch.empty_like(a_gathered)
        ops.fused_experts_cpu_local_skip(
            expert_out,
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

        with dp_metadata.sp_local_sizes(
            sequence_parallel_size=sp_size, pcp_size=1, use_ep=True
        ):
            # Non-SP combines to the DP-local slice per TP lane. SP combines
            # to the TP-local token chunk, then the TP all-gather below
            # restores the DP-local slice.
            combined = ep_group.combine(
                expert_out,
                is_sequence_parallel=is_sequence_parallel,
            )

    tp_group = get_tp_group()
    if is_sequence_parallel:
        combined = tp_group.all_gather(combined, dim=0)[:M_per_dp]
    else:
        # TP all-reduce merges complementary expert shards for each DP replica.
        combined = tp_group.all_reduce(combined)

    torch.testing.assert_close(combined, ref[lo:hi].bfloat16(), atol=1e-2, rtol=1e-2)

    dist.barrier()


def _moe_prepare_finalize_compile_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
):
    backend, quant_format = params
    os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_moe_prepare_finalize_{port}")
    _init_tp_dp_environment(
        rank, tp_size, dp_size, port, dp_port, enable_expert_parallel=True
    )
    torch._dynamo.reset()

    from vllm.config.parallel import ParallelConfig
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
        mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        determine_expert_map,
    )
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUExpertsFp8,
        CPUExpertsMxfp4,
        prepare_mxfp4_moe_layer_for_cpu,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    token_counts = [5, 4, 3]
    hidden_size = 128
    intermediate_size = 128
    num_experts = 10
    topk = 2
    dp_rank = rank // tp_size
    num_real_tokens = token_counts[dp_rank]

    torch.manual_seed(0)
    all_hidden = torch.randn(
        sum(token_counts), hidden_size, dtype=torch.bfloat16
    ) / math.sqrt(hidden_size)
    if quant_format == "mxfp4":
        w1 = torch.randint(
            256,
            (num_experts, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
        )
        w2 = torch.randint(
            256, (num_experts, hidden_size, intermediate_size // 2), dtype=torch.uint8
        )
        w1_s = torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            123,
            dtype=torch.uint8,
        )
        w2_s = torch.full(
            (num_experts, hidden_size, intermediate_size // 32), 123, dtype=torch.uint8
        )
        w1_bias = torch.randn(num_experts, 2 * intermediate_size) / 10
        w2_bias = torch.randn(num_experts, hidden_size) / 10
        activation = MoEActivation.SWIGLUOAI
        alpha, limit = 1.702, 7.0
        quant_method = ops.CPUQuantMethod.MXFP4
    else:
        w1, w2, w1_s, w2_s = _make_fp8_moe_weights(
            num_experts, intermediate_size, hidden_size, BLOCK_SIZE
        )
        w1_bias = w2_bias = alpha = limit = None
        activation = MoEActivation.SILU
        quant_method = ops.CPUQuantMethod.FP8_W8A16
    scores = torch.softmax(
        torch.randn(sum(token_counts), num_experts, dtype=torch.bfloat16),
        dim=-1,
        dtype=torch.float32,
    )
    topk_weights_all, topk_ids_all = torch.topk(scores, topk)
    topk_ids_all = topk_ids_all.to(torch.int32)
    reference = _run_single_rank_reference(
        all_hidden,
        w1,
        w2,
        w1_s,
        w2_s,
        topk_weights_all,
        topk_ids_all,
        num_experts,
        quant_method=quant_method,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        alpha=alpha,
        limit=limit,
        act_method=ops.CPUActMethod.SWIGLU
        if quant_method == ops.CPUQuantMethod.MXFP4
        else ops.CPUActMethod.SILU_AND_MUL,
    )

    start = sum(token_counts[:dp_rank])
    end = start + num_real_tokens
    hidden = sequence_parallel_chunk(all_hidden[start:end].clone())
    topk_weights = sequence_parallel_chunk(topk_weights_all[start:end].clone())
    topk_ids = sequence_parallel_chunk(topk_ids_all[start:end].clone()).long()
    expected = sequence_parallel_chunk(reference[start:end].clone())

    parallel_config = FusedMoEParallelConfig.make(
        tp_size_=tp_size,
        pcp_size_=1,
        dp_size_=dp_size,
        sp_size_=tp_size,
        vllm_parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            enable_expert_parallel=True,
        ),
    )
    local_num_experts, expert_map, _ = determine_expert_map(
        ep_size=parallel_config.ep_size,
        ep_rank=parallel_config.ep_rank,
        global_num_experts=num_experts,
    )
    assert expert_map is not None
    local_experts = _get_local_expert_slice(expert_map, local_num_experts)
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=local_num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=parallel_config,
        activation=activation,
        in_dtype=all_hidden.dtype,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        has_bias=quant_format == "mxfp4",
    )
    if quant_format == "mxfp4":
        w1_local, w2_local, w1_scale, w2_scale = prepare_mxfp4_moe_layer_for_cpu(
            w1[local_experts].contiguous(),
            w2[local_experts].contiguous(),
            w1_s[local_experts].contiguous(),
            w2_s[local_experts].contiguous(),
        )
        quant_config = mxfp4_w4a16_moe_quant_config(
            w1_scale,
            w2_scale,
            w1_bias=w1_bias[local_experts].contiguous(),
            w2_bias=w2_bias[local_experts].contiguous(),
            gemm1_alpha=alpha,
            gemm1_clamp_limit=limit,
        )
        experts = CPUExpertsMxfp4(moe_config, quant_config)
    else:
        quant_config = FusedMoEQuantConfig.make(
            torch.float8_e4m3fn,
            block_shape=BLOCK_SIZE,
            w1_scale=w1_s[local_experts].contiguous(),
            w2_scale=w2_s[local_experts].contiguous(),
        )
        experts = CPUExpertsFp8(moe_config, quant_config)
        w1_local = _prepack_experts(w1[local_experts].contiguous())
        w2_local = _prepack_experts(w2[local_experts].contiguous())
    prepare_finalize = MoEPrepareAndFinalizeNaiveDPEPModular(
        is_sequence_parallel=True,
        num_dispatchers=world_size,
    )
    kernel = FusedMoEKernel(prepare_finalize, experts)

    with _make_forward_context(
        dp_rank,
        dp_size,
        num_real_tokens,
        token_counts,
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        assert dp_metadata.local_sizes is None

        def step(hidden_states, weights, ids):
            with dp_metadata.sp_local_sizes(
                sequence_parallel_size=tp_size, pcp_size=1, use_ep=True
            ) as sizes:
                output = kernel.apply(
                    hidden_states=hidden_states,
                    w1=w1_local,
                    w2=w2_local,
                    topk_ids=ids,
                    topk_weights=weights,
                    activation=activation,
                    global_num_experts=num_experts,
                    expert_map=expert_map,
                    apply_router_weight_on_input=False,
                )
                assert dp_metadata.local_sizes is sizes
                return output

        output = step(hidden, topk_weights, topk_ids)
        assert dp_metadata.local_sizes is None
        assert (
            output.untyped_storage().data_ptr() != hidden.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
        dist.barrier()

        with torch._dynamo.config.patch(
            capture_dynamic_output_shape_ops=True,
            error_on_recompile=True,
        ):
            compiled_step = torch.compile(step, fullgraph=True, backend=backend)
            output = compiled_step(hidden, topk_weights, topk_ids)
        assert dp_metadata.local_sizes is None
        assert (
            output.untyped_storage().data_ptr() != hidden.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)

    dist.barrier()


@pytest.mark.distributed
@pytest.mark.parametrize(
    "tp_size,dp_size,is_sequence_parallel",
    [(1, 6, False), (2, 3, False), (2, 3, True)],
    ids=["dp6", "tp2-dp3", "tp2-dp3-sp"],
)
def test_cpu_moe_ep_matches_all_experts_reference(
    tp_size, dp_size, is_sequence_parallel
):
    """Nondivisible expert placement matches the full FP8 expert reference."""
    _spawn_workers(
        _moe_ep_worker,
        world_size=6,
        tp_size=tp_size,
        dp_size=dp_size,
        params=(
            7 if is_sequence_parallel else 8,
            256,
            512,
            10,
            2,
            0,
            is_sequence_parallel,
        ),
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        pytest.param(
            "inductor",
            marks=pytest.mark.skipif(
                os.environ.get("VLLM_CPU_EP_COMPILE_TEST") != "1",
                reason="six-rank CPU EP compile regression is opt-in",
            ),
        ),
    ],
)
@pytest.mark.parametrize("quant_format", ["fp8", "mxfp4"])
def test_cpu_moe_prepare_finalize_sequence_parallel_compile_tp2_dp3(
    backend, quant_format
):
    """Prepare/finalize writes the combined CPU result into its output buffer."""
    _spawn_workers(
        _moe_prepare_finalize_compile_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=(backend, quant_format),
    )
