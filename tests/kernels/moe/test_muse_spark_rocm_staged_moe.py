# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark ROCm MXFP4 experts through AITER's two-stage MoE API.

On one GPU, the native-route experts must match the repeated-input experts bit
for bit. On 2 and 8 GPUs, the column-sharded down projection must match the
AITER experts followed by the reduce-scatter finalize.
"""

import os
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
)
from vllm.models.muse_spark.amd import staged_moe
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseSparkAiterExperts,
    MuseSparkQuarkOCPMXMoEQuantConfig,
)
from vllm.platforms import current_platform

ROCM_GFX950 = False
ROCM_AITER_SUPPORTED = False
if current_platform.is_rocm():
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_SUPPORTED = is_aiter_found_and_supported()

requires_staged_aiter = pytest.mark.skipif(
    not (ROCM_GFX950 and ROCM_AITER_SUPPORTED and staged_moe.aiter_has_staged_moe()),
    reason="requires a GFX950 GPU and AITER with the two-stage MoE API",
)

# Muse Spark Open: 256 experts, top 8, hidden 4096, intermediate 4096.
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, WORLD = 256, 8, 4096, 4096, 8


def _moe_config(
    *,
    tp_size: int = WORLD,
    ep_size: int = 1,
    hidden_dim: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    activation: MoEActivation = MoEActivation.SILU,
    device: str = "cpu",
) -> FusedMoEConfig:
    """Rank 0's MoE config; under EP the MoE runs at tp_size 1."""
    parallel_config = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=1,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=ep_size > 1,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOP_K,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_local_experts=NUM_EXPERTS // ep_size,
        num_logical_experts=NUM_EXPERTS,
        activation=activation,
        device=torch.device(device),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        moe_backend="auto",
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, True),
        ({"tp_size": 2}, True),
        ({"tp_size": 1}, False),
        ({"tp_size": 1, "ep_size": 8}, False),
        ({"tp_size": 8, "ep_size": 8}, False),
        ({"hidden_dim": 3072}, False),
        ({"intermediate_size": 3072}, False),
        ({"activation": MoEActivation.GELU}, False),
    ],
)
def test_column_sharded_down_deployments(monkeypatch, kwargs, expected) -> None:
    monkeypatch.setattr(staged_moe, "aiter_has_staged_moe", lambda: True)
    config = _moe_config(**kwargs)
    assert staged_moe.column_sharded_down_supported(config) is expected


def _post_expert_norm() -> torch.nn.Module:
    from vllm.models.muse_spark.common.layers import MuseSparkRMSNormWithInputScale

    norm = MuseSparkRMSNormWithInputScale(
        hidden_size=HIDDEN,
        eps=1e-5,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    )
    with torch.no_grad():
        norm.weight.normal_(0.0, 0.1)
    return norm.cuda()


def _mxfp4_experts(config: FusedMoEConfig):
    """Random MXFP4 experts in the AITER layout and their quant config."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    experts = config.num_local_experts
    intermediate = config.intermediate_size_per_partition

    def quantize(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.randn(experts * rows, cols, device="cuda", dtype=torch.bfloat16)
        packed, scale = dynamic_mxfp4_quant(w * cols**-0.5)
        return packed.view(experts, rows, -1), scale.view(experts, rows, -1)

    w13, w13_scale = quantize(2 * intermediate, HIDDEN)
    w2, w2_scale = quantize(HIDDEN, intermediate)
    w13, w2, w13_scale, w2_scale, _, _ = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            layer=SimpleNamespace(w13_input_scale=None, w2_input_scale=None),
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
        )
    )
    base = make_mxfp4_moe_quant_config(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
    )
    assert base is not None
    base_fields = {
        f.name: getattr(base, f.name) for f in fields(FusedMoEQuantConfig) if f.init
    }
    quant_config = MuseSparkQuarkOCPMXMoEQuantConfig(
        **base_fields, post_expert_norm_fn=_post_expert_norm().forward
    )
    return w13, w2, quant_config


@pytest.fixture(scope="module", params=["tp", "ep"])
def rank(request):
    """One TP8 or EP8 rank: native-route experts and repeated-input experts."""
    if not (ROCM_GFX950 and ROCM_AITER_SUPPORTED):
        pytest.skip("requires a GFX950 GPU with supported AITER")
    if not staged_moe.aiter_has_staged_moe():
        pytest.skip("requires AITER with the two-stage MoE API")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1")
        mp.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        rocm_aiter_ops.refresh_env_variables()
        torch.manual_seed(0)
        ep = request.param == "ep"
        config = _moe_config(
            tp_size=1 if ep else WORLD, ep_size=WORLD if ep else 1, device="cuda"
        )
        w13, w2, quant_config = _mxfp4_experts(config)
        expert_map = expert_mask = None
        if ep:
            _, expert_map, expert_mask = determine_expert_map(
                WORLD, 0, NUM_EXPERTS, return_expert_mask=True
            )
            expert_map, expert_mask = expert_map.cuda(), expert_mask.cuda()
        native = MuseSparkAiterExperts(config, quant_config)
        native.process_weights_after_loading(
            SimpleNamespace(w13_weight=w13, w2_weight=w2, _expert_map=expert_map)
        )
        yield SimpleNamespace(
            ep=ep,
            w13=w13,
            w2=w2,
            native=native,
            expanded=MuseSparkAiterExperts(config, quant_config),
            expert_map=expert_map,
            expert_mask=expert_mask,
        )
    rocm_aiter_ops.refresh_env_variables()


def _batch(num_tokens: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    logits = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=gen)
    topk_logits, topk_ids = logits.topk(TOP_K, dim=-1)
    return hidden_states, torch.softmax(topk_logits, dim=-1), topk_ids.to(torch.int32)


def _apply(r, experts, hidden_states, topk_weights, topk_ids, routes) -> None:
    experts.apply(
        routes,
        hidden_states,
        r.w13,
        r.w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        NUM_EXPERTS,
        r.expert_mask,
        None,
        None,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        None,
        False,
    )


def _finalize(experts, topk_weights, topk_ids, routes) -> torch.Tensor:
    reducer = experts.finalize_weight_and_reduce_impl()
    # TP's pre-norm all-reduce is a no-op on one rank.
    reducer.pre_norm_all_reduce = False
    out = torch.empty(routes.shape[0], HIDDEN, device="cuda", dtype=torch.bfloat16)
    return reducer.apply(out, routes, topk_weights, topk_ids, False)


# 128 and 4096 tokens hit tuned FlyDSL v2 stage2 rows, 1000 the heuristic one;
# TP keeps batches below 65 tokens on the repeated input.
@pytest.mark.parametrize("num_tokens", [16, 128, 1000, 4096])
@torch.inference_mode()
def test_native_routes_match_repeated_input(rank, num_tokens: int) -> None:
    r = rank
    if not r.native.uses_staged_moe(num_tokens):
        assert not r.ep and num_tokens < 65
        pytest.skip("batch size stays on the repeated input under TP")
    hidden_states, topk_weights, topk_ids = _batch(num_tokens, num_tokens)
    shape = (num_tokens, TOP_K, HIDDEN)
    # The repeated-input experts park their routes in a buffer of their own.
    unused = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    _apply(r, r.expanded, hidden_states, topk_weights, topk_ids, unused)
    expanded = r.expanded._take_route_output().clone()
    expected = _finalize(r.expanded, topk_weights, topk_ids, expanded).clone()

    # Routes to experts of other EP ranks stay NaN and must never be read.
    native = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    _apply(r, r.native, hidden_states, topk_weights, topk_ids, native)
    actual = _finalize(r.native, topk_weights, topk_ids, native)

    local = torch.ones_like(topk_ids, dtype=torch.bool)
    if r.ep:
        local = r.expert_map[topk_ids.long()] >= 0
        assert native[~local].isnan().all()
    torch.testing.assert_close(native[local], expanded[local], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_native_routes_graph_replay(rank) -> None:
    r = rank
    hidden_states, topk_weights, topk_ids = _batch(128, 7)
    routes = torch.empty(128, TOP_K, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        _apply(r, r.native, hidden_states, topk_weights, topk_ids, routes)
        return _finalize(r.native, topk_weights, topk_ids, routes)

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    hidden_states.copy_(torch.flip(hidden_states, dims=[0]))
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(captured.clone(), run(), atol=0, rtol=0)


# Muse Spark Open's MoE at TP8: 512 intermediate columns per rank.
_I_PER_RANK = INTERMEDIATE // WORLD


def _quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint-layout packed MXFP4 and unshuffled E8M0 scales."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    e, rows, cols = w.shape
    packed, scales = dynamic_mxfp4_quant(w.reshape(-1, cols))
    return (
        packed.view(torch.uint8).view(e, rows, -1),
        scales.view(torch.uint8).view(e, rows, -1)[..., : cols // 32].contiguous(),
    )


def _run_rank(device: torch.device, num_tokens_list: list[int]) -> None:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.models.muse_spark.amd.muse_moe_runner import (
        MuseSparkTopKWeightAndReduce,
    )

    rank = get_tensor_model_parallel_rank()
    cols = HIDDEN // get_tensor_model_parallel_world_size()
    own = slice(rank * cols, (rank + 1) * cols)
    gen = torch.Generator(device=device).manual_seed(1234 + rank)

    def randn(*shape: int) -> torch.Tensor:
        return 0.05 * torch.randn(
            *shape, device=device, dtype=torch.bfloat16, generator=gen
        )

    w13_raw, w13_scale = _quantize_mxfp4(randn(NUM_EXPERTS, 2 * _I_PER_RANK, HIDDEN))
    w2_raw, w2_scale_raw = _quantize_mxfp4(randn(NUM_EXPERTS, HIDDEN, _I_PER_RANK))
    weights = staged_moe.ColumnShardedDownWeights.from_checkpoint(w2_raw, w2_scale_raw)
    # The re-sharded w2 must hold this rank's rows of the gathered slices.
    full_w2 = tensor_model_parallel_all_gather(w2_raw.contiguous(), dim=-1)
    torch.testing.assert_close(
        staged_moe._all_to_all_rows(w2_raw), full_w2[:, own], rtol=0, atol=0
    )

    w13_scale = e8m0_shuffle(w13_scale.reshape(-1, w13_scale.shape[-1])).view(
        w13_scale.shape
    )
    w2_scale = e8m0_shuffle(w2_scale_raw.reshape(-1, w2_scale_raw.shape[-1])).view(
        w2_scale_raw.shape
    )
    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = rocm_aiter_ops.shuffle_weights(w13_raw.view(fp4), w2_raw.view(fp4))
    w13.is_shuffled = w2.is_shuffled = True
    torch.manual_seed(7)
    norm = _post_expert_norm().to(device)

    for num_tokens in num_tokens_list:
        # Every TP rank sees the same batch.
        torch.manual_seed(num_tokens)
        hidden_states = torch.randn(
            num_tokens, HIDDEN, device=device, dtype=torch.bfloat16
        )
        topk_ids = torch.rand(num_tokens, NUM_EXPERTS, device=device).topk(TOP_K)
        topk_ids = topk_ids.indices.to(torch.int32)
        topk_weights = torch.rand(num_tokens, TOP_K, device=device)
        unit = torch.ones(num_tokens * TOP_K, 1, device=device)

        # Reference: per-rank partial sums of the repeated input, all-reduced.
        partial = fused_moe(
            hidden_states.repeat_interleave(TOP_K, dim=0),
            w13,
            w2,
            unit,
            topk_ids.view(-1, 1),
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        )
        out = torch.empty(
            num_tokens, TOP_K, HIDDEN, device=device, dtype=torch.bfloat16
        )
        routes = staged_moe.staged_fused_moe(
            hidden_states,
            w13,
            w2,
            topk_ids,
            unit.view(num_tokens, TOP_K),
            w13_scale,
            w2_scale,
            out,
            column_weights=weights,
        )
        assert isinstance(routes, staged_moe.ColumnShardedRoutes)

        # The reference rounds each rank's partial sum to bf16 (at most 2^-8 of
        # its magnitude), this path rounds the full sum once; 2x margin.
        ref = tensor_model_parallel_all_reduce(partial.float())[:, own]
        abs_sum = tensor_model_parallel_all_reduce(partial.float().abs())[:, own]
        diff = routes.routes.reshape(-1, cols).float() - ref
        bound = 2.0**-7 * (abs_sum + ref.abs()) + 1e-6
        assert (diff.abs() <= bound).all(), f"T={num_tokens}: routes out of bound"

        finalize = MuseSparkTopKWeightAndReduce(norm.forward, pre_norm_all_reduce=True)
        expected = finalize.apply(
            None, partial.view(num_tokens, TOP_K, HIDDEN), topk_weights, topk_ids, False
        )
        finalize.take_route_output = lambda routes=routes: routes
        actual = finalize.apply(None, out, topk_weights, topk_ids, False)
        err = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert err < 1e-2, f"T={num_tokens}: output error {err:.3g}"
        # Every rank must hold the same output.
        first = tensor_model_parallel_all_gather(actual[None], dim=0)[0]
        torch.testing.assert_close(actual, first, rtol=0, atol=0)


def _worker(rank: int, world_size: int, port: int, num_tokens: list[int]) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.system_utils import update_environment_variables

    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "VLLM_ROCM_USE_AITER": "1",
        }
    )
    try:
        with set_current_vllm_config(VllmConfig()), torch.inference_mode():
            init_distributed_environment()
            initialize_model_parallel(tensor_model_parallel_size=world_size)
            _run_rank(device, num_tokens)
    finally:
        cleanup_dist_env_and_memory()
    # A normal exit runs C++ static destructors while RCCL / c10d threads may
    # still be winding down, and aborts; failures have raised above.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@requires_staged_aiter
@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.distributed(num_gpus=8)
def test_column_sharded_down_matches_reduce_scatter(world_size: int) -> None:
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    from vllm.utils.network_utils import get_open_port

    # 1031 is not a multiple of the stage 2 tile.
    mp.spawn(
        _worker,
        args=(world_size, get_open_port(), [1024, 1031, 8192]),
        nprocs=world_size,
        join=True,
    )
