# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE layer.

Run `pytest tests/kernels/test_moe_layer.py`.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo, parallel_launch_with_config)
from tests.kernels.moe.utils import (TestMLP, make_test_weights,
                                     moe_quantize_weights)
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
from vllm.platforms import current_platform
from vllm.utils import cdiv, has_deep_ep, has_pplx

SHAPE_COMBOS = [
    (1, 16, 16),

    (1, 128, 128),
    #(32, 1024, 512),
#    (32, 512, 512),
    (222, 4096, 2048),
    #(256, 4096, 2048),
]

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]

# dp_size, tp_size, use_ep
PARALLEL_COMBOS = [
    [1, 1, False],
#    [1, 2, False],
    #[1, 4, False],
    [2, 1, True],
    #    [2, 2, True],
    #    [4, 1, True],
]

BACKENDS = ["naive"]

if has_pplx():
    BACKENDS += ["pplx"]

if has_deep_ep():
    BACKENDS += ["deepep_low_latency", "deepep_high_throughput"]

QUANT_METHODS = [
    None,
    "fp8",
    #"modelopt",
    #"compressed-tensors",
]


def rank_chunk(num: int, r: int, w: int) -> int:
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(
    t: torch.Tensor,
    r: int,
    w: int,
    dim: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    chunk = cdiv(t.shape[dim], w)
    t = t.narrow(dim, r * chunk, chunk)
    if device is not None:
        t = t.to(device)
    return t


def maybe_chunk_by_rank(
    t: Optional[torch.Tensor],
    r: int,
    w: int,
    dim: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    if t is not None:
        return chunk_by_rank(t, r, w, dim, device)
    else:
        return t


def chunk_scales_by_rank(
    t: Optional[torch.Tensor],
    r: int,
    w: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    if t is not None and t.numel() > 1:
        chunk = rank_chunk(t.shape[0], r, w)
        t = t[(r * chunk):(r + 1) * chunk]

    if t is not None and device is not None:
        t = t.to(device)

    return t


def chunk_scales(
    t: Optional[torch.Tensor],
    start: int,
    end: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    if t is not None and t.numel() > 1:
        t = t[start:end]

    if t is not None and device is not None:
        t = t.to(device)

    return t


def make_quant_config(
    quantization: Optional[str],
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[
        Optional[QuantizationConfig],
        torch.Tensor,  # quantized w1
        Optional[torch.Tensor],  # quantized w1 scales
        torch.Tensor,  # quantized w2
        Optional[torch.Tensor],  # quantized w1 scales
]:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant_config = None
    w1q = w1
    w2q = w2
    w1s = None
    w2s = None

    if quantization == "fp8":
        quant_config = Fp8Config(True)
        w1q, w1s, _ = moe_quantize_weights(w1, None, torch.float8_e4m3fn,
                                           False, None)
        w2q, w2s, _ = moe_quantize_weights(w2, None, torch.float8_e4m3fn,
                                           False, None)
        assert w1s is not None and w2s is not None
    elif quantization == "modelopt" or quantization == "compressed_tensors":
        raise NotImplementedError

    return quant_config, w1q, w1s, w2q, w2s


@dataclass
class SharedExpertsConfig:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_s: Optional[torch.Tensor] = None
    w2_s: Optional[torch.Tensor] = None
    quant_dtype: Union[torch.dtype, str, None] = None


def make_fused_moe_layer(
    quantization: Optional[str],
    use_ep: bool,
    hidden_size: int,
    intermediate_size: int,
    in_dtype: torch.dtype,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    reduce_results: bool,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    renormalize: bool = False,
    shared_experts: Optional[torch.nn.Module] = None,
    use_grouped_topk: bool = False,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    indices_type: Optional[torch.dtype] = None,
    expert_map: Optional[torch.Tensor] = None,
    enable_eplb: bool = False,
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
) -> tuple[Callable, FusedMoE]:
    quant_config, w1q, w1s, w2q, w2s = make_quant_config(quantization, w1, w2)

    kwargs = dict()
    if shared_experts is None:
        builder = FusedMoE
    else:
        builder = SharedFusedMoE
        kwargs["shared_experts"] = shared_experts

    layer = builder(
        num_experts=global_num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=in_dtype,
        reduce_results=reduce_results,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        quant_config=quant_config,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        prefix="test_layer",
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=activation,
        enable_eplb=enable_eplb,
        num_redundant_experts=num_redundant_experts,
        has_bias=has_bias,
        **kwargs,
    )

    local_num_experts = layer.local_num_experts
    print(f"LOCAL NE {global_num_experts}, {local_num_experts}")

    #
    # TODO: make sure parameter names correct for diff quantization types
    #
    layer.register_parameter("w13_weight",
                             torch.nn.Parameter(w1q, requires_grad=False))

    layer.register_parameter("w2_weight",
                             torch.nn.Parameter(w2q, requires_grad=False))

    if w1s is not None:
        assert w2s is not None
        layer.register_parameter("w13_weight_scale",
                                 torch.nn.Parameter(w1s, requires_grad=False))
        layer.register_parameter("w2_weight_scale",
                                 torch.nn.Parameter(w2s, requires_grad=False))

    if use_ep or tp_size > 1:
        assert layer.quant_method is not None
        layer.quant_method.init_prepare_finalize(layer)

    def _moe(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if shared_experts is None:
            final_shared_states = None
            final_hidden_states = layer(hidden_states, router_logits)
        else:
            final_shared_states, final_hidden_states = layer(
                hidden_states, router_logits)

        if shared_experts is not None:
            assert final_shared_states is not None
            final_hidden_states += final_shared_states

        if layer.tp_size > 1:
            final_hidden_states = layer.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states)

        return final_hidden_states

    return _moe, layer


def make_fake_moe_layer(
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    in_dtype: torch.dtype,
    quant_dtype: Optional[torch.dtype],
    renormalize: bool = False,
    shared_experts_config: Optional[SharedExpertsConfig] = None,
    use_grouped_topk: bool = False,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    indices_type: Optional[torch.dtype] = None,
    expert_map: Optional[torch.Tensor] = None,
    enable_eplb: bool = False,  # for now
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
) -> Callable:

    if quant_dtype is not None:
        w1, w1_s, _ = moe_quantize_weights(w1, None, quant_dtype, False, None)
        w2, w2_s, _ = moe_quantize_weights(w2, None, quant_dtype, False, None)
    else:
        w1_s = None
        w2_s = None

    if shared_experts_config is not None:
        shared_experts = TestMLP(
            shared_experts_config.w1,
            shared_experts_config.w2,
            in_dtype,
        )
    else:
        shared_experts = None

    def _moe(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=indices_type,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

        if shared_experts is not None:
            shared_output = shared_experts(hidden_states)
        else:
            shared_output = None

        output = fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_s,
            w2_scale=w2_s,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

        if shared_experts is not None:
            assert shared_output is not None
            output += shared_output

        return output

    return _moe


def _test_loop(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    ep_size: int,
    dp_size: int,
    tp_size: int,
    baseline_output: torch.Tensor,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    m: int,
    n: int,
    k: int,
    top_k: int,
    quantization: Optional[str],
    shared_experts_config: Optional[SharedExpertsConfig],
):
    print(f"PGI={pgi}")

    world_size = tp_size * dp_size
    use_ep = ep_size > 1

    assert vllm_config.parallel_config.enable_expert_parallel == use_ep

    torch.set_printoptions(profile="full")
    current_platform.seed_everything(7)

    in_dtype = hidden_states.dtype

    device = torch.cuda.current_device()
    dp_rank = vllm_config.parallel_config.rank
    tp_rank = pgi.rank

    print(f"CHUNK DP_RANK={dp_rank} DP={dp_size} TP_RANK={tp_rank} "
          f"TP={tp_size} EP={ep_size}")
    print(f"BEFORE W1 {w1.shape}")
    print(f"BEFORE W2 {w2.shape}")

    if ep_size > 1:
        #hidden_states = chunk_by_rank(hidden_states, dp_rank, dp_size, dim=1, device=device)
        #router_logits = chunk_by_rank(router_logits, dp_rank, dp_size, dim=1, device=device)
        w1 = chunk_by_rank(w1, dp_rank, dp_size, dim=0, device=device)
        w2 = chunk_by_rank(w2, dp_rank, dp_size, dim=0, device=device)

    if tp_size > 1:
        #hidden_states = chunk_by_rank(hidden_states, tp_rank, tp_size, dim=1, device=device)
        #router_logits = chunk_by_rank(router_logits, tp_rank, tp_size, dim=1, device=device)
        w1 = chunk_by_rank(w1, tp_rank, tp_size, dim=1, device=device)
        w2 = chunk_by_rank(w2, tp_rank, tp_size, dim=2, device=device)
        n = n // tp_size

    print(f"AFTER W1 {w1.shape}")
    print(f"AFTER W2 {w2.shape}")

    hidden_states = hidden_states.to(device)
    router_logits = router_logits.to(device)
    baseline_output = baseline_output.to(device)

    if shared_experts_config is not None:
        if False and tp_size > 1:
            s_w1 = chunk_by_rank(shared_experts_config.w1, tp_rank,
                                 tp_size, dim=1, device=device)
            s_w2 = chunk_by_rank(shared_experts_config.w2, tp_rank,
                                 tp_size, dim=2, device=device)
        else:
            s_w1 = shared_experts_config.w1.to(device)
            s_w2 = shared_experts_config.w2.to(device)

        shared_experts = TestMLP(
            w1=s_w1,
            w2=s_w2,
            out_dtype=in_dtype,
        )
    else:
        shared_experts = None

    with set_current_vllm_config(vllm_config):
        current_vllm_config = get_current_vllm_config()
        print(f"PCONF = {current_vllm_config.parallel_config}")
        moe_fn, moe_layer = make_fused_moe_layer(
            quantization=quantization,
            use_ep=use_ep,
            hidden_size=k,
            intermediate_size=n,
            in_dtype=in_dtype,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            reduce_results=False,
            w1=w1,
            w2=w2,
            top_k=top_k,
            global_num_experts=num_experts,
            shared_experts=shared_experts,
        )

        # What?
        if moe_layer.expert_map is not None:
            moe_layer.expert_map = moe_layer.expert_map.to(
                torch.cuda.current_device())

        # output should be completely reduced at this point
        num_tokens = m
        num_tokens_across_dp = torch.tensor([num_tokens] * world_size,
                                            device=torch.cuda.current_device(),
                                            dtype=torch.int)

        # Make moe_forward happy
        vllm_config.compilation_config.static_forward_context[
            "test_layer"] = moe_layer

        print(f"ORIG RANK HIDDEN_STATES {hidden_states}")

        with set_forward_context(
                None,
                vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                #num_tokens_across_dp=None,
        ):
            output = moe_fn(hidden_states, router_logits)

    if False and quantization is not None:
        atol = 5e-2
        rtol = 5e-2
    else:
        atol = 3.5e-2
        rtol = 3.5e-2

    print(f"OUTPUT {output}")

    torch.testing.assert_close(baseline_output, output, atol=atol, rtol=rtol)


# TODO: add cudagraphs/torch.compile
@pytest.mark.parametrize("m, n, k", SHAPE_COMBOS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("quantization", QUANT_METHODS)
@pytest.mark.parametrize("dp_size, tp_size, use_ep", PARALLEL_COMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_shared_experts", [False, True])
# split test between 1, 2, 4 gpus
#@multi_gpu_test(num_gpus=2)
def test_moe_layer(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    top_k: int,
    quantization: Optional[str],
    dp_size: int,
    tp_size: int,
    use_ep: bool,
    backend: str,
    use_shared_experts: bool,
):
    torch.set_printoptions(profile="full")
    current_platform.seed_everything(7)
    world_size = tp_size * dp_size

    if not use_ep and backend != "naive":
        pytest.skip(f"Skipping backend {backend} w/o EP.")

    if backend == "deepep_low_latency":
        from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (  # noqa: E501
            DeepEPLLPrepareAndFinalize)
        if k not in DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES:
            pytest.skip(f"Skipping unsupported K {k} in {backend} w/o EP.")

    test_env = dict()
    #test_env["VLLM_USE_DEEP_GEMM"] = "1"
    test_env["VLLM_ALL2ALL_BACKEND"] = backend

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=use_ep,
    )

    vllm_config = VllmConfig(parallel_config=parallel_config)

    in_dtype = torch.bfloat16

    # Just fp8 for now.
    quant_dtype = (torch.float8_e4m3fn if quantization is not None else None)

    (w1, _, _, _), (w2, _, _, _) = make_test_weights(
        num_experts,
        n,
        k,
        in_dtype=in_dtype,
    )

    if use_shared_experts:
        shared_experts_config = SharedExpertsConfig(
            w1=torch.randn((k, n * 2), device="cuda", dtype=in_dtype) / 15,
            w2=torch.randn((n, k), device="cuda", dtype=in_dtype) / 15,
        )
    else:
        shared_experts_config = None

    baseline_layer = make_fake_moe_layer(
        w1=w1,
        w2=w2,
        top_k=top_k,
        global_num_experts=num_experts,
        in_dtype=in_dtype,
        quant_dtype=None,  #quant_dtype,
        renormalize=False,
        shared_experts_config=shared_experts_config,
    )

    hidden_states = torch.randn((m, k), device="cuda", dtype=in_dtype) / 10
    router_logits = torch.randn((m, num_experts),
                                device="cuda",
                                dtype=in_dtype)

    print(f"ORIG HIDDEN_STATES {hidden_states}")

    hidden_states_clone = hidden_states.clone().detach()
    router_logits_clone = router_logits.clone().detach()

    baseline_output = baseline_layer(hidden_states, router_logits)

    print(f"BASE {baseline_output}")

    parallel_launch_with_config(
        world_size,
        _test_loop,
        vllm_config,
        test_env,
        1 if not use_ep else world_size, # or dp_size?
        dp_size,
        tp_size,
        baseline_output,
        hidden_states_clone,
        router_logits_clone,
        w1,
        w2,
        num_experts,
        m,
        n,
        k,
        top_k,
        quantization,
        shared_experts_config,
    )
