# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE layer.

Run `pytest tests/kernels/test_moe_layer.py`.
"""
from typing import Callable, Optional

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo, parallel_launch_with_config)
from tests.kernels.moe.utils import (make_naive_shared_experts,
                                     make_test_weights, moe_quantize_weights)
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
from vllm.platforms import current_platform

SHAPE_COMBOS = [
    # TODO: figure out why this fails, seems to be test problem
    #(1, 128, 128),
    #    (2, 128, 512),
    #    (3, 1024, 2048),
    #    (4, 128, 128),
    (32, 1024, 512),
    #    (45, 512, 2048),
    #    (64, 1024, 512),
    (222, 2048, 1024),
    (256, 1408, 2048),
]

NUM_EXPERTS = [8, 64]
TOP_KS = [1, 6]

# dp_size, tp_size, use_ep
PARALLEL_COMBOS = [
    [1, 1, False],
    [1, 2, False],
#    [1, 4, False],
#    [2, 1, True],
#    [2, 2, True],
#    [4, 1, True],
]

BACKENDS = [
    None,
    "naive",
    "pplx",
    "deepep_low_latency",
    "deepep_high_throughput",
]

QUANT_METHODS = [
    None,
    "fp8",
    #"modelopt",
    #"compressed-tensors",
]

def rank_chunk(num: int, r: int, w: int) -> int:
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(t: torch.Tensor, r: int, w: int) -> torch.Tensor:
    chunk = rank_chunk(t.shape[0], r, w)
    return t[(r * chunk):(r + 1) * chunk]


def maybe_chunk_by_rank(t: Optional[torch.Tensor], r: int,
                        w: int) -> Optional[torch.Tensor]:
    if t is not None:
        return chunk_by_rank(t, r, w)
    else:
        return t


def chunk_scales_by_rank(t: Optional[torch.Tensor], r: int,
                         w: int) -> Optional[torch.Tensor]:
    if t is not None and t.numel() > 1:
        chunk = rank_chunk(t.shape[0], r, w)
        return t[(r * chunk):(r + 1) * chunk]
    else:
        return t


def chunk_scales(t: Optional[torch.Tensor], start: int,
                 end: int) -> Optional[torch.Tensor]:
    if t is not None and t.numel() > 1:
        return t[start:end]
    else:
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
        w1q = w1q.transpose(0, 1)
        w2q = w2q.transpose(0, 1)
        w1s = w1s.transpose(0, 1)
        w2s = w2s.transpose(0, 1)
    elif quantization == "modelopt" or quantization == "compressed_tensors":
        assert False, "TBD"

    return quant_config, w1q, w1s, w2q, w2s


def make_fused_moe_layer(
    quantization: Optional[str],
    use_ep: bool,
    hidden_size: int,
    intermediate_size: int,
    params_dtype: torch.dtype,
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
):
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
        params_dtype=params_dtype,
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

    if use_ep:
        assert layer.quant_method is not None
        layer.quant_method.init_prepare_finalize(layer)

    return layer


def make_fake_moe_layer(
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
    enable_eplb: bool = False,  # for now
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
) -> Callable:

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
    logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    m: int,
    n: int,
    k: int,
    top_k: int,
    quantization: Optional[str],
    shared_experts: Optional[torch.nn.Module],
):

    #    if use_shared_experts:
    #        # Note: this config is only needed for the non-naive shared experts.
    #        new_vllm_config = copy.deepcopy(vllm_config)
    #        new_vllm_config.parallel_config.data_parallel_size = pgi.world_size
    #        new_vllm_config.parallel_config.enable_expert_parallel = True
    #        _set_vllm_config(new_vllm_config, pgi.world_size, pgi.rank,
    #                         pgi.local_rank)

    print(f"PCONF {vllm_config.parallel_config}")

    world_size = tp_size * dp_size

    current_platform.seed_everything(7)

    in_dtype = hidden_states.dtype

    # TODO: chunk up weights, activations, scores
    device = torch.cuda.current_device()
    w1 = w1.to(device)
    w2 = w2.to(device)
    hidden_states = hidden_states.to(device)
    logits = logits.to(device)
    baseline_output = baseline_output.to(device)

    moe_layer = make_fused_moe_layer(
        quantization=quantization,
        use_ep=ep_size > 1,
        hidden_size=k,
        intermediate_size=n,
        params_dtype=in_dtype,
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

    #dtype_str = get_config_dtype_str(in_dtype, use_fp8_w8a8=True)
    #moe_config = get_default_config(m, num_experts, n, k, top_k, dtype_str, False)

    # output should be completely reduced at this point
    num_tokens = m
    num_tokens_across_dp = torch.tensor([num_tokens] * world_size,
                                        device=torch.cuda.current_device(),
                                        dtype=torch.int)

    # Make moe_forward happy
    vllm_config.compilation_config.static_forward_context[
        "test_layer"] = moe_layer

    with set_forward_context(
            None,
            vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
    ):
        output = moe_layer(hidden_states, logits)

    torch.testing.assert_close(baseline_output, output, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("m, n, k", SHAPE_COMBOS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("quantization", QUANT_METHODS)
@pytest.mark.parametrize("dp_size, tp_size, use_ep", PARALLEL_COMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_shared_experts", [False])  #[False, True])
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
    backend: Optional[str],
    use_shared_experts: bool,
):
    current_platform.seed_everything(7)
    world_size = tp_size * dp_size

    test_env = dict()
    test_env["VLLM_USE_DEEP_GEMM"] = "1"
    if backend is not None:
        test_env["VLLM_ALL2ALL_BACKEND"] = backend

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = dp_size
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    vllm_config.parallel_config.enable_expert_parallel = use_ep

    in_dtype = torch.bfloat16

    (w1, _, _, _), (w2, _, _, _) = make_test_weights(
        num_experts,
        n,
        k,
        in_dtype=in_dtype,
    )

    if use_shared_experts:
        shared_experts = make_naive_shared_experts(
            n,
            k,
            in_dtype=in_dtype,
        )
    else:
        shared_experts = None

    baseline_layer = make_fake_moe_layer(
        w1=w1,
        w2=w2,
        top_k=top_k,
        global_num_experts=num_experts,
        renormalize=False,
        shared_experts=shared_experts,
    )

    hidden_states = torch.randn((m, k), device="cuda", dtype=in_dtype) / 10
    logits = torch.randn((m, num_experts), device="cuda", dtype=in_dtype)

    baseline_output = baseline_layer(hidden_states, logits)

    parallel_launch_with_config(
        world_size,
        _test_loop,
        vllm_config,
        test_env,
        1 if not use_ep else world_size,
        dp_size,
        tp_size,
        baseline_output,
        hidden_states,
        logits,
        w1,
        w2,
        num_experts,
        m,
        n,
        k,
        top_k,
        quantization,
        shared_experts,
    )

    # monkeypatch.setenv('RANK', "0")
    # monkeypatch.setenv('LOCAL_RANK', "0")
    # monkeypatch.setenv('WORLD_SIZE', "1")
    # monkeypatch.setenv('MASTER_ADDR', 'localhost')
    # monkeypatch.setenv('MASTER_PORT', '12345')
    # init_distributed_environment()

    # # Instantiate our and huggingface's MoE blocks
    # vllm_config.compilation_config.static_forward_context = dict()
    # with (set_current_vllm_config(vllm_config),
    #       set_forward_context(None, vllm_config)):
    #     config = MixtralConfig()
    #     hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    #     vllm_moe = MixtralMoE(
    #         num_experts=config.num_local_experts,
    #         top_k=config.num_experts_per_tok,
    #         hidden_size=config.hidden_size,
    #         intermediate_size=config.intermediate_size,
    #         params_dtype=dtype,
    #         tp_size=1,
    #         dp_size=1,
    #     ).cuda()

    #     # Load the weights
    #     vllm_moe.gate.weight.data[:] = hf_moe.gate.weight.data
    #     for i in range(config.num_local_experts):
    #         weights = (hf_moe.experts[i].w1.weight.data,
    #                    hf_moe.experts[i].w3.weight.data)
    #         vllm_moe.experts.w13_weight[i][:] = torch.cat(weights, dim=0)
    #         vllm_moe.experts.w2_weight[i][:] = hf_moe.experts[i].w2.weight.data

    #     # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    #     hf_inputs = torch.randn(
    #         (1, 64, config.hidden_size)).to(dtype).to("cuda")
    #     # vLLM uses 1D query [num_tokens, hidden_dim]
    #     vllm_inputs = hf_inputs.flatten(0, 1)

    #     # Pad the weight if moe padding is enabled
    #     if padding:
    #         vllm_moe.experts.w13_weight = Parameter(F.pad(
    #             vllm_moe.experts.w13_weight, (0, 128), "constant", 0)[...,
    #                                                                   0:-128],
    #                                                 requires_grad=False)
    #         vllm_moe.experts.w2_weight = Parameter(F.pad(
    #             vllm_moe.experts.w2_weight, (0, 128), "constant", 0)[...,
    #                                                                  0:-128],
    #                                                requires_grad=False)
    #         torch.cuda.synchronize()
    #         torch.cuda.empty_cache()

    #     # Run forward passes for both MoE blocks
    #     hf_states, _ = hf_moe.forward(hf_inputs)
    #     vllm_states = vllm_moe.forward(vllm_inputs)
