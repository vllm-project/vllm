# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-expert capture on the monolithic (fused router + experts) MoE path.

A monolithic kernel that `supports_routing_replay_capture` must write, into
the leading rows of `routing_replay_out`, exactly the experts it routed each
token to. The router logits plant an unambiguous top-k per token, so the test
checks the reported experts against that answer and not just their range.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxint4_moe import (
    TrtLlmMxint4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter,
)
from vllm.platforms import current_platform

try:
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
except ImportError:
    pytest.skip("flashinfer not available", allow_module_level=True)

if not has_flashinfer_trtllm_fused_moe() or not current_platform.is_cuda():
    pytest.skip(
        "Requires FlashInfer TRT-LLM fused MoE on CUDA",
        allow_module_level=True,
    )

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "TRT-LLM fused MoE kernels require SM100+",
        allow_module_level=True,
    )

NUM_EXPERTS = 32
NUM_GROUPS = 4  # DeepSeekV3 grouped routing
TOPK_GROUPS = 2
HIDDEN = 1024
INTERMEDIATE = 1024
DEVICE = torch.device("cuda:0")

# A kernel under test, and its call: (router_logits, routing_replay_out) -> None.
Kernel = tuple[FusedMoEExpertsMonolithic, Callable[[torch.Tensor, torch.Tensor], None]]


def _planted_routing(num_tokens: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Router logits with one unambiguous answer under every routing method.

    Each token gives +4 to `top_k` experts drawn from at most two expert
    groups and -4 to the rest, so softmax top-k, sigmoid top-k and DeepSeekV3
    grouped top-k all pick the same experts, with margins no rounding can flip.
    Returns the logits and the planted experts of each token, sorted.
    """
    group_size = NUM_EXPERTS // NUM_GROUPS
    planted = []
    for _ in range(num_tokens):
        groups = torch.randperm(NUM_GROUPS)[:TOPK_GROUPS]
        candidates = (groups[:, None] * group_size + torch.arange(group_size)).flatten()
        planted.append(candidates[torch.randperm(len(candidates))[:top_k]])
    experts = torch.stack(planted)
    logits = torch.full((num_tokens, NUM_EXPERTS), -4.0).scatter_(1, experts, 4.0)
    return logits.to(DEVICE), experts.sort(dim=-1).values.to(DEVICE)


def _routing_kwargs(routing_method: RoutingMethodType) -> dict:
    if routing_method != RoutingMethodType.DeepSeekV3:
        return {}
    return dict(
        num_expert_group=NUM_GROUPS,
        topk_group=TOPK_GROUPS,
        e_score_correction_bias=torch.zeros(
            NUM_EXPERTS, device=DEVICE, dtype=torch.bfloat16
        ),
        routed_scaling_factor=1.0,
    )


def _moe_config(top_k: int, routing_method: RoutingMethodType) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=top_k,
        hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        device=DEVICE,
        routing_method=routing_method,
        max_num_tokens=16,
    )


def _bf16_kernel(top_k: int, routing_method: RoutingMethodType) -> Kernel:
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import convert_to_block_layout

    def block_major_k(w: torch.Tensor) -> torch.Tensor:
        """(E, M, K) -> the BlockMajorK layout `trtllm_bf16_moe` expects."""
        return torch.stack(
            [
                convert_to_block_layout(shuffle_matrix_a(e.view(torch.uint8), 64), 128)
                for e in w
            ]
        ).view(torch.bfloat16)

    experts = TrtLlmBf16ExpertsMonolithic(
        moe_config=_moe_config(top_k, routing_method),
        quant_config=FusedMoEQuantConfig.make(
            quant_dtype=None,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
        ),
    )
    w13 = block_major_k(
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=DEVICE).bfloat16()
        * 0.1
    )
    w2 = block_major_k(
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device=DEVICE).bfloat16() * 0.1
    )

    def forward(router_logits, routing_replay_out):
        hidden_states = torch.randn(len(router_logits), HIDDEN, device=DEVICE) * 0.1
        experts.apply(
            hidden_states=hidden_states.bfloat16(),
            w1=w13,
            w2=w2,
            router_logits=router_logits,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            a1q_scale=None,
            apply_router_weight_on_input=False,
            routing_replay_out=routing_replay_out,
            **_routing_kwargs(routing_method),
        )

    return experts, forward


def _fp8_block_kernel(top_k: int, routing_method: RoutingMethodType) -> Kernel:
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
    )

    block = 128
    w13, w2 = _shuffle_deepseek_fp8_moe_weights(
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=DEVICE).to(
            torch.float8_e4m3fn
        ),
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device=DEVICE).to(
            torch.float8_e4m3fn
        ),
    )
    experts = TrtLlmFp8ExpertsMonolithic(
        moe_config=_moe_config(top_k, routing_method),
        quant_config=fp8_w8a8_moe_quant_config(
            w1_scale=torch.ones(
                NUM_EXPERTS, 2 * INTERMEDIATE // block, HIDDEN // block, device=DEVICE
            ),
            w2_scale=torch.ones(
                NUM_EXPERTS, HIDDEN // block, INTERMEDIATE // block, device=DEVICE
            ),
            block_shape=[block, block],
            per_act_token_quant=False,
        ),
    )

    def forward(router_logits, routing_replay_out):
        num_tokens = len(router_logits)
        hidden_states = torch.randn(num_tokens, HIDDEN, device=DEVICE) * 0.1
        experts.apply(
            hidden_states=hidden_states.to(torch.float8_e4m3fn),
            w1=w13,
            w2=w2,
            router_logits=router_logits,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            # Per-block activation scales, (num_tokens, hidden / 128).
            a1q_scale=torch.ones(num_tokens, HIDDEN // block, device=DEVICE),
            apply_router_weight_on_input=False,
            routing_replay_out=routing_replay_out,
            **_routing_kwargs(routing_method),
        )

    return experts, forward


def _nvfp4_weights(
    w: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed FP4 values and FP8 per-16 block scales, `(E, M, K/16)`."""
    from flashinfer import fp4_quantize

    q, scale = fp4_quantize(
        w, global_scale, 16, sf_use_ue8m0=False, is_sf_swizzled_layout=False
    )
    return q, scale.view(torch.float8_e4m3fn).reshape(*w.shape[:2], -1)


def _nvfp4_kernel(top_k: int, routing_method: RoutingMethodType) -> Kernel:
    from flashinfer import fp4_quantize

    block = 16
    global_scale = torch.tensor(1.0, device=DEVICE)
    w13, w13_scale = _nvfp4_weights(
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=DEVICE).bfloat16(),
        global_scale,
    )
    w2, w2_scale = _nvfp4_weights(
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device=DEVICE).bfloat16(),
        global_scale,
    )
    ones = torch.ones(NUM_EXPERTS, device=DEVICE)
    experts = TrtLlmNvFp4ExpertsMonolithic(
        moe_config=_moe_config(top_k, routing_method),
        quant_config=FusedMoEQuantConfig.make(
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            g1_alphas=ones,
            g2_alphas=ones,
            a1_gscale=global_scale,
            a2_gscale=ones,
        ),
    )

    def forward(router_logits, routing_replay_out):
        hidden_states = torch.randn(len(router_logits), HIDDEN, device=DEVICE) * 0.1
        # apply() reinterprets the packed uint8 activation scales itself.
        hidden_q, hidden_scale = fp4_quantize(
            hidden_states.bfloat16(),
            global_scale,
            block,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        experts.apply(
            hidden_states=hidden_q,
            w1=w13,
            w2=w2,
            router_logits=router_logits,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            a1q_scale=hidden_scale,
            apply_router_weight_on_input=False,
            routing_replay_out=routing_replay_out,
            **_routing_kwargs(routing_method),
        )

    return experts, forward


KERNELS = {"bf16": _bf16_kernel, "fp8_block": _fp8_block_kernel, "nvfp4": _nvfp4_kernel}


@pytest.mark.parametrize("kernel", list(KERNELS))
@pytest.mark.parametrize(
    "routing_method", [RoutingMethodType.Renormalize, RoutingMethodType.DeepSeekV3]
)
@pytest.mark.parametrize("simulated", [False, True])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("num_tokens", [2, 7, 16])
def test_routing_replay_reports_the_routed_experts(
    kernel: str,
    routing_method: RoutingMethodType,
    top_k: int,
    num_tokens: int,
    simulated: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    experts, forward = KERNELS[kernel](top_k, routing_method)
    assert experts.supports_routing_replay_capture()
    router_logits, planted = _planted_routing(num_tokens, top_k)
    # One spare row: the kernel must write the batch's rows and nothing else.
    routing_replay_out = torch.full(
        (num_tokens + 1, top_k), -1, dtype=torch.int16, device=DEVICE
    )

    if simulated:
        import vllm.envs as envs

        monkeypatch.setitem(
            envs.environment_variables,
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
            lambda: "uniform_random",
        )
        router = RoutingSimulatorRouter(top_k, NUM_EXPERTS, routing_method)
        router_logits = router.prepare_monolithic_logits(router_logits)

    forward(router_logits, routing_replay_out)

    reported = routing_replay_out[:num_tokens].long().sort(dim=-1).values
    if simulated:
        assert not torch.equal(reported, planted), (reported, planted)
        assert (reported >= 0).all()
        assert (reported < NUM_EXPERTS).all()
        assert (reported[:, 1:] != reported[:, :-1]).all()
    else:
        assert torch.equal(reported, planted), (reported, planted)
    assert (routing_replay_out[num_tokens:] == -1).all()


PARTIAL_KERNELS: dict[str, type[FusedMoEExpertsMonolithic]] = {
    "bf16": TrtLlmBf16ExpertsMonolithic,
    "fp8": TrtLlmFp8ExpertsMonolithic,
    "nvfp4": TrtLlmNvFp4ExpertsMonolithic,
    "mxfp4": TrtLlmMxfp4ExpertsMonolithic,
    "mxint4": TrtLlmMxint4ExpertsMonolithic,
}


def _partial_experts(kernel, config):
    """Identical sparse MLPs make local-range filtering exactly checkable."""
    count = config.num_local_experts
    w13 = torch.zeros(
        count, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=DEVICE
    )
    w2 = torch.zeros(count, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=DEVICE)
    w13[:, 0, 0] = 16
    w13[:, INTERMEDIATE, 0] = 1
    w2[:, :, 0] = 1 / 16
    if kernel == "bf16":
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            convert_moe_weights_to_flashinfer_trtllm_block_layout,
        )

        convert_moe_weights_to_flashinfer_trtllm_block_layout({}, w13, w2)
        quant = FusedMoEQuantConfig.make(quant_dtype=None)
    elif kernel == "nvfp4":
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
            prepare_static_weights_for_trtllm_fp4_moe,
            reorder_w1w3_to_w3w1,
        )

        global_scale = torch.tensor(1.0, device=DEVICE)
        w13, s13 = _nvfp4_weights(w13, global_scale)
        w2, s2 = _nvfp4_weights(w2, global_scale)
        w13, s13 = reorder_w1w3_to_w3w1(w13, s13)
        w13, s13, w2, s2 = prepare_static_weights_for_trtllm_fp4_moe(
            w13, w2, s13, s2, HIDDEN, INTERMEDIATE, count, True
        )
        ones = torch.ones(count, device=DEVICE)
        quant = FusedMoEQuantConfig.make(
            quant_dtype="nvfp4",
            w1_scale=s13,
            w2_scale=s2,
            g1_alphas=ones,
            g2_alphas=ones,
            a1_gscale=global_scale,
            a2_gscale=ones,
        )
    elif kernel == "fp8":
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            _shuffle_deepseek_fp8_moe_weights,
        )

        w13, w2 = _shuffle_deepseek_fp8_moe_weights(
            w13.to(torch.float8_e4m3fn), w2.to(torch.float8_e4m3fn)
        )
        quant = fp8_w8a8_moe_quant_config(
            w1_scale=torch.ones(
                count, 2 * INTERMEDIATE // 128, HIDDEN // 128, device=DEVICE
            ),
            w2_scale=torch.ones(
                count, HIDDEN // 128, INTERMEDIATE // 128, device=DEVICE
            ),
            block_shape=[128, 128],
        )
    elif kernel == "mxfp4":
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            Mxfp4MoeBackend,
            convert_weight_to_mxfp4_moe_kernel_format,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            mxfp4_quantize,
        )

        w13, s13 = mxfp4_quantize(w13)
        w2, s2 = mxfp4_quantize(w2)
        w13, w2, s13, s2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
            torch.nn.Module(),
            w13,
            w2,
            s13,
            s2,
            _cache_permute_indices={},
        )
        quant = mxfp4_w4a16_moe_quant_config(s13, s2)
    else:
        from tests.kernels.moe.test_marlin_vs_trtllm_mxint4 import (
            mxint4_quantize_moe_weights,
        )
        from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (  # noqa: E501
            prepare_static_weights_for_trtllm_mxint4_moe,
        )

        w13, s13 = mxint4_quantize_moe_weights(w13)
        w2, s2 = mxint4_quantize_moe_weights(w2)
        packed = prepare_static_weights_for_trtllm_mxint4_moe(w13, s13, w2, s2)
        w13, w2 = packed["gemm1_weights"], packed["gemm2_weights"]
        quant = FusedMoEQuantConfig.make(
            quant_dtype=None,
            weight_dtype="int4",
            w1_scale=packed["gemm1_scales"],
            w2_scale=packed["gemm2_scales"],
        )
    return PARTIAL_KERNELS[kernel](config, quant), w13, w2


@pytest.mark.parametrize("kernel", list(PARTIAL_KERNELS))
@pytest.mark.parametrize(
    "routing_method",
    [RoutingMethodType.Renormalize, RoutingMethodType.RenormalizeNaive],
)
def test_partial_execution_preserves_weights_and_captures_nonlocal_ids(
    kernel, routing_method
):
    """Only local weights exist; nonlocal routes keep their original weight."""
    # A nonzero offset distinguishes global and local expert IDs.
    ep_rank = 1
    config = _moe_config(2, routing_method)
    config = replace(
        config,
        num_local_experts=NUM_EXPERTS // 2,
        router_logits_dtype=torch.bfloat16,
        moe_parallel_config=replace(
            config.moe_parallel_config, ep_size=2, ep_rank=ep_rank, use_ep=True
        ),
    )
    assert PARTIAL_KERNELS[kernel].supports_partial_expert_execution(config)
    experts, w13, w2 = _partial_experts(kernel, config)
    local = ep_rank * config.num_local_experts
    remote = (1 - ep_rank) * config.num_local_experts
    pairs = torch.tensor(
        [
            [local, local + 1],
            [local, remote],
            [remote, remote + 1],
            [remote, local + 1],
        ],
        device=DEVICE,
    )
    logits = torch.full((4, NUM_EXPERTS), -1000.0, dtype=torch.bfloat16, device=DEVICE)
    logits.scatter_(1, pairs, 0)
    reference_logits = torch.full_like(logits, -1000)
    reference_logits[:, local : local + 2] = 0
    inputs = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    inputs[:, 0] = 1
    if kernel == "fp8":
        inputs = inputs.to(torch.float8_e4m3fn)
        scales = torch.ones(4, HIDDEN // 128, device=DEVICE)
    elif kernel == "nvfp4":
        from flashinfer import fp4_quantize

        input_scale = experts.quant_config.a1_gscale
        assert input_scale is not None
        inputs, scales = fp4_quantize(
            inputs,
            input_scale,
            16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
    else:
        scales = None
    ids = torch.empty(4, 2, dtype=torch.int16, device=DEVICE)

    def forward(scores):
        return experts.apply(
            hidden_states=inputs,
            w1=w13,
            w2=w2,
            router_logits=scores,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            a1q_scale=scales,
            apply_router_weight_on_input=False,
            routing_replay_out=ids,
        )

    reference = forward(reference_logits).clone()
    assert reference.abs().min() > 0
    expected = (
        reference
        * torch.tensor([1, 0.5, 0, 0.5], dtype=reference.dtype, device=DEVICE)[:, None]
    )
    torch.testing.assert_close(forward(logits), expected, atol=0, rtol=0)
    torch.testing.assert_close(ids.long().sort(-1).values, pairs.sort(-1).values)
    torch.testing.assert_close(
        forward(logits[2:3].expand_as(logits).contiguous()),
        torch.zeros_like(reference),
        atol=0,
        rtol=0,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward(logits)
    logits.copy_(logits.flip(0))
    graph.replay()
    torch.testing.assert_close(output, expected.flip(0), atol=0, rtol=0)
    torch.testing.assert_close(
        ids.long().sort(-1).values, pairs.flip(0).sort(-1).values
    )
