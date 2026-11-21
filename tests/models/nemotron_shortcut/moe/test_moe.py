# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Shortcut's MoE layer: numerics, checkpoint loading and graph replay."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.nemotron_shortcut.nvidia.moe import NemotronShortcutMoE
from vllm.models.nemotron_shortcut.nvidia.moe.flashinfer import has_cute_dsl_options
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from ..utils import build, make_config

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton experts need CUDA"
)


@pytest.fixture(autouse=True)
def _forward_context(dist_init, workspace_init):
    with torch.inference_mode(), set_forward_context(None, get_current_vllm_config()):
        yield


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k,shared_size", [(1, None), (2, 16)])
@pytest.mark.parametrize("bias_space", ["score", "logit"])
def test_matches_a_per_expert_reference(dtype, top_k, shared_size, bias_space):
    """Selection-only bias, weighting before the down projection, FP32 sum.

    `"logit"` selects on `logits + bias` instead of `sigmoid(logits) + bias`;
    the mixing weights are the unbiased sigmoid scores either way.
    """
    config = make_config(
        num_experts_per_tok=top_k,
        moe_shared_expert_intermediate_size=shared_size,
        moe_router_bias_space=bias_space,
    )
    with set_default_torch_dtype(dtype):
        moe = build(lambda: NemotronShortcutMoE(config))
    experts = moe.experts.routed_experts
    # Exercise the soft clamp and the low-precision rounding.
    experts.w13_weight.mul_(100)
    bias = moe.gate.e_score_correction_bias
    bias.copy_(torch.linspace(-0.6, 0.6, config.n_routed_experts, device="cuda"))
    routed = torch.randn(7, config.hidden_size, device="cuda", dtype=dtype) * 3
    shared = torch.randn_like(routed)

    logits = F.linear(routed.float(), moe.gate.weight.float())
    scores = logits.sigmoid()
    ids = ((logits if bias_space == "logit" else scores) + bias).topk(top_k).indices
    if bias_space == "logit":
        # Otherwise this case would not tell the two spaces apart.
        score_ids = (scores + bias).topk(top_k).indices
        assert not torch.equal(ids.sort(-1).values, score_ids.sort(-1).values)
    weights = scores.gather(1, ids)
    if top_k > 1:
        weights /= weights.sum(-1, keepdim=True) + 1e-20
    weights *= config.routed_scaling_factor
    clamp = config.activation_clamp_value
    latent = F.linear(routed, moe.fc1_latent_proj.weight)
    expected = torch.zeros_like(latent, dtype=torch.float32)
    for expert in range(config.n_routed_experts):
        token, slot = torch.where(ids == expert)
        hidden = F.linear(latent[token], experts.w13_weight[expert])
        hidden = (clamp * torch.tanh(hidden.float() / clamp)).to(dtype)
        hidden = (hidden.relu().square().float() * weights[token, slot, None]).to(dtype)
        expected[token] += F.linear(hidden, experts.w2_weight[expert]).float()
    expected = expected.to(dtype).float()
    expected *= torch.rsqrt(
        expected.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    expected = expected.to(dtype) * moe.fc2_norm.weight
    expected = F.linear(expected, moe.fc2_latent_proj.weight)
    if shared_size is not None:
        hidden = F.linear(shared, moe.shared_experts.up_proj.weight)
        hidden = (clamp * torch.tanh(hidden.float() / clamp)).to(dtype)
        expected += F.linear(
            hidden.relu().square(), moe.shared_experts.down_proj.weight
        )

    torch.testing.assert_close(
        moe(routed, shared),
        expected,
        atol=2e-3 if dtype == torch.bfloat16 else 2e-6,
        rtol=2e-2 if dtype == torch.bfloat16 else 2e-5,
    )


def test_loads_packed_experts_and_router_bias():
    """Loading overwrites every declared checkpoint tensor, experts included."""
    moe = build(lambda: NemotronShortcutMoE(make_config()))
    weights = {name: torch.randn_like(param) for name, param in moe.named_parameters()}
    wrapper = nn.Module()
    wrapper.moe = moe
    checkpoint = {
        "moe."
        + name.replace("experts.routed_experts.w13_weight", "experts.up_proj").replace(
            "experts.routed_experts.w2_weight", "experts.down_proj"
        ): value
        for name, value in weights.items()
    }

    assert AutoWeightsLoader(wrapper).load_weights(checkpoint.items()) == {
        "moe." + name for name in weights
    }
    for name, param in moe.named_parameters():
        torch.testing.assert_close(param, weights[name], atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or not has_cute_dsl_options("routing_weights_before_fc2", "clamped_relu2_limit"),
    reason="NVFP4 experts need SM10x and FlashInfer's routing_weights_before_fc2",
)
def test_loads_nvfp4_expert_scales_without_folding_twice():
    """Fused checkpoint keys keep each expert's own global scales."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    quant = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=["*.fc1_latent_proj", "*.fc2_latent_proj"],
    )
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        moe = NemotronShortcutMoE(
            make_config(
                hidden_size=256,
                moe_latent_size=256,
                moe_intermediate_size=512,
                moe_shared_expert_intermediate_size=None,
            ),
            quant_config=quant,
            prefix="moe",
        )
    experts = moe.experts.routed_experts
    checkpoint, expected = {}, {}
    per_expert = torch.arange(1, 9, device="cuda").float()
    for projection, parameter, rows, width, byte in (
        ("up_proj", "w13", 512, 256, 17),
        ("down_proj", "w2", 256, 512, 91),
    ):
        tensors = {
            ("", "_weight"): (per_expert[:, None, None] + byte)
            .expand(8, rows, width // 2)
            .to(torch.uint8)
            .contiguous(),
            ("_scale", "_weight_scale"): (per_expert[:, None, None] / 4)
            .expand(8, rows, width // 16)
            .to(torch.float8_e4m3fn)
            .contiguous(),
            ("_scale_2", "_weight_scale_2"): per_expert / 128,
            ("_input_scale", "_input_scale"): per_expert / 1024,
        }
        for (source, target), tensor in tensors.items():
            checkpoint[f"moe.experts.{projection}{source}"] = tensor
            expected[parameter + target] = tensor.reshape_as(
                getattr(experts, parameter + target)
            )
    wrapper = nn.Module()
    wrapper.moe = moe

    assert AutoWeightsLoader(wrapper).load_weights(checkpoint.items()) == {
        "moe.experts.routed_experts." + name for name in expected
    }
    for name, tensor in expected.items():
        actual = getattr(experts, name)
        assert actual.dtype == tensor.dtype
        torch.testing.assert_close(actual.float(), tensor.float(), atol=0, rtol=0)
    experts.quant_method.process_weights_after_loading(experts)
    assert experts.quant_method.moe_kernel is not None
    kernel = experts.quant_method.moe_kernel.fused_experts
    # The ModelOpt loader folds the shared activation scale exactly once.
    for alphas, weight, activation in (
        (kernel.g1_alphas, "w13_weight_scale_2", "w13_input_scale"),
        (kernel.g2_alphas, "w2_weight_scale_2", "w2_input_scale"),
    ):
        assert alphas is not None
        torch.testing.assert_close(
            alphas.flatten(), expected[weight].flatten() * expected[activation].max()
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_graph_replay_routes_the_new_inputs(compiled):
    """A replayed capture routes the new inputs, not the captured ones."""
    moe = build(lambda: NemotronShortcutMoE(make_config()))
    forward = torch.compile(moe, fullgraph=True) if compiled else moe
    moe.gate.e_score_correction_bias.zero_()
    # More than 32 tokens: torch.compile cannot trace GateLinear's CuTe-DSL tier
    # for at most 32 tokens, which vLLM never traces (it compiles at
    # max_num_batched_tokens).
    routed = torch.randn(48, 32, device="cuda")
    shared = torch.randn_like(routed)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            forward(routed, shared)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward(routed, shared)
    graph.replay()
    first = output.clone()
    routed.neg_()  # Top-2 of eight logits becomes the bottom-2.
    shared.add_(1)
    graph.replay()

    torch.testing.assert_close(output, moe(routed, shared), atol=2e-6, rtol=2e-5)
    assert not torch.equal(output, first)
