# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""routed_scaling_factor applied to the output with shared experts (#65533)."""

import pytest
import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, RoutedScaleMode
from vllm.utils.torch_utils import set_random_seed

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_TOKENS = 16
ROUTED_SCALING_FACTOR = 2.5


class SharedMLP(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.up = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE * 2, bias=False, dtype=dtype)
        self.down = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False, dtype=dtype)
        with torch.no_grad():
            self.up.weight.div_(10)
            self.down.weight.div_(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.up(x).chunk(2, dim=-1)
        return self.down(nn.functional.silu(gate) * up)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "routed_scale_mode",
    [RoutedScaleMode.OUTPUT, RoutedScaleMode.OUTPUT_FP16_GUARD],
)
def test_routed_scale_applied_to_output_with_shared_experts(
    dtype: torch.dtype,
    routed_scale_mode: RoutedScaleMode,
    dist_init,
    workspace_init,
):
    """OUTPUT gives scale * routed + shared for every dtype. OUTPUT_FP16_GUARD
    gives routed + shared / scale in FP16 (the decoder layer is expected to
    compensate) and scale * routed + shared otherwise."""
    set_random_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    shared_experts = SharedMLP(dtype)
    common = dict(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        renormalize=True,
        params_dtype=dtype,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
    )

    with set_current_vllm_config(vllm_config):
        moe = FusedMoEFactory(
            shared_experts=shared_experts,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            routed_scale_mode=routed_scale_mode,
            prefix="moe",
            **common,
        )
        routed_only = FusedMoEFactory(prefix="routed_only", **common)
        experts, routed_only_experts = moe.routed_experts, routed_only.routed_experts
        with torch.no_grad():
            experts.w13_weight.normal_().div_(10)
            experts.w2_weight.normal_().div_(10)
            routed_only_experts.w13_weight.copy_(experts.w13_weight)
            routed_only_experts.w2_weight.copy_(experts.w2_weight)
        for e in (experts, routed_only_experts):
            e.quant_method.process_weights_after_loading(e)

        hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=dtype)
        router_logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=dtype)
        with set_forward_context(None, vllm_config, num_tokens=NUM_TOKENS):
            actual = moe(hidden_states, router_logits)
            routed = routed_only(hidden_states, router_logits)
            shared = shared_experts(hidden_states)

    if (
        routed_scale_mode is RoutedScaleMode.OUTPUT_FP16_GUARD
        and dtype == torch.float16
    ):
        expected = routed + shared / ROUTED_SCALING_FACTOR
    else:
        expected = routed * ROUTED_SCALING_FACTOR + shared
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_routed_scale_mode_scales_output():
    assert not RoutedScaleMode.ROUTER.scales_output
    assert RoutedScaleMode.OUTPUT.scales_output
    assert RoutedScaleMode.OUTPUT_FP16_GUARD.scales_output


def _mha_moe_config(model: str):
    from transformers import AXK1Config, DeepseekV2Config

    common = dict(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        moe_intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        first_k_dense_replace=0,
        torch_dtype=torch.float16,
    )
    if model == "axk1":
        return AXK1Config(mlp_layer_types=["sparse"], **common)
    return DeepseekV2Config(**common)


@pytest.mark.parametrize("model", ["deepseek_v2", "axk1"])
def test_mha_decoder_layer_does_not_divide_shared_output(
    model: str, tmp_path, dist_init, workspace_init
):
    """MHA DeepseekV2/AXK1 decoder layers skip the FP16 compensation, so their
    MoE must scale the routed output instead of dividing the shared output."""
    from vllm.config import ModelConfig
    from vllm.model_executor.models.AXK1 import AXK1DecoderLayer
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

    config = _mha_moe_config(model)
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=32,
        dtype="float16",
    )
    vllm_config = VllmConfig(model_config=model_config)
    layer_cls = AXK1DecoderLayer if model == "axk1" else DeepseekV2DecoderLayer
    with set_current_vllm_config(vllm_config):
        layer = layer_cls(
            vllm_config=vllm_config, prefix="model.layers.0", config=config
        )
    assert layer.use_mha
    assert layer.mlp.experts.routed_scale_mode is RoutedScaleMode.OUTPUT
