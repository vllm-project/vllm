# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import GraniteConfig

from vllm.model_executor.models.granite import granite_layer_attn_params
from vllm.model_executor.models.granitemoe import granitemoe_split_expert_weights

from ...utils import check_logprobs_close

MODELS = [
    # TODO(sang): Sliding window should be tested separately.
    "ibm/PowerLM-3b",
    "ibm/PowerMoE-3b",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.cpu_model
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


def test_granite_swa_features_are_off_without_swa_config():
    """A plain Granite config must not pick up sliding windows or sinks."""
    config = GraniteConfig(num_hidden_layers=4)
    theta = config.rope_parameters["rope_theta"]

    assert [granite_layer_attn_params(config, i) for i in range(4)] == [
        (None, theta, False)
    ] * 4


@pytest.mark.parametrize(
    "attention_sinks, expected_sink", [(None, True), (False, False)]
)
def test_granite_swa_features_resolve_per_layer(attention_sinks, expected_sink):
    """`layer_types`/`layer_rope_theta` apply per layer; theta 0 means NoPE."""
    kwargs = {} if attention_sinks is None else {"attention_sinks": attention_sinks}
    config = GraniteConfig(
        num_hidden_layers=3,
        sliding_window=128,
        layer_types=["full_attention", "sliding_attention", "sliding_attention"],
        layer_rope_theta=[10000.0, 0.0, 1000000.0],
        **kwargs,
    )

    assert [granite_layer_attn_params(config, i) for i in range(3)] == [
        (None, 10000.0, expected_sink),
        (128, 0.0, expected_sink),
        (128, 1000000.0, expected_sink),
    ]


def test_granitemoe_expert_weights_accept_both_checkpoint_spellings():
    """The legacy and current MoE tensor names differ but the layouts match."""
    num_experts, hidden, inter = 3, 8, 4
    gate_up = torch.randn(num_experts, 2 * inter, hidden)
    down = torch.randn(num_experts, hidden, inter)
    router = torch.randn(num_experts, hidden)

    def weights(gate_up_name, down_name, router_name):
        return [
            (f"model.layers.0.block_sparse_moe.{gate_up_name}", gate_up),
            (f"model.layers.0.block_sparse_moe.{down_name}", down),
            (f"model.layers.0.block_sparse_moe.{router_name}", router),
            # Must pass through: the shared MLP reuses the legacy expert names.
            ("model.layers.0.shared_mlp.input_linear.weight", down[0]),
        ]

    legacy = dict(
        granitemoe_split_expert_weights(
            weights(
                "input_linear.weight", "output_linear.weight", "router.layer.weight"
            )
        )
    )
    current = dict(
        granitemoe_split_expert_weights(
            weights("experts.gate_up_proj", "experts.down_proj", "router.weight")
        )
    )

    assert legacy.keys() == current.keys()
    for name, weight in legacy.items():
        assert torch.equal(weight, current[name]), name

    prefix = "model.layers.0.block_sparse_moe"
    assert torch.equal(legacy[f"{prefix}.gate.weight"], router)
    assert torch.equal(legacy["model.layers.0.shared_mlp.input_linear.weight"], down[0])
    for e in range(num_experts):
        assert torch.equal(
            legacy[f"{prefix}.experts.{e}.w1.weight"], gate_up[e][:inter]
        )
        assert torch.equal(
            legacy[f"{prefix}.experts.{e}.w3.weight"], gate_up[e][inter:]
        )
        assert torch.equal(legacy[f"{prefix}.experts.{e}.w2.weight"], down[e])
