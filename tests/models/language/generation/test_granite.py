# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import GraniteConfig

from vllm.model_executor.models.granite import granite_layer_attn_params

from ...utils import check_logprobs_close, check_transformers_version

# model -> minimum transformers version, or None if unconstrained
MODELS = {
    # TODO(sang): Sliding window should be tested separately.
    "ibm/PowerLM-3b": None,
    "ibm/PowerMoE-3b": None,
    "ibm-granite/granite-swash-2b": "5.15.1",
    "ibm-granite/granite-swash-3b-a600m": "5.15.1",
}


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
    check_transformers_version(model, min_transformers_version=MODELS[model])

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
