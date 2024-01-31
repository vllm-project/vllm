"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from vllm.model_executor.models.mixtral import MixtralMoE

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(
    hf_runner,
    vllm_runner,
    example_long_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_long_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy(example_long_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_long_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")

@torch.inference_mode
def test_mixtral_moe():
    "Make sure our Mixtral MoE implementation agrees with the one from huggingface."

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        tp_size=1,
    )

    # Load the weights
    vllm_moe.gate.linear_weights["weight"][:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.ws[i][:] = torch.cat(weights, dim=0).to("cuda")
        vllm_moe.w2s[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    inputs = torch.randn((1, 64, config.hidden_size)).to("cuda")

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(inputs)
    vllm_states = vllm_moe.forward(inputs)

    assert torch.allclose(hf_states, vllm_states, rtol=1e-3, atol=1e-3)
